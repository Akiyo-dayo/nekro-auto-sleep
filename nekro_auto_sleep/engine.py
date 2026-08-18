"""Pure state-machine, wake protocol, settlement, and idle-sleep-back logic.

Domain layer — no host imports. All time parameters are aware UTC datetimes.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Callable, Literal

from zoneinfo import ZoneInfo

from .models import (
    ChatSleepState,
    ConfigSnapshot,
    NotificationStatus,
    PendingWakeOffer,
    SleepCycle,
    SleepSegment,
    SleepStatus,
    TimerInterval,
    WakeAttempt,
)
from .schedule import (
    compute_cycle_boundaries,
    create_config_snapshot,
    generate_cycle_id,
    generate_quality_seed,
    is_near_wake,
    pick_wake_time,
)

logger = logging.getLogger("nekro_auto_sleep.engine")


# ---------------------------------------------------------------------------
# Action results returned to the caller (the plugin __init__)
# ---------------------------------------------------------------------------


class SleepAction:
    """Base class for engine action results."""


class ActionNone(SleepAction):
    """No external action needed."""


class ActionSendFixed(SleepAction):
    """Send a fixed message (not through LLM), record=False."""

    def __init__(self, text: str, block_mode: Literal["preserve", "strict"] = "preserve"):
        self.text = text
        self.block_mode = block_mode


class ActionForceWake(SleepAction):
    """The call was answered (or was urgent) — FORCE_TRIGGER this message."""

    def __init__(self, reason: str = "confirmed"):
        self.reason = reason


class ActionSendWakeNotice(SleepAction):
    """Send natural wake-up notice."""

    def __init__(self, text: str):
        self.text = text


class ActionSendResumeSleep(SleepAction):
    """Send resume-sleep confirmation."""

    def __init__(self, text: str):
        self.text = text


# ---------------------------------------------------------------------------
# Cycle creation
# ---------------------------------------------------------------------------


def create_sleep_cycle(
    chat_key: str,
    sleep_date_local: str,
    config_snapshot: ConfigSnapshot,
) -> SleepCycle:
    """Create a new SleepCycle with pre-computed random wake time."""
    from datetime import date as date_type

    sd = date_type.fromisoformat(sleep_date_local)
    tz = ZoneInfo(config_snapshot.timezone)

    sleep_at, wake_start, wake_end = compute_cycle_boundaries(
        sd,
        tz,
        config_snapshot.sleep_time,
        config_snapshot.wake_time_start,
        config_snapshot.wake_time_end,
    )

    planned_wake = pick_wake_time(
        chat_key,
        sd,
        wake_start,
        wake_end,
        config_snapshot.wake_random_step_minutes,
    )

    return SleepCycle(
        cycle_id=generate_cycle_id(chat_key, sd),
        sleep_date=sleep_date_local,
        timezone=config_snapshot.timezone,
        sleep_at=sleep_at,
        planned_wake_at=planned_wake,
        config_snapshot=config_snapshot,
        quality_seed=generate_quality_seed(chat_key, sd),
        sleep_segments=[],
        wake_attempts=[],
        timer_intervals=[],
    )


# ---------------------------------------------------------------------------
# Sleep segment management
# ---------------------------------------------------------------------------


def open_sleep_segment(state: ChatSleepState, now_utc: datetime) -> ChatSleepState:
    """Open a new sleep segment. Closes any unclosed segment first."""
    if state.cycle is None:
        return state
    segments = list(state.cycle.sleep_segments)
    for i, seg in enumerate(segments):
        if seg.close_at is None:
            segments[i] = seg.model_copy(update={"close_at": now_utc})
    segments.append(SleepSegment(open_at=now_utc))
    state.cycle = state.cycle.model_copy(update={"sleep_segments": segments})
    return state


def close_sleep_segment(state: ChatSleepState, now_utc: datetime) -> ChatSleepState:
    """Close the current open sleep segment."""
    if state.cycle is None:
        return state
    segments = list(state.cycle.sleep_segments)
    for i, seg in enumerate(segments):
        if seg.close_at is None:
            segments[i] = seg.model_copy(update={"close_at": now_utc})
    state.cycle = state.cycle.model_copy(update={"sleep_segments": segments})
    return state


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


def transition_to_sleep(
    state: ChatSleepState,
    now_utc: datetime,
    config_snapshot: ConfigSnapshot,
    sleep_date_local: str | None = None,
    segment_open_at: datetime | None = None,
) -> ChatSleepState:
    """AWAKE -> ASLEEP: create cycle and open segment.

    `sleep_date_local` / `segment_open_at` exist for boot reconciliation: when the
    process was down across the bedtime boundary, the cycle belongs to the night
    that already started, and the sleep segment has to be backdated to the real
    bedtime — otherwise the morning report claims the bot slept five minutes.
    """
    from datetime import date as date_type

    tz = ZoneInfo(config_snapshot.timezone)
    if sleep_date_local is None:
        sleep_date_local = now_utc.astimezone(tz).date().isoformat()

    cycle = create_sleep_cycle(state.chat_key, sleep_date_local, config_snapshot)
    state = state.model_copy(
        update={
            "status": SleepStatus.ASLEEP,
            "cycle": cycle,
            "pending_offer": None,
            "offers_sent_tonight": 0,
            "last_offer_at": None,
            "snooze_until": None,
            "idle_sleep_deadline": None,
        }
    )
    state = open_sleep_segment(state, segment_open_at or now_utc)
    return state


def transition_to_awake_early(
    state: ChatSleepState,
    now_utc: datetime,
    user_id: str,
    idle_minutes: int,
    reason: str = "confirmed",
) -> ChatSleepState:
    """ASLEEP -> AWAKE_EARLY: the wake-up question was answered yes."""
    state = close_sleep_segment(state, now_utc)
    deadline = now_utc + timedelta(minutes=idle_minutes)
    state = state.model_copy(
        update={
            "status": SleepStatus.AWAKE_EARLY,
            "pending_offer": None,
            "idle_sleep_deadline": deadline,
            "woken_at": now_utc,
            "woken_by": user_id,
            "woken_reason": reason,
        }
    )
    return state


def transition_to_awake(
    state: ChatSleepState,
    now_utc: datetime,
) -> ChatSleepState:
    """-> AWAKE: natural wake or settlement."""
    state = close_sleep_segment(state, now_utc)
    if state.cycle and state.cycle.settled_at is None:
        state.cycle = state.cycle.model_copy(update={"settled_at": now_utc})
    if state.status == SleepStatus.AWAKE_EARLY and state.cycle:
        state.cycle = state.cycle.model_copy(update={"ended_while_early_awake": True})
    state = state.model_copy(
        update={
            "status": SleepStatus.AWAKE,
            "idle_sleep_deadline": None,
            "pending_offer": None,
            "woken_at": None,
            "woken_by": None,
            "woken_reason": None,
        }
    )
    return state


def transition_resume_sleep(
    state: ChatSleepState,
    now_utc: datetime,
) -> ChatSleepState:
    """AWAKE_EARLY -> ASLEEP: bot or idle timeout goes back to sleep."""
    state = state.model_copy(
        update={
            "status": SleepStatus.ASLEEP,
            "idle_sleep_deadline": None,
            "woken_at": None,
            "woken_by": None,
            "woken_reason": None,
        }
    )
    state = open_sleep_segment(state, now_utc)
    return state


# ---------------------------------------------------------------------------
# Wake protocol
# ---------------------------------------------------------------------------


AnswerIntent = Literal["yes", "no", "unclear"]

# Trailing particles and punctuation that carry no meaning for a yes/no answer.
# Stripping them is what turns 好的 / 要啊 / 嗯~ into the bare keyword.
_TRAILING_NOISE = "的啊呀吧了哦喔噢呢嘛呗嘞哒~～!！?？。.,，、;；:：… "
_PUNCTUATION = "!！?？。.,，、;；:：…()（）\"“”'‘’"


def _fold(text: str) -> str:
    """Case/width folding and punctuation removal, whitespace collapsed.

    Whitespace is collapsed rather than deleted: ASCII keywords are matched on
    word boundaries, and squashing "wake up" into "wakeup" would break that.
    """
    if not text:
        return ""
    folded = []
    for ch in text.strip():
        code = ord(ch)
        if code == 0x3000:  # ideographic space
            ch = " "
        elif 0xFF01 <= code <= 0xFF5E:  # full-width ASCII
            ch = chr(code - 0xFEE0)
        folded.append(ch)
    cleaned = "".join(c for c in folded if c not in _PUNCTUATION).lower()
    return " ".join(cleaned.split())


def _normalize(text: str) -> str:
    """Fold a reply and trim the trailing particles a bare answer picks up."""
    return _fold(text).rstrip(_TRAILING_NOISE)


def _is_ascii(word: str) -> bool:
    return all(ord(c) < 128 for c in word)


def _matches_whole_answer(candidate: str, keyword: str) -> bool:
    """Whether the whole reply *is* this keyword (allowing 嗯嗯 / 好好)."""
    if not candidate or not keyword:
        return False
    if candidate == keyword:
        return True
    return len(keyword) == 1 and set(candidate) == {keyword}


def _matches_inside(folded: str, keyword: str) -> bool:
    """Whether the keyword appears inside a longer reply, safely.

    Single characters are deliberately excluded here. 要 / 好 / 对 are far too
    common to read as consent mid-sentence — with a plain substring match
    「我要睡了」 and 「对不起吵到你了」 both came out as "yes". They still count
    when they *are* the whole answer, which is how people actually reply to a
    yes/no question.
    """
    if len(keyword) < 2:
        return False
    if _is_ascii(keyword):
        return re.search(rf"\b{re.escape(keyword)}\b", folded) is not None
    return keyword in folded


def _matches(folded: str, keywords: list[str]) -> bool:
    """Match a folded reply against a keyword list.

    Whole-answer matching is tried against both the folded reply and its
    particle-trimmed form, because the trimming that turns 「好的」 into 「好」
    would otherwise turn 「算了」 into 「算」 and lose the keyword entirely.
    Keywords themselves are only folded, never trimmed.
    """
    trimmed = folded.rstrip(_TRAILING_NOISE)
    folded_keywords = [_fold(kw) for kw in keywords]
    folded_keywords = [kw for kw in folded_keywords if kw]

    for kw in folded_keywords:
        if _matches_whole_answer(folded, kw) or _matches_whole_answer(trimmed, kw):
            return True
    return any(_matches_inside(folded, kw) for kw in folded_keywords)


def classify_answer(text: str, snap: ConfigSnapshot) -> AnswerIntent:
    """Read a reply to "shall I wake X up?" as yes / no / neither.

    Two passes. A short reply is matched as a whole, so bare 要 / 嗯 / 好的 work
    without those characters being able to fire from the middle of an unrelated
    sentence. Longer replies only match keywords of two characters or more (and
    whole words for ASCII), which is what keeps 「我要睡了」 from reading as
    consent.

    Negatives win ties on purpose: 「不用叫醒了」 holds both a negative and an
    affirmative keyword, and reading it as consent is how a refusal used to wake
    the bot up.
    """
    folded = _fold(text)
    if not folded:
        return "unclear"
    if _matches(folded, snap.negative_keywords):
        return "no"
    if _matches(folded, snap.affirmative_keywords):
        return "yes"
    return "unclear"


def is_urgent(text: str, snap: ConfigSnapshot) -> bool:
    """An emergency skips the two-step handshake."""
    folded = _fold(text)
    if not folded:
        return False
    return _matches(folded, snap.urgent_keywords)


def _record_attempt(
    state: ChatSleepState,
    now_utc: datetime,
    user_id: str,
    confirmed: bool,
) -> ChatSleepState:
    if state.cycle is None:
        return state
    attempts = list(state.cycle.wake_attempts)
    attempts.append(
        WakeAttempt(
            user_id=user_id,
            chat_key=state.chat_key,
            attempted_at=now_utc,
            is_confirmed=confirmed,
            confirmed_at=now_utc if confirmed else None,
        )
    )
    state.cycle = state.cycle.model_copy(update={"wake_attempts": attempts})
    return state


def _confirm_latest_attempt(state: ChatSleepState, now_utc: datetime) -> ChatSleepState:
    """Mark only the most recent unanswered call as the one that worked.

    The previous version flipped every unconfirmed attempt by that user, which
    erased the record of the pings they had already been ignored on — and those
    pings are exactly what the quality model is supposed to charge for.
    """
    if state.cycle is None:
        return state
    attempts = list(state.cycle.wake_attempts)
    for i in range(len(attempts) - 1, -1, -1):
        if not attempts[i].is_confirmed:
            attempts[i] = attempts[i].model_copy(
                update={"is_confirmed": True, "confirmed_at": now_utc}
            )
            break
    state.cycle = state.cycle.model_copy(update={"wake_attempts": attempts})
    return state


def offer_is_live(state: ChatSleepState, now_utc: datetime) -> bool:
    return state.pending_offer is not None and now_utc <= state.pending_offer.expires_at


def _may_answer(state: ChatSleepState, user_id: str, snap: ConfigSnapshot) -> bool:
    if snap.answer_scope == "anyone":
        return True
    return state.pending_offer is not None and state.pending_offer.user_id == user_id


def _offer_is_suppressed(
    state: ChatSleepState,
    now_utc: datetime,
    snap: ConfigSnapshot,
) -> str | None:
    """Why the bot should stay quiet instead of asking again, if it should."""
    if state.snooze_until is not None and now_utc < state.snooze_until:
        return "snoozed"
    if state.offers_sent_tonight >= max(0, snap.max_offers_per_night):
        return "nightly limit"
    if state.last_offer_at is not None:
        cooldown = timedelta(minutes=max(0, snap.offer_cooldown_minutes))
        if now_utc - state.last_offer_at < cooldown:
            return "cooldown"
    return None


def handle_message_while_asleep(
    state: ChatSleepState,
    now_utc: datetime,
    user_id: str,
    text: str,
    persona_name: str,
    is_valid_call: bool,
) -> tuple[ChatSleepState, SleepAction]:
    """Handle one inbound message while the chat is ASLEEP.

    Two phases. With no question outstanding, a valid call gets one — subject to
    a per-night cap, a cooldown and a snooze, so a chat that keeps saying the
    wake word at 3am gets one reply, not one per message. With a question
    outstanding, the message is read as an answer instead of as another call,
    which is what makes plain "要" work and plain "算了" stop meaning yes.
    """
    if state.cycle is None:
        return state, ActionNone()

    snap = state.cycle.config_snapshot

    if is_valid_call and is_urgent(text, snap):
        state = _record_attempt(state, now_utc, user_id, confirmed=True)
        state = state.model_copy(update={"pending_offer": None})
        state = transition_to_awake_early(
            state, now_utc, user_id, snap.early_wake_idle_minutes, reason="urgent"
        )
        return state, ActionForceWake(reason="urgent")

    if offer_is_live(state, now_utc):
        if not _may_answer(state, user_id, snap):
            return state, ActionNone()

        intent = classify_answer(text, snap)
        if intent == "unclear" and snap.unclear_answer == "wake" and is_valid_call:
            intent = "yes"

        if intent == "yes":
            state = _confirm_latest_attempt(state, now_utc)
            state = transition_to_awake_early(
                state, now_utc, user_id, snap.early_wake_idle_minutes
            )
            return state, ActionForceWake()

        if intent == "no":
            # Explicitly told to stay asleep: drop the question and stop asking
            # for a while, rather than treating the refusal as consent.
            state = state.model_copy(
                update={
                    "pending_offer": None,
                    "snooze_until": now_utc
                    + timedelta(minutes=max(0, snap.snooze_minutes)),
                }
            )
            return state, ActionNone()

        # Neither yes nor no: stay asleep, keep the question alive, say nothing.
        return state, ActionNone()

    if not is_valid_call:
        return state, ActionNone()

    suppressed = _offer_is_suppressed(state, now_utc, snap)
    if suppressed is not None:
        logger.debug("Not offering to wake %s (%s)", state.chat_key, suppressed)
        return state, ActionNone()

    state = _record_attempt(state, now_utc, user_id, confirmed=False)
    state = state.model_copy(
        update={
            "pending_offer": PendingWakeOffer(
                user_id=user_id,
                offered_at=now_utc,
                expires_at=now_utc
                + timedelta(seconds=snap.wake_confirm_window_seconds),
            ),
            "offers_sent_tonight": state.offers_sent_tonight + 1,
            "last_offer_at": now_utc,
        }
    )

    near = is_near_wake(
        now_utc,
        state.cycle.sleep_at,
        state.cycle.planned_wake_at,
        snap.near_wake_minutes,
    )
    template = snap.near_wake_prompt if near else snap.asleep_prompt
    try:
        prompt = template.format(persona=persona_name)
    except (KeyError, IndexError, ValueError):
        logger.warning("Bad wake prompt template %r, falling back", template)
        prompt = f"【{persona_name}已经睡了 要叫醒{persona_name}吗？】"

    return state, ActionSendFixed(text=prompt, block_mode=snap.history_mode)


# ---------------------------------------------------------------------------
# Natural wake settlement
# ---------------------------------------------------------------------------


def should_send_wake_notice(
    cycle: SleepCycle,
    policy: str = "always",
) -> bool:
    """Decide whether to announce the natural wake-up.

    `always` matches the original requirement (the bot reports when it wakes up
    on its own). `if_disturbed` is the pre-fix behaviour, kept for operators who
    do not want a daily message in quiet channels.
    """
    if cycle.ended_while_early_awake:
        return False
    if policy == "never":
        return False
    if policy == "if_disturbed":
        return bool(cycle.wake_attempts)
    return True


def format_sleep_duration(total_seconds: float) -> str:
    """Format sleep duration for the wake notice (spec §10.3)."""
    total_minutes = int(total_seconds / 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    if minutes == 0 and hours > 0:
        return f"{hours} 小时"
    elif hours == 0:
        return f"{minutes} 分钟"
    else:
        return f"{hours} 小时 {minutes} 分钟"


def compute_actual_sleep_seconds(cycle: SleepCycle) -> float:
    """Compute total sleep time from segments, excluding timer intervals and early-wake gaps."""
    total = 0.0
    for seg in cycle.sleep_segments:
        if seg.close_at is None:
            continue
        seg_start = seg.open_at
        seg_end = seg.close_at
        seg_duration = (seg_end - seg_start).total_seconds()
        if seg_duration > 0:
            total += seg_duration
    return total


def settle_natural_wake(
    state: ChatSleepState,
    now_utc: datetime,
    persona_name: str,
    quality_fn: Callable[[SleepCycle, float], int],
    notice_policy: str = "always",
) -> tuple[ChatSleepState, SleepAction | None]:
    """Handle natural wake-up at planned_wake_at.

    Takes a `quality_fn` rather than a pre-computed score on purpose. The final
    sleep segment is still open when settlement starts, and
    `compute_actual_sleep_seconds` skips unclosed segments — so any caller that
    scored the cycle *before* this function ran measured zero seconds of sleep
    and reported the quality floor, while the duration string (rebuilt after the
    close) came out correct. Closing first and scoring here makes the two
    numbers structurally incapable of disagreeing.
    """
    if state.cycle is None:
        state = transition_to_awake(state, now_utc)
        return state, None

    was_early_awake = state.status == SleepStatus.AWAKE_EARLY

    state = transition_to_awake(state, now_utc)
    cycle = state.cycle
    if cycle is None:  # pragma: no cover - transition never drops the cycle
        return state, None

    if was_early_awake:
        return state, None

    if not should_send_wake_notice(cycle, notice_policy):
        return state, None

    sleep_secs = compute_actual_sleep_seconds(cycle)
    quality_percent = quality_fn(cycle, sleep_secs)
    duration_str = format_sleep_duration(sleep_secs)

    text = f"【{persona_name}已起床：昨日睡眠质量 {quality_percent}%，睡眠时长 {duration_str}】"

    state.cycle = cycle.model_copy(
        update={"notification_status": NotificationStatus.SENDING}
    )

    return state, ActionSendWakeNotice(text=text)


def mark_notice_sent(state: ChatSleepState) -> ChatSleepState:
    """Mark wake notice as sent after successful delivery."""
    if state.cycle:
        state.cycle = state.cycle.model_copy(
            update={"notification_status": NotificationStatus.SENT}
        )
    return state


def mark_notice_failed(state: ChatSleepState) -> ChatSleepState:
    """Revert notice status on delivery failure."""
    if state.cycle:
        state.cycle = state.cycle.model_copy(
            update={"notification_status": NotificationStatus.PENDING}
        )
    return state


# ---------------------------------------------------------------------------
# Resume sleep (bot-initiated or idle timeout)
# ---------------------------------------------------------------------------


def handle_resume_sleep(
    state: ChatSleepState,
    now_utc: datetime,
    persona_name: str,
) -> tuple[ChatSleepState, SleepAction]:
    """AWAKE_EARLY -> ASLEEP: bot calls resume_sleep tool or idle timeout.

    Raises ValueError if not AWAKE_EARLY or past planned wake.
    """
    if state.status != SleepStatus.AWAKE_EARLY:
        raise ValueError(f"Cannot resume sleep: status is {state.status}, not AWAKE_EARLY")

    if state.cycle and now_utc >= state.cycle.planned_wake_at:
        raise ValueError("Cannot resume sleep: already past planned wake time")

    state = transition_resume_sleep(state, now_utc)
    text = f"【{persona_name}已睡下】"
    return state, ActionSendResumeSleep(text=text)


def handle_idle_sleep_back(
    state: ChatSleepState,
    now_utc: datetime,
) -> ChatSleepState:
    """AWAKE_EARLY -> ASLEEP: silent sleep-back on idle timeout (spec §9.3)."""
    if state.status != SleepStatus.AWAKE_EARLY:
        return state

    if state.cycle and now_utc >= state.cycle.planned_wake_at:
        return state

    state = transition_resume_sleep(state, now_utc)
    return state


# ---------------------------------------------------------------------------
# Idle deadline management
# ---------------------------------------------------------------------------


def refresh_idle_deadline(
    state: ChatSleepState,
    now_utc: datetime,
) -> ChatSleepState:
    """Refresh the idle-sleep deadline after a real user interaction."""
    if state.status != SleepStatus.AWAKE_EARLY:
        return state
    if state.cycle is None:
        return state

    minutes = state.cycle.config_snapshot.early_wake_idle_minutes
    deadline = now_utc + timedelta(minutes=minutes)
    return state.model_copy(update={"idle_sleep_deadline": deadline})


def is_idle_expired(state: ChatSleepState, now_utc: datetime) -> bool:
    """Check if the idle-sleep deadline has passed."""
    if state.status != SleepStatus.AWAKE_EARLY:
        return False
    if state.idle_sleep_deadline is None:
        return False
    return now_utc >= state.idle_sleep_deadline


# ---------------------------------------------------------------------------
# Pending offer cleanup
# ---------------------------------------------------------------------------


def cleanup_expired_offers(
    state: ChatSleepState,
    now_utc: datetime,
) -> ChatSleepState:
    """Drop the pending wake-up question once it has expired (lazy cleanup)."""
    if state.pending_offer is None:
        return state
    if now_utc <= state.pending_offer.expires_at:
        return state
    return state.model_copy(update={"pending_offer": None})


# ---------------------------------------------------------------------------
# Timer lease helpers (state-only, no host coupling)
# ---------------------------------------------------------------------------


def open_timer_interval(
    state: ChatSleepState,
    task_id: str,
    now_utc: datetime,
    source_type: str = "TIMER_ONESHOT",
) -> ChatSleepState:
    """Record the start of a scheduled task running during the night.

    Deliberately does **not** close the sleep segment. The bot did not get out
    of bed for a timer, so night duty overlays the stretch of sleep instead of
    splitting it — closing the segment here (as the first version did) also
    meant nothing reopened it, and the rest of the night stopped being counted
    as sleep at all.
    """
    if state.cycle is None:
        return state
    from .models import SourceType

    intervals = list(state.cycle.timer_intervals)
    intervals.append(
        TimerInterval(
            task_id=task_id,
            start_at=now_utc,
            source_type=SourceType(source_type),
        )
    )
    state.cycle = state.cycle.model_copy(update={"timer_intervals": intervals})
    return state


def close_timer_interval(
    state: ChatSleepState,
    task_id: str,
    now_utc: datetime,
) -> ChatSleepState:
    """Record end of a timer task execution."""
    if state.cycle is None:
        return state
    intervals = list(state.cycle.timer_intervals)
    for i, ti in enumerate(intervals):
        if ti.task_id == task_id and ti.end_at is None:
            intervals[i] = ti.model_copy(update={"end_at": now_utc})
    state.cycle = state.cycle.model_copy(update={"timer_intervals": intervals})
    return state


def has_active_timer_lease(state: ChatSleepState) -> bool:
    """Check if any timer interval is still open."""
    if state.cycle is None:
        return False
    return any(ti.end_at is None for ti in state.cycle.timer_intervals)
