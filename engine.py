"""Pure state-machine, wake protocol, settlement, and idle-sleep-back logic.

Domain layer — no host imports. All time parameters are aware UTC datetimes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Literal

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
    """User confirmed wake — FORCE_TRIGGER this message, inject wake info."""

    def __init__(self, inject_text: str):
        self.inject_text = inject_text


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
) -> ChatSleepState:
    """AWAKE -> ASLEEP: create cycle and open segment."""
    from datetime import date as date_type

    tz = ZoneInfo(config_snapshot.timezone)
    sleep_date_local = now_utc.astimezone(tz).date().isoformat()

    cycle = create_sleep_cycle(state.chat_key, sleep_date_local, config_snapshot)
    state = state.model_copy(
        update={
            "status": SleepStatus.ASLEEP,
            "cycle": cycle,
            "pending_wake_offers": {},
            "idle_sleep_deadline": None,
        }
    )
    state = open_sleep_segment(state, now_utc)
    return state


def transition_to_awake_early(
    state: ChatSleepState,
    now_utc: datetime,
    user_id: str,
    idle_minutes: int,
) -> ChatSleepState:
    """ASLEEP -> AWAKE_EARLY: user confirmed second wake call."""
    state = close_sleep_segment(state, now_utc)
    deadline = now_utc + timedelta(minutes=idle_minutes)
    state = state.model_copy(
        update={
            "status": SleepStatus.AWAKE_EARLY,
            "pending_wake_offers": {},
            "idle_sleep_deadline": deadline,
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
            "pending_wake_offers": {},
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
        }
    )
    state = open_sleep_segment(state, now_utc)
    return state


# ---------------------------------------------------------------------------
# Wake protocol
# ---------------------------------------------------------------------------


def handle_valid_call_while_asleep(
    state: ChatSleepState,
    now_utc: datetime,
    user_id: str,
    persona_name: str,
) -> tuple[ChatSleepState, SleepAction]:
    """Handle a valid user call during ASLEEP state.

    Returns (new_state, action) where action tells the caller what to do.
    """
    if state.cycle is None:
        return state, ActionNone()

    snap = state.cycle.config_snapshot
    confirm_window = timedelta(seconds=snap.wake_confirm_window_seconds)

    pending = dict(state.pending_wake_offers)
    existing = pending.get(user_id)

    if existing is not None and now_utc <= existing.expires_at:
        # Second call within window -> confirm wake
        wake_attempts = list(state.cycle.wake_attempts)
        for i, wa in enumerate(wake_attempts):
            if wa.user_id == user_id and not wa.is_confirmed:
                wake_attempts[i] = wa.model_copy(
                    update={"is_confirmed": True, "confirmed_at": now_utc}
                )

        state.cycle = state.cycle.model_copy(update={"wake_attempts": wake_attempts})
        pending.pop(user_id, None)
        state = state.model_copy(update={"pending_wake_offers": pending})

        state = transition_to_awake_early(
            state, now_utc, user_id, snap.early_wake_idle_minutes
        )

        inject = "你刚刚被用户提前叫醒了。"
        return state, ActionForceWake(inject_text=inject)

    # First call or expired previous offer -> send fixed prompt
    pending[user_id] = PendingWakeOffer(
        user_id=user_id,
        offered_at=now_utc,
        expires_at=now_utc + confirm_window,
    )

    wake_attempts = list(state.cycle.wake_attempts)
    wake_attempts.append(
        WakeAttempt(
            user_id=user_id,
            chat_key=state.chat_key,
            attempted_at=now_utc,
        )
    )
    state.cycle = state.cycle.model_copy(update={"wake_attempts": wake_attempts})
    state = state.model_copy(update={"pending_wake_offers": pending})

    near = is_near_wake(
        now_utc,
        state.cycle.sleep_at,
        state.cycle.planned_wake_at,
        snap.near_wake_ratio,
    )

    if near:
        text = f"【{persona_name}还没起床 要叫醒{persona_name}吗？】"
    else:
        text = f"【{persona_name}已经睡了 要叫醒{persona_name}吗？】"

    return state, ActionSendFixed(text=text, block_mode=snap.history_mode)


# ---------------------------------------------------------------------------
# Natural wake settlement
# ---------------------------------------------------------------------------


def should_send_wake_notice(cycle: SleepCycle) -> bool:
    """Check if natural wake notice should be sent (spec §10.2)."""
    if cycle.ended_while_early_awake:
        return False
    has_real_attempt = any(True for wa in cycle.wake_attempts)
    return has_real_attempt


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
    quality_percent: int,
) -> tuple[ChatSleepState, SleepAction | None]:
    """Handle natural wake-up at planned_wake_at (spec §10.1, §10.2)."""
    if state.cycle is None:
        state = transition_to_awake(state, now_utc)
        return state, None

    was_early_awake = state.status == SleepStatus.AWAKE_EARLY

    state = transition_to_awake(state, now_utc)

    if was_early_awake:
        return state, None

    if not should_send_wake_notice(state.cycle):
        return state, None

    sleep_secs = compute_actual_sleep_seconds(state.cycle)
    duration_str = format_sleep_duration(sleep_secs)

    text = f"【{persona_name}已起床：昨日睡眠质量 {quality_percent}%，睡眠时长 {duration_str}】"

    state.cycle = state.cycle.model_copy(
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
    """Remove expired pending wake offers (lazy cleanup)."""
    if not state.pending_wake_offers:
        return state
    active = {
        uid: offer
        for uid, offer in state.pending_wake_offers.items()
        if now_utc <= offer.expires_at
    }
    if len(active) != len(state.pending_wake_offers):
        state = state.model_copy(update={"pending_wake_offers": active})
    return state


# ---------------------------------------------------------------------------
# Timer lease helpers (state-only, no host coupling)
# ---------------------------------------------------------------------------


def open_timer_interval(
    state: ChatSleepState,
    task_id: str,
    now_utc: datetime,
    source_type: str = "TIMER_ONESHOT",
) -> ChatSleepState:
    """Record start of a timer task execution during sleep."""
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
    state = close_sleep_segment(state, now_utc)
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
