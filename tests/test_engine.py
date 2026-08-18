"""Tests for engine module — state machine transitions, wake protocol, settlement."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from nekro_auto_sleep.engine import (
    ActionForceWake,
    ActionNone,
    ActionSendResumeSleep,
    ActionSendFixed,
    ActionSendWakeNotice,
    close_sleep_segment,
    compute_actual_sleep_seconds,
    handle_idle_sleep_back,
    handle_resume_sleep,
    clear_wake_decision,
    handle_message_while_asleep,
    is_urgent,
    offer_is_live,
    is_idle_expired,
    open_sleep_segment,
    refresh_idle_deadline,
    settle_natural_wake,
    should_send_wake_notice,
    transition_resume_sleep,
    transition_to_awake,
    transition_to_awake_early,
    transition_to_sleep,
)
from nekro_auto_sleep.models import (
    ChatSleepState,
    SleepCycle,
    SleepSegment,
    SleepStatus,
    WakeAttempt,
)

from .conftest import CHAT_KEY, TZ, UTC


class TestTransitionToSleep:
    def test_basic(self, default_snapshot):
        state = ChatSleepState(chat_key=CHAT_KEY)
        now = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)  # 23:00 Shanghai
        new_state = transition_to_sleep(state, now, default_snapshot)
        assert new_state.status == SleepStatus.ASLEEP
        assert new_state.cycle is not None
        assert len(new_state.cycle.sleep_segments) == 1
        assert new_state.cycle.sleep_segments[0].close_at is None


BED = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)  # 23:00 Shanghai
NIGHT = BED + timedelta(hours=1)


def _sleeping(default_snapshot) -> ChatSleepState:
    return transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), BED, default_snapshot)


def _call(state, now, user="user1", text="醒醒", valid=True):
    return handle_message_while_asleep(state, now, user, text, "Bot", valid)


class TestUrgentShortcut:
    """The one keyword path left, and it ships disabled."""

    def test_matches_the_whole_keyword_only(self, default_snapshot):
        assert is_urgent("出事了 快醒醒", default_snapshot) is True
        assert is_urgent("救命", default_snapshot) is True
        # 「没出事」 embeds 出事 but not the keyword 出事了.
        assert is_urgent("没出事", default_snapshot) is False
        assert is_urgent("这事不急", default_snapshot) is False

    def test_disabled_when_no_keywords_are_configured(self, default_snapshot):
        snap = default_snapshot.model_copy(update={"urgent_keywords": []})
        assert is_urgent("救命", snap) is False

    def test_a_single_character_keyword_only_counts_as_the_whole_message(
        self, default_snapshot
    ):
        """Otherwise one common character turns every sentence into an emergency."""
        snap = default_snapshot.model_copy(update={"urgent_keywords": ["急"]})
        assert is_urgent("急", snap) is True
        assert is_urgent("急！", snap) is True
        assert is_urgent("这事不急", snap) is False
        assert is_urgent("我等下有急事要出门", snap) is False

    def test_ascii_keywords_need_word_boundaries(self, default_snapshot):
        snap = default_snapshot.model_copy(update={"urgent_keywords": ["sos"]})
        assert is_urgent("sos", snap) is True
        assert is_urgent("SOS 快来", snap) is True
        assert is_urgent("sostenuto", snap) is False


class TestWakeProtocol:
    def test_first_call_sends_fixed(self, default_snapshot):
        state, action = _call(_sleeping(default_snapshot), NIGHT)
        assert isinstance(action, ActionSendFixed)
        assert "已经睡了" in action.text
        assert state.status == SleepStatus.ASLEEP
        assert offer_is_live(state, NIGHT)

    def test_near_wake_changes_the_wording(self, default_snapshot):
        state = _sleeping(default_snapshot)
        near = state.cycle.planned_wake_at - timedelta(minutes=30)
        _state, action = _call(state, near)
        assert "还没起床" in action.text

    def test_prompt_wording_is_configurable(self, default_snapshot):
        snap = default_snapshot.model_copy(
            update={"asleep_prompt": "[{persona} zzz - wake up? yes/no]"}
        )
        state = transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), BED, snap)
        _state, action = _call(state, NIGHT)
        assert action.text == "[Bot zzz - wake up? yes/no]"

    def test_a_broken_template_falls_back_instead_of_crashing(self, default_snapshot):
        snap = default_snapshot.model_copy(update={"asleep_prompt": "{nope}"})
        state = transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), BED, snap)
        _state, action = _call(state, NIGHT)
        assert "Bot" in action.text

    @pytest.mark.parametrize("reply", ["要", "算了你睡吧", "我要睡了", "你在哪", "嗯"])
    def test_any_reply_reaches_the_llm(self, default_snapshot, reply):
        """The plugin no longer decides what the reply meant.

        Hand-matching 要 / 算了 could only ever approximate it, and 「我要睡了」
        came out as consent. The second message force-triggers a round and the
        model reads the sentence.
        """
        state = _sleeping(default_snapshot)
        state, _ = _call(state, NIGHT)
        state, action = _call(state, NIGHT + timedelta(seconds=20), text=reply)

        assert isinstance(action, ActionForceWake)
        assert action.reason == "deciding"
        assert state.status == SleepStatus.AWAKE_EARLY
        assert state.wake_decision_pending is True
        assert state.woken_by == "user1"

    def test_bystander_cannot_trigger_the_decision_round(self, default_snapshot):
        state = _sleeping(default_snapshot)
        state, _ = _call(state, NIGHT, user="user1")
        state, action = _call(state, NIGHT + timedelta(seconds=20), user="user2", text="要")

        assert isinstance(action, ActionNone)
        assert state.status == SleepStatus.ASLEEP

    def test_bystander_can_answer_with_open_scope(self, default_snapshot):
        snap = default_snapshot.model_copy(update={"answer_scope": "anyone"})
        state = transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), BED, snap)
        state, _ = _call(state, NIGHT, user="user1")
        state, action = _call(state, NIGHT + timedelta(seconds=20), user="user2", text="要")
        assert isinstance(action, ActionForceWake)

    def test_an_expired_question_is_a_fresh_call_again(self, default_snapshot):
        state = _sleeping(default_snapshot)
        state, _ = _call(state, NIGHT)
        late = NIGHT + timedelta(seconds=200)  # past the 180s window
        state, action = _call(state, late, text="随便说点什么", valid=False)

        assert isinstance(action, ActionNone)
        assert state.status == SleepStatus.ASLEEP

    def test_ordinary_message_never_creates_an_offer(self, default_snapshot):
        state, action = _call(_sleeping(default_snapshot), NIGHT, text="随便聊聊", valid=False)
        assert isinstance(action, ActionNone)
        assert state.pending_offer is None
        assert state.cycle.wake_attempts == []

    def test_urgent_wakes_in_one_step(self, default_snapshot):
        state, action = _call(_sleeping(default_snapshot), NIGHT, text="出事了！快醒醒")
        assert isinstance(action, ActionForceWake)
        assert action.reason == "urgent"
        assert state.status == SleepStatus.AWAKE_EARLY
        assert state.woken_reason == "urgent"
        assert state.wake_decision_pending is False


class TestWakeDecision:
    """What the model does with the round it was handed."""

    def _decide(self, default_snapshot):
        state = _sleeping(default_snapshot)
        state, _ = _call(state, NIGHT)
        state, _ = _call(state, NIGHT + timedelta(seconds=20), text="嗯？")
        return state

    def test_declining_goes_back_to_sleep_without_saying_anything(self, default_snapshot):
        state = self._decide(default_snapshot)
        moment = NIGHT + timedelta(seconds=40)

        state, action = handle_resume_sleep(state, moment, "Bot")

        assert isinstance(action, ActionNone), "declining must not announce itself"
        assert state.status == SleepStatus.ASLEEP
        assert state.wake_decision_pending is False
        assert state.snooze_until == moment + timedelta(minutes=30)

    def test_declining_is_charged_as_a_call_not_a_wake_up(self, default_snapshot):
        state = self._decide(default_snapshot)
        state, _ = handle_resume_sleep(state, NIGHT + timedelta(seconds=40), "Bot")

        attempts = state.cycle.wake_attempts
        assert len(attempts) == 1
        assert attempts[0].is_confirmed is False

    def test_declining_does_not_fragment_the_night(self, default_snapshot):
        state = self._decide(default_snapshot)
        state, _ = handle_resume_sleep(state, NIGHT + timedelta(seconds=40), "Bot")

        assert len(state.cycle.sleep_segments) == 1
        assert state.cycle.sleep_segments[0].close_at is None

    def test_staying_up_then_turning_in_announces_normally(self, default_snapshot):
        state = self._decide(default_snapshot)
        # The model replied and the conversation carried on.
        state = clear_wake_decision(state)

        state, action = handle_resume_sleep(state, NIGHT + timedelta(minutes=5), "Bot")

        assert isinstance(action, ActionSendResumeSleep)
        assert "Bot已睡下" in action.text
        assert len(state.cycle.sleep_segments) == 2

    def test_a_reply_from_the_user_ends_the_decision_round(self, default_snapshot):
        state = self._decide(default_snapshot)
        assert clear_wake_decision(state).wake_decision_pending is False


class TestOfferRateLimit:
    def test_cooldown_between_offers(self, default_snapshot):
        state = _sleeping(default_snapshot)
        state, first = _call(state, NIGHT)
        assert isinstance(first, ActionSendFixed)

        # Offer expires after 180s; calling again 5 minutes later is inside the
        # 20-minute cooldown, so the bot stays quiet instead of asking again.
        state, action = _call(state, NIGHT + timedelta(minutes=5))
        assert isinstance(action, ActionNone)
        assert state.offers_sent_tonight == 1

    def test_asks_again_after_the_cooldown(self, default_snapshot):
        state = _sleeping(default_snapshot)
        state, _ = _call(state, NIGHT)
        state, action = _call(state, NIGHT + timedelta(minutes=25))
        assert isinstance(action, ActionSendFixed)
        assert state.offers_sent_tonight == 2

    def test_nightly_cap(self, default_snapshot):
        state = _sleeping(default_snapshot)
        now = NIGHT
        for _ in range(3):
            state, action = _call(state, now)
            assert isinstance(action, ActionSendFixed)
            now += timedelta(minutes=25)

        state, action = _call(state, now)
        assert isinstance(action, ActionNone)
        assert state.offers_sent_tonight == 3

    def test_every_ignored_call_is_still_charged_to_quality(self, default_snapshot):
        state = _sleeping(default_snapshot)
        state, _ = _call(state, NIGHT)
        state, _ = _call(state, NIGHT + timedelta(minutes=25))
        state, _ = _call(state, NIGHT + timedelta(minutes=25, seconds=20), text="要")

        attempts = state.cycle.wake_attempts
        assert len(attempts) == 2
        # Reaching the LLM confirms only the call that was actually answered;
        # the one the bot slept through stays on the record as unanswered, which
        # is what the quality model charges for. The first protocol flipped
        # every earlier attempt by that user and erased the evidence.
        assert [a.is_confirmed for a in attempts] == [False, True]


class TestResumeSleep:
    def test_resume_from_early(self, default_snapshot):
        state = ChatSleepState(chat_key=CHAT_KEY)
        now = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        state = transition_to_sleep(state, now, default_snapshot)

        wake_time = now + timedelta(hours=1)
        state = transition_to_awake_early(state, wake_time, "user1", 10)

        resume_time = wake_time + timedelta(minutes=5)
        state, action = handle_resume_sleep(state, resume_time, "Bot")
        assert state.status == SleepStatus.ASLEEP
        assert "Bot已睡下" in action.text

    def test_resume_fails_when_awake(self, default_snapshot):
        state = ChatSleepState(chat_key=CHAT_KEY, status=SleepStatus.AWAKE)
        with pytest.raises(ValueError, match="AWAKE_EARLY"):
            handle_resume_sleep(state, datetime.now(UTC), "Bot")


class TestIdleSleepBack:
    def test_idle_expired(self, default_snapshot):
        state = ChatSleepState(chat_key=CHAT_KEY)
        now = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        state = transition_to_sleep(state, now, default_snapshot)

        wake_time = now + timedelta(hours=1)
        state = transition_to_awake_early(state, wake_time, "user1", 10)
        assert state.idle_sleep_deadline is not None

        past_deadline = state.idle_sleep_deadline + timedelta(seconds=1)
        assert is_idle_expired(state, past_deadline)

        state = handle_idle_sleep_back(state, past_deadline)
        assert state.status == SleepStatus.ASLEEP

    def test_refresh_deadline(self, default_snapshot):
        state = ChatSleepState(chat_key=CHAT_KEY)
        now = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        state = transition_to_sleep(state, now, default_snapshot)

        wake_time = now + timedelta(hours=1)
        state = transition_to_awake_early(state, wake_time, "user1", 10)
        old_deadline = state.idle_sleep_deadline

        refresh_time = wake_time + timedelta(minutes=5)
        state = refresh_idle_deadline(state, refresh_time)
        assert state.idle_sleep_deadline > old_deadline


class TestNaturalWake:
    def test_sends_notice_when_attempted(self, default_snapshot):
        state = ChatSleepState(chat_key=CHAT_KEY)
        now = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        state = transition_to_sleep(state, now, default_snapshot)

        call_time = now + timedelta(hours=1)
        state, _ = handle_message_while_asleep(
            state, call_time, "user1", "醒醒", "Bot", True
        )

        wake_time = state.cycle.planned_wake_at
        state, action = settle_natural_wake(
            state, wake_time, "Bot", lambda cycle, secs: 103
        )
        assert isinstance(action, ActionSendWakeNotice)
        assert "103%" in action.text
        assert state.status == SleepStatus.AWAKE

    def test_quiet_night_still_announces_by_default(self, default_snapshot):
        """A night nobody interrupted still gets a wake-up report."""
        state = ChatSleepState(chat_key=CHAT_KEY)
        now = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        state = transition_to_sleep(state, now, default_snapshot)

        wake_time = state.cycle.planned_wake_at
        state, action = settle_natural_wake(
            state, wake_time, "Bot", lambda cycle, secs: 100
        )
        assert isinstance(action, ActionSendWakeNotice)
        assert state.status == SleepStatus.AWAKE

    def test_no_notice_without_attempts_under_if_disturbed(self, default_snapshot):
        state = ChatSleepState(chat_key=CHAT_KEY)
        now = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        state = transition_to_sleep(state, now, default_snapshot)

        wake_time = state.cycle.planned_wake_at
        state, action = settle_natural_wake(
            state, wake_time, "Bot", lambda cycle, secs: 100, "if_disturbed"
        )
        assert action is None
        assert state.status == SleepStatus.AWAKE

    def test_no_notice_when_early_awake_until_end(self, default_snapshot):
        state = ChatSleepState(chat_key=CHAT_KEY)
        now = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        state = transition_to_sleep(state, now, default_snapshot)

        call_time = now + timedelta(hours=1)
        state, _ = handle_message_while_asleep(
            state, call_time, "user1", "醒醒", "Bot", True
        )
        state, _ = handle_message_while_asleep(
            state, call_time + timedelta(seconds=30), "user1", "要", "Bot", True
        )
        assert state.status == SleepStatus.AWAKE_EARLY

        wake_time = state.cycle.planned_wake_at
        state, action = settle_natural_wake(
            state, wake_time, "Bot", lambda cycle, secs: 100
        )
        assert action is None


class TestSleepDuration:
    def test_continuous_sleep(self, default_snapshot):
        seg = SleepSegment(
            open_at=datetime(2026, 8, 13, 15, 0, tzinfo=UTC),
            close_at=datetime(2026, 8, 14, 0, 30, tzinfo=UTC),
        )
        cycle = SleepCycle(
            cycle_id="test",
            sleep_date="2026-08-13",
            timezone="Asia/Shanghai",
            sleep_at=datetime(2026, 8, 13, 15, 0, tzinfo=UTC),
            planned_wake_at=datetime(2026, 8, 14, 0, 30, tzinfo=UTC),
            config_snapshot=default_snapshot,
            quality_seed="abc",
            sleep_segments=[seg],
        )
        secs = compute_actual_sleep_seconds(cycle)
        assert secs == 9.5 * 3600
