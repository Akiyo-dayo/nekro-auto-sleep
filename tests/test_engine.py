"""Tests for engine module — state machine transitions, wake protocol, settlement."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from nekro_auto_sleep.engine import (
    ActionForceWake,
    ActionSendFixed,
    ActionSendWakeNotice,
    close_sleep_segment,
    compute_actual_sleep_seconds,
    handle_idle_sleep_back,
    handle_resume_sleep,
    handle_valid_call_while_asleep,
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


class TestWakeProtocol:
    def _make_sleeping_state(self, default_snapshot) -> ChatSleepState:
        state = ChatSleepState(chat_key=CHAT_KEY)
        now = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        return transition_to_sleep(state, now, default_snapshot)

    def test_first_call_sends_fixed(self, default_snapshot):
        state = self._make_sleeping_state(default_snapshot)
        now = datetime(2026, 8, 13, 16, 0, tzinfo=UTC)
        new_state, action = handle_valid_call_while_asleep(
            state, now, "user1", "小明"
        )
        assert isinstance(action, ActionSendFixed)
        assert "小明已经睡了" in action.text
        assert "user1" in new_state.pending_wake_offers

    def test_first_call_near_wake_different_text(self, default_snapshot):
        state = self._make_sleeping_state(default_snapshot)
        # Near planned wake time
        near = state.cycle.planned_wake_at - timedelta(minutes=5)
        _new_state, action = handle_valid_call_while_asleep(
            state, near, "user1", "小明"
        )
        assert isinstance(action, ActionSendFixed)
        assert "还没起床" in action.text

    def test_second_call_within_window(self, default_snapshot):
        state = self._make_sleeping_state(default_snapshot)
        now1 = datetime(2026, 8, 13, 16, 0, tzinfo=UTC)
        state, _ = handle_valid_call_while_asleep(state, now1, "user1", "Bot")

        now2 = now1 + timedelta(seconds=60)
        state, action = handle_valid_call_while_asleep(state, now2, "user1", "Bot")
        assert isinstance(action, ActionForceWake)
        assert state.status == SleepStatus.AWAKE_EARLY

    def test_different_user_cannot_confirm(self, default_snapshot):
        state = self._make_sleeping_state(default_snapshot)
        now1 = datetime(2026, 8, 13, 16, 0, tzinfo=UTC)
        state, _ = handle_valid_call_while_asleep(state, now1, "user1", "Bot")

        now2 = now1 + timedelta(seconds=60)
        state, action = handle_valid_call_while_asleep(state, now2, "user2", "Bot")
        assert isinstance(action, ActionSendFixed)
        assert state.status == SleepStatus.ASLEEP

    def test_expired_offer_resets(self, default_snapshot):
        state = self._make_sleeping_state(default_snapshot)
        now1 = datetime(2026, 8, 13, 16, 0, tzinfo=UTC)
        state, _ = handle_valid_call_while_asleep(state, now1, "user1", "Bot")

        now2 = now1 + timedelta(seconds=200)  # Past 180s window
        state, action = handle_valid_call_while_asleep(state, now2, "user1", "Bot")
        assert isinstance(action, ActionSendFixed)
        assert state.status == SleepStatus.ASLEEP


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
        state, _ = handle_valid_call_while_asleep(state, call_time, "user1", "Bot")

        wake_time = state.cycle.planned_wake_at
        state, action = settle_natural_wake(state, wake_time, "Bot", 103)
        assert isinstance(action, ActionSendWakeNotice)
        assert "103%" in action.text
        assert state.status == SleepStatus.AWAKE

    def test_no_notice_without_attempts(self, default_snapshot):
        state = ChatSleepState(chat_key=CHAT_KEY)
        now = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        state = transition_to_sleep(state, now, default_snapshot)

        wake_time = state.cycle.planned_wake_at
        state, action = settle_natural_wake(state, wake_time, "Bot", 100)
        assert action is None
        assert state.status == SleepStatus.AWAKE

    def test_no_notice_when_early_awake_until_end(self, default_snapshot):
        state = ChatSleepState(chat_key=CHAT_KEY)
        now = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        state = transition_to_sleep(state, now, default_snapshot)

        call_time = now + timedelta(hours=1)
        state, _ = handle_valid_call_while_asleep(state, call_time, "user1", "Bot")
        state, _ = handle_valid_call_while_asleep(
            state, call_time + timedelta(seconds=30), "user1", "Bot"
        )
        assert state.status == SleepStatus.AWAKE_EARLY

        wake_time = state.cycle.planned_wake_at
        state, action = settle_natural_wake(state, wake_time, "Bot", 100)
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
