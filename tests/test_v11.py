"""Tests for v1.1.0 additions: timer intervals, burdern isolation, fun layer."""

from __future__ import annotations

from datetime import datetime, timedelta

from zoneinfo import ZoneInfo

from nekro_auto_sleep.engine import (
    build_wake_inject_text,
    close_timer_interval,
    open_timer_interval,
    settle_natural_wake,
    should_send_wake_notice,
)
from nekro_auto_sleep.models import (
    ChatSleepState,
    SleepCycle,
    SleepSegment,
    SleepStatus,
    WakeAttempt,
)
from nekro_auto_sleep.quality import (
    compute_timer_burden,
    compute_user_burden,
    compute_streak_note,
    pick_dream,
    quality_tier,
    stable_pick,
    BAD_DREAMS,
)
from nekro_auto_sleep.schedule import create_config_snapshot

UTC = ZoneInfo("UTC")
CHAT_KEY = "onebot_v11-group_123456789"


def _make_snapshot():
    return create_config_snapshot(
        timezone="Asia/Shanghai",
        sleep_time="23:00",
        wake_time_start="06:45",
        wake_time_end="08:30",
        wake_random_step_minutes=1,
        near_wake_ratio=0.15,
        wake_confirm_window_seconds=180,
        history_mode="preserve",
        call_keywords="醒醒,起床,在吗",
        fallback_persona_name="Bot",
        early_wake_idle_minutes=10,
        quality_min=60,
        quality_max=120,
        quality_jitter_points=4.0,
    )


def _make_state(**cycle_overrides) -> ChatSleepState:
    snap = _make_snapshot()
    sleep_at = datetime(2026, 9, 1, 15, 0, tzinfo=UTC)
    wake_at = datetime(2026, 9, 2, 0, 30, tzinfo=UTC)
    cycle = SleepCycle(
        cycle_id="test-cycle",
        sleep_date="2026-09-01",
        timezone="Asia/Shanghai",
        sleep_at=sleep_at,
        planned_wake_at=wake_at,
        config_snapshot=snap,
        quality_seed="abcdef1234567890",
        sleep_segments=[SleepSegment(open_at=sleep_at)],
        **cycle_overrides,
    )
    return ChatSleepState(chat_key=CHAT_KEY, status=SleepStatus.ASLEEP, cycle=cycle)


class TestTimerIntervals:
    def test_open_ignores_when_awake(self):
        state = ChatSleepState(chat_key=CHAT_KEY, status=SleepStatus.AWAKE)
        now = datetime.now(UTC)
        out = open_timer_interval(state, "t1", now)
        assert out.cycle is None
        assert out.status == SleepStatus.AWAKE

    def test_open_closes_segment_and_records_interval(self):
        state = _make_state()
        now = datetime(2026, 9, 1, 16, 0, tzinfo=UTC)
        out = open_timer_interval(state, "t1", now)
        assert out.cycle is not None
        assert len(out.cycle.timer_intervals) == 1
        assert out.cycle.timer_intervals[0].end_at is None
        assert out.cycle.sleep_segments[-1].close_at == now

    def test_close_reopens_segment_while_asleep(self):
        state = _make_state()
        start = datetime(2026, 9, 1, 16, 0, tzinfo=UTC)
        end = datetime(2026, 9, 1, 16, 10, tzinfo=UTC)
        out = open_timer_interval(state, "t1", start)
        out = close_timer_interval(out, "t1", end)
        assert out.cycle is not None
        assert out.cycle.timer_intervals[0].end_at == end
        # a fresh open segment exists so later sleep still counts
        assert out.cycle.sleep_segments[-1].close_at is None
        assert out.cycle.sleep_segments[-1].open_at == end

    def test_close_keeps_segment_when_awake_early(self):
        state = _make_state()
        state = state.model_copy(update={"status": SleepStatus.AWAKE_EARLY})
        # simulate that the segment was already closed when the bot woke early
        closed = state.cycle.sleep_segments[0].model_copy(
            update={"close_at": datetime(2026, 9, 1, 15, 30, tzinfo=UTC)}
        )
        state.cycle = state.cycle.model_copy(update={"sleep_segments": [closed]})
        start = datetime(2026, 9, 1, 16, 0, tzinfo=UTC)
        end = datetime(2026, 9, 1, 16, 10, tzinfo=UTC)
        out = open_timer_interval(state, "t1", start)  # ignored: not ASLEEP
        out = close_timer_interval(out, "t1", end)
        assert out.cycle is not None
        assert not out.cycle.timer_intervals
        # no new segment opened while awake-early
        assert all(seg.close_at is not None for seg in out.cycle.sleep_segments)

    def test_double_close_does_not_leak_segments(self):
        state = _make_state()
        start = datetime(2026, 9, 1, 16, 0, tzinfo=UTC)
        end = datetime(2026, 9, 1, 16, 10, tzinfo=UTC)
        out = open_timer_interval(state, "t1", start)
        out = close_timer_interval(out, "t1", end)
        n = len(out.cycle.sleep_segments)
        out = close_timer_interval(out, "t1", end)  # second call: nothing to close
        assert len(out.cycle.sleep_segments) == n


class TestBurdenNoDoubleCount:
    def test_timer_gap_not_charged_to_user(self):
        state = _make_state()
        now = datetime(2026, 9, 1, 16, 0, tzinfo=UTC)
        end = datetime(2026, 9, 1, 17, 0, tzinfo=UTC)
        with_timer = open_timer_interval(state, "t1", now)
        with_timer = close_timer_interval(with_timer, "t1", end)
        assert with_timer.cycle is not None

        # same shape but only the wake gap, no timer bookkeeping
        manual = _make_state()
        seg1 = SleepSegment(open_at=manual.cycle.sleep_at, close_at=now)
        seg2 = SleepSegment(open_at=end, close_at=manual.cycle.planned_wake_at)
        manual.cycle = manual.cycle.model_copy(
            update={"sleep_segments": [seg1, seg2]}
        )

        target = (manual.cycle.planned_wake_at - manual.cycle.sleep_at).total_seconds()
        burden_timer = compute_user_burden(with_timer.cycle, target)
        burden_manual = compute_user_burden(manual.cycle, target)
        assert burden_timer < burden_manual * 0.5

    def test_timer_burden_counts_for_timer_only(self):
        state = _make_state()
        start = datetime(2026, 9, 1, 16, 0, tzinfo=UTC)
        end = datetime(2026, 9, 1, 17, 0, tzinfo=UTC)
        out = open_timer_interval(state, "t1", start)
        out = close_timer_interval(out, "t1", end)
        target = (out.cycle.planned_wake_at - out.cycle.sleep_at).total_seconds()
        assert compute_timer_burden(out.cycle, target) > 0


class TestWakeInjectText:
    def test_deep_night_gets_grumpy(self):
        state = _make_state()
        # 30 minutes after sleep_at -> deep night
        now = state.cycle.sleep_at + timedelta(minutes=30)
        text = build_wake_inject_text(state, now)
        assert "起床气" in text

    def test_near_wake_is_gentle(self):
        state = _make_state()
        now = state.cycle.planned_wake_at - timedelta(minutes=20)
        text = build_wake_inject_text(state, now)
        assert "临近自然醒" in text

    def test_mid_sleep_is_sleepy(self):
        state = _make_state()
        now = state.cycle.sleep_at + timedelta(hours=4)
        text = build_wake_inject_text(state, now)
        assert "困" in text


class TestWakeNoticeCondition:
    def test_default_requires_attempts(self):
        state = _make_state()
        state = settle_natural_wake(state, state.cycle.planned_wake_at, "Bot", 100)[0]
        assert not should_send_wake_notice(state.cycle)

    def test_always_sends_without_attempts(self):
        state = _make_state()
        state = settle_natural_wake(state, state.cycle.planned_wake_at, "Bot", 100)[0]
        assert should_send_wake_notice(state.cycle, always=True)

    def test_early_awake_settlement_never_notifies(self):
        state = _make_state()
        state.cycle = state.cycle.model_copy(
            update={
                "wake_attempts": [
                    WakeAttempt(
                        user_id="u1", chat_key=CHAT_KEY,
                        attempted_at=datetime(2026, 9, 1, 16, 0, tzinfo=UTC),
                    )
                ]
            }
        )
        state.cycle = state.cycle.model_copy(update={"ended_while_early_awake": True})
        assert not should_send_wake_notice(state.cycle, always=True)


class TestFunLayer:
    def test_tier_ordering(self):
        assert quality_tier(120)[0] == "神清气爽"
        assert quality_tier(100)[0] == "睡得不错"
        assert quality_tier(85)[0] == "睡得一般"
        assert quality_tier(72)[0] == "睡得迷糊"
        assert quality_tier(60)[0] == "睡眼惺忪"

    def test_dream_deterministic(self):
        d1 = pick_dream("chat:2026-09-01", 100)
        d2 = pick_dream("chat:2026-09-01", 100)
        assert d1 == d2
        assert d1 is not None

    def test_perfect_sleep_no_dream(self):
        assert pick_dream("chat:2026-09-01", 118) is None

    def test_bad_sleep_nightmare_pool(self):
        assert pick_dream("chat:2026-09-01", 62) in BAD_DREAMS

    def test_stable_pick_stable(self):
        assert stable_pick("x", ("a", "b", "c")) == stable_pick("x", ("a", "b", "c"))

    def test_streak_note_first_day(self):
        note = compute_streak_note({"2026-09-01": 97}, "2026-09-01")
        assert note is not None and "第一天" in note

    def test_streak_note_counts_consecutive_good_nights(self):
        history = {
            "2026-08-29": 96,
            "2026-08-30": 98,
            "2026-08-31": 97,
            "2026-09-01": 99,
        }
        note = compute_streak_note(history, "2026-09-01")
        assert note is not None and "连续 4 天" in note

    def test_streak_note_trend_down(self):
        history = {
            "2026-08-31": 99,
            "2026-09-01": 78,
        }
        note = compute_streak_note(history, "2026-09-01")
        assert note is not None and "掉了" in note


class TestHistoryPersistence:
    def test_state_defaults_empty_history(self):
        state = ChatSleepState(chat_key=CHAT_KEY)
        assert state.quality_history == {}

    def test_history_roundtrip(self):
        state = _make_state()
        state = state.model_copy(
            update={"quality_history": {"2026-09-01": 96, "2026-08-31": 88}}
        )
        raw = state.model_dump_json()
        from nekro_auto_sleep.models import ChatSleepState as CS

        restored = CS.model_validate_json(raw)
        assert restored.quality_history == {"2026-09-01": 96, "2026-08-31": 88}
