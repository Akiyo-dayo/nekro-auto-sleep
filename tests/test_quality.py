"""Tests for the sleep-quality model.

The golden table at the bottom is the point of this file: it pins the *shape* of
the curve, not just individual formulas. Retuning the constants in `quality.py`
without moving these bands means the retune changed what the percentage means.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from nekro_auto_sleep.engine import (
    close_timer_interval,
    compute_actual_sleep_seconds,
    open_timer_interval,
    handle_message_while_asleep,
    transition_resume_sleep,
    transition_to_awake,
    transition_to_sleep,
)
from nekro_auto_sleep.models import ChatSleepState, TimerInterval
from nekro_auto_sleep.quality import (
    QualityBreakdown,
    compute_quality,
    compute_quality_detail,
    compute_stable_jitter,
    night_duty_seconds,
    stage_weight,
)

from .conftest import CHAT_KEY, UTC

BED = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)  # 23:00 Shanghai


def _sleep(snapshot) -> ChatSleepState:
    return transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), BED, snapshot)


def _night(
    snapshot,
    *,
    pings: int = 0,
    ping_at_hours: float = 3.0,
    early_wake_minutes: int = 0,
    cut_short_hours: float | None = None,
) -> QualityBreakdown:
    """Play out one night and score it.

    The rate limiter is disabled here on purpose: these tests are about what the
    percentage says once events happened, not about how many of them the bot
    would have replied to (that lives in test_engine.py).
    """
    snapshot = snapshot.model_copy(
        update={"max_offers_per_night": 999, "offer_cooldown_minutes": 0}
    )
    state = _sleep(snapshot)

    for i in range(pings):
        moment = BED + timedelta(hours=ping_at_hours, minutes=25 * i)
        state, _ = handle_message_while_asleep(state, moment, f"u{i}", "在吗", "Bot", True)

    if early_wake_minutes:
        moment = BED + timedelta(hours=ping_at_hours + 1.5)
        state, _ = handle_message_while_asleep(state, moment, "u9", "在吗", "Bot", True)
        state, _ = handle_message_while_asleep(
            state, moment + timedelta(seconds=20), "u9", "要", "Bot", True
        )
        state = transition_resume_sleep(state, moment + timedelta(minutes=early_wake_minutes))

    wake_at = state.cycle.planned_wake_at
    if cut_short_hours is not None:
        wake_at = BED + timedelta(hours=cut_short_hours)

    state = transition_to_awake(state, wake_at)
    return compute_quality_detail(state.cycle, compute_actual_sleep_seconds(state.cycle))


class TestStageWeight:
    def test_inverted_u_peaks_mid_night(self):
        """Being woken at 3am costs more than at bedtime or just before the alarm.

        The old weight fell monotonically from bedtime, which scored a 23:30 ping
        as the single worst moment of the night.
        """
        sleep_at = BED
        wake_at = BED + timedelta(hours=8)
        just_asleep = stage_weight(sleep_at + timedelta(minutes=15), sleep_at, wake_at)
        middle = stage_weight(sleep_at + timedelta(hours=4), sleep_at, wake_at)
        near_alarm = stage_weight(wake_at - timedelta(minutes=15), sleep_at, wake_at)

        assert middle > just_asleep
        assert middle > near_alarm
        assert 0.34 <= just_asleep <= 0.5
        assert middle == pytest.approx(1.0, abs=0.01)

    def test_clamped_outside_the_window(self):
        sleep_at = BED
        wake_at = BED + timedelta(hours=8)
        assert stage_weight(sleep_at - timedelta(hours=1), sleep_at, wake_at) == pytest.approx(0.35)
        assert stage_weight(wake_at + timedelta(hours=1), sleep_at, wake_at) == pytest.approx(0.35)

    def test_degenerate_window(self):
        assert stage_weight(BED, BED, BED) == 1.0


class TestStableJitter:
    def test_deterministic(self, default_snapshot):
        a = compute_stable_jitter("c1", "2026-08-13", "seed", 4.0)
        b = compute_stable_jitter("c1", "2026-08-13", "seed", 4.0)
        assert a == b

    def test_in_range(self):
        for i in range(50):
            value = compute_stable_jitter(f"c{i}", "2026-08-13", "seed", 4.0)
            assert -4.0 <= value <= 4.0

    def test_varies_by_cycle(self):
        values = {compute_stable_jitter(f"c{i}", "2026-08-13", "s", 4.0) for i in range(20)}
        assert len(values) > 1


class TestNightDuty:
    def test_open_interval_runs_to_the_planned_wake(self, default_snapshot):
        state = _sleep(default_snapshot)
        start = BED + timedelta(hours=2)
        state.cycle = state.cycle.model_copy(
            update={"timer_intervals": [TimerInterval(task_id="t1", start_at=start)]}
        )
        expected = (state.cycle.planned_wake_at - start).total_seconds()
        assert night_duty_seconds(state.cycle) == pytest.approx(expected)

    def test_half_of_a_duty_stretch_is_charged_as_lost_rest(self, default_snapshot):
        state = _sleep(default_snapshot)
        start = BED + timedelta(hours=2)
        state.cycle = state.cycle.model_copy(
            update={
                "timer_intervals": [
                    TimerInterval(task_id="t1", start_at=start, end_at=start + timedelta(hours=1))
                ]
            }
        )
        state = transition_to_awake(state, state.cycle.planned_wake_at)
        slept = compute_actual_sleep_seconds(state.cycle)

        detail = compute_quality_detail(state.cycle, slept)
        assert detail.effective_hours == pytest.approx(slept / 3600 - 0.5, abs=0.01)

    def test_duty_does_not_break_the_sleep_segment(self, default_snapshot):
        """A timer is not the bot getting out of bed."""
        state = _sleep(default_snapshot)
        state = open_timer_interval(state, "t1", BED + timedelta(hours=2))
        assert state.cycle.sleep_segments[-1].close_at is None

        state = close_timer_interval(state, "t1", BED + timedelta(hours=2, minutes=5))
        state = transition_to_awake(state, state.cycle.planned_wake_at)
        assert len(state.cycle.sleep_segments) == 1


class TestTargetSelection:
    def test_default_target_is_this_nights_own_plan(self, default_snapshot):
        detail = _night(default_snapshot)
        state = _sleep(default_snapshot)
        planned = (state.cycle.planned_wake_at - state.cycle.sleep_at).total_seconds()
        assert detail.target_hours == pytest.approx(planned / 3600, abs=0.001)
        assert detail.coverage_ratio == pytest.approx(1.0)

    def test_the_random_wake_point_does_not_move_the_score(self, default_snapshot):
        """An early draw is not a bad night.

        The wake-up time is drawn from a range on purpose. Scoring an
        undisturbed night against anything fixed made that draw worth some
        fifteen points, so the number moved for a reason that has nothing to do
        with how the night actually went.
        """
        scores = set()
        for wake_after_hours in (7.75, 8.5, 9.5):
            state = _sleep(default_snapshot)
            state.cycle = state.cycle.model_copy(
                update={
                    "planned_wake_at": state.cycle.sleep_at
                    + timedelta(hours=wake_after_hours)
                }
            )
            state = transition_to_awake(state, state.cycle.planned_wake_at)
            scores.add(
                compute_quality_detail(
                    state.cycle, compute_actual_sleep_seconds(state.cycle)
                ).score
            )

        assert len(scores) == 1, f"score moved with the wake draw: {sorted(scores)}"
        assert scores.pop() > 100

    def test_a_clean_night_scores_above_100(self, default_snapshot):
        detail = _night(default_snapshot)
        assert detail.bonus_clean_night > 0
        assert detail.score > 100

    def test_a_single_call_removes_the_clean_night_bonus(self, default_snapshot):
        detail = _night(default_snapshot, pings=1)
        assert detail.bonus_clean_night == 0
        assert detail.score < 100

    def test_explicit_target_overrides(self, default_snapshot):
        snap = default_snapshot.model_copy(update={"sleep_target_hours": 6.0})
        detail = _night(snap)
        assert detail.target_hours == pytest.approx(6.0)
        # Sleeping well past the target is capped rather than unbounded.
        assert detail.coverage_ratio == pytest.approx(1.25)

    def test_over_target_night_scores_above_100(self, default_snapshot):
        snap = default_snapshot.model_copy(
            update={"sleep_target_hours": 7.0, "quality_jitter_points": 0.0}
        )
        assert _night(snap).score > 100

    def test_under_explicit_target_scores_below_100(self, default_snapshot):
        snap = default_snapshot.model_copy(
            update={"sleep_target_hours": 12.0, "quality_jitter_points": 0.0}
        )
        assert _night(snap).score < 100


class TestBreakdown:
    def test_terms_add_up_to_the_raw_score(self, default_snapshot):
        d = _night(default_snapshot, pings=2, early_wake_minutes=30)
        expected = (
            d.base
            - d.penalty_fragmentation
            - d.penalty_calls
            - d.penalty_wakes
            + d.bonus_clean_night
            + d.jitter
        )
        assert d.raw == pytest.approx(expected, abs=0.02)

    def test_counts_are_recorded(self, default_snapshot):
        d = _night(default_snapshot, pings=2, early_wake_minutes=30)
        assert d.calls == 2
        assert d.wakes == 1
        assert d.segments == 2

    def test_serialisable(self, default_snapshot):
        payload = _night(default_snapshot).as_dict()
        assert payload["score"] == _night(default_snapshot).score
        assert set(payload) >= {"base", "penalty_calls", "penalty_wakes", "jitter", "raw"}


class TestClamping:
    def test_floor(self, default_snapshot):
        snap = default_snapshot.model_copy(update={"quality_min": 20, "quality_max": 120})
        assert _night(snap, cut_short_hours=0.05).score == 20

    def test_ceiling_is_reachable(self, default_snapshot):
        """QUALITY_MAX used to be unreachable: the raw score topped out at 100."""
        snap = default_snapshot.model_copy(
            update={"sleep_target_hours": 4.0, "quality_max": 110, "quality_jitter_points": 0.0}
        )
        assert _night(snap).score == 110


class TestGoldenCurve:
    """The shape of the curve, in bands. Retuning the constants moves these."""

    @pytest.mark.parametrize(
        ("label", "kwargs", "low", "high"),
        [
            ("undisturbed", {}, 101, 108),
            ("one ignored call", {"pings": 1}, 94, 101),
            ("three ignored calls", {"pings": 3}, 86, 94),
            ("six ignored calls", {"pings": 6}, 75, 84),
            ("woken for 30 min", {"early_wake_minutes": 30}, 76, 85),
            ("woken for two hours", {"early_wake_minutes": 120}, 58, 67),
            ("woken for four hours", {"early_wake_minutes": 240}, 38, 47),
            ("only slept three hours", {"cut_short_hours": 3}, 35, 44),
            ("barely slept", {"cut_short_hours": 1}, 20, 26),
        ],
    )
    def test_bands(self, default_snapshot, label, kwargs, low, high):
        score = _night(default_snapshot, **kwargs).score
        assert low <= score <= high, f"{label}: {score} outside [{low}, {high}]"

    def test_calls_hurt_less_at_the_edges_of_the_night(self, default_snapshot):
        mid = _night(default_snapshot, pings=3, ping_at_hours=3.0).score
        early = _night(default_snapshot, pings=3, ping_at_hours=0.5).score
        late = _night(default_snapshot, pings=3, ping_at_hours=7.5).score

        assert early > mid
        assert late > mid

    def test_the_scale_is_actually_used(self, default_snapshot):
        """Every realistic night used to collapse onto the floor or near 100."""
        scores = [
            _night(default_snapshot).score,
            _night(default_snapshot, pings=6).score,
            _night(default_snapshot, early_wake_minutes=120).score,
            _night(default_snapshot, cut_short_hours=1).score,
        ]
        assert max(scores) - min(scores) >= 60


class TestPublicEntryPoint:
    def test_compute_quality_matches_the_detail(self, default_snapshot):
        state = _sleep(default_snapshot)
        state = transition_to_awake(state, state.cycle.planned_wake_at)
        seconds = compute_actual_sleep_seconds(state.cycle)
        assert compute_quality(state.cycle, seconds) == compute_quality_detail(
            state.cycle, seconds
        ).score
