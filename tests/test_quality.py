"""Tests for the quality scoring model."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from nekro_auto_sleep.models import (
    ConfigSnapshot,
    SleepCycle,
    SleepSegment,
    TimerInterval,
    WakeAttempt,
)
from nekro_auto_sleep.quality import (
    compute_coverage,
    compute_fragmentation,
    compute_quality,
    compute_stable_jitter,
)
from nekro_auto_sleep.schedule import create_config_snapshot

UTC = ZoneInfo("UTC")


def _make_snapshot(**overrides) -> ConfigSnapshot:
    defaults = {
        "timezone": "Asia/Shanghai",
        "sleep_time": "23:00",
        "wake_time_start": "06:45",
        "wake_time_end": "08:30",
        "wake_random_step_minutes": 1,
        "near_wake_ratio": 0.15,
        "wake_confirm_window_seconds": 180,
        "history_mode": "preserve",
        "call_keywords": "醒醒,起床,在吗",
        "fallback_persona_name": "Bot",
        "early_wake_idle_minutes": 10,
        "quality_min": 60,
        "quality_max": 120,
        "quality_jitter_points": 4.0,
    }
    defaults.update(overrides)
    return create_config_snapshot(**defaults)


class TestCoverage:
    def test_full(self):
        assert compute_coverage(9 * 3600, 9 * 3600) == 1.0

    def test_half(self):
        assert abs(compute_coverage(4.5 * 3600, 9 * 3600) - 0.5) < 0.01

    def test_clamped(self):
        assert compute_coverage(20 * 3600, 9 * 3600) == 1.0

    def test_zero_target(self):
        assert compute_coverage(100, 0) == 1.0


class TestFragmentation:
    def test_single_segment(self):
        seg = SleepSegment(
            open_at=datetime(2026, 8, 13, 15, 0, tzinfo=UTC),
            close_at=datetime(2026, 8, 14, 0, 0, tzinfo=UTC),
        )
        frag = compute_fragmentation([seg], 9 * 3600)
        assert frag == 0.0

    def test_two_equal_segments(self):
        seg1 = SleepSegment(
            open_at=datetime(2026, 8, 13, 15, 0, tzinfo=UTC),
            close_at=datetime(2026, 8, 13, 19, 30, tzinfo=UTC),
        )
        seg2 = SleepSegment(
            open_at=datetime(2026, 8, 13, 20, 0, tzinfo=UTC),
            close_at=datetime(2026, 8, 14, 0, 30, tzinfo=UTC),
        )
        total = 4.5 * 3600 + 4.5 * 3600
        frag = compute_fragmentation([seg1, seg2], total)
        assert frag == 0.5

    def test_no_sleep(self):
        frag = compute_fragmentation([], 0)
        assert frag == 1.0


class TestStableJitter:
    def test_deterministic(self):
        j1 = compute_stable_jitter("chat", "2026-08-13", "seed123", 4.0)
        j2 = compute_stable_jitter("chat", "2026-08-13", "seed123", 4.0)
        assert j1 == j2

    def test_in_range(self):
        j = compute_stable_jitter("chat", "2026-08-13", "seed123", 4.0)
        assert -4.0 <= j <= 4.0


class TestQualityScore:
    def test_perfect_sleep(self):
        snap = _make_snapshot()
        sleep_at = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        wake_at = datetime(2026, 8, 14, 0, 30, tzinfo=UTC)
        seg = SleepSegment(open_at=sleep_at, close_at=wake_at)
        cycle = SleepCycle(
            cycle_id="test",
            sleep_date="2026-08-13",
            timezone="Asia/Shanghai",
            sleep_at=sleep_at,
            planned_wake_at=wake_at,
            config_snapshot=snap,
            quality_seed="abcdef1234567890",
            sleep_segments=[seg],
        )
        quality = compute_quality(cycle, (wake_at - sleep_at).total_seconds())
        assert 90 <= quality <= 120

    def test_disrupted_sleep(self):
        snap = _make_snapshot()
        sleep_at = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        wake_at = datetime(2026, 8, 14, 0, 30, tzinfo=UTC)
        seg1 = SleepSegment(
            open_at=sleep_at,
            close_at=sleep_at + timedelta(hours=3),
        )
        seg2 = SleepSegment(
            open_at=sleep_at + timedelta(hours=4),
            close_at=wake_at,
        )
        attempt = WakeAttempt(
            user_id="user1",
            chat_key="test",
            attempted_at=sleep_at + timedelta(hours=3),
        )
        cycle = SleepCycle(
            cycle_id="test",
            sleep_date="2026-08-13",
            timezone="Asia/Shanghai",
            sleep_at=sleep_at,
            planned_wake_at=wake_at,
            config_snapshot=snap,
            quality_seed="abcdef1234567890",
            sleep_segments=[seg1, seg2],
            wake_attempts=[attempt],
        )
        actual = 3 * 3600 + 5.5 * 3600
        quality = compute_quality(cycle, actual)
        assert quality < 100

    def test_clamped_to_range(self):
        snap = _make_snapshot(quality_min=60, quality_max=120)
        sleep_at = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        wake_at = datetime(2026, 8, 14, 0, 30, tzinfo=UTC)
        cycle = SleepCycle(
            cycle_id="test",
            sleep_date="2026-08-13",
            timezone="Asia/Shanghai",
            sleep_at=sleep_at,
            planned_wake_at=wake_at,
            config_snapshot=snap,
            quality_seed="abcdef1234567890",
            sleep_segments=[],
        )
        quality = compute_quality(cycle, 0)
        assert quality >= 60
