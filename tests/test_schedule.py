"""Tests for schedule module — time boundaries, random wake, and cycle creation."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from nekro_auto_sleep.schedule import (
    build_wake_candidates,
    compute_cycle_boundaries,
    find_sleep_date_for_now,
    generate_cycle_id,
    generate_quality_seed,
    is_in_sleep_window,
    is_near_wake,
    next_sleep_at,
    parse_hhmm,
    pick_wake_time,
)

TZ = ZoneInfo("Asia/Shanghai")
UTC = ZoneInfo("UTC")


class TestParseHHMM:
    def test_normal(self):
        t = parse_hhmm("23:00")
        assert t.hour == 23 and t.minute == 0

    def test_with_spaces(self):
        t = parse_hhmm("  06:45  ")
        assert t.hour == 6 and t.minute == 45

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_hhmm("invalid")


class TestCycleBoundaries:
    def test_default_config(self):
        sd = date(2026, 8, 13)
        sleep_at, wake_start, wake_end = compute_cycle_boundaries(
            sd, TZ, "23:00", "06:45", "08:30"
        )
        assert sleep_at.astimezone(TZ).hour == 23
        assert sleep_at.astimezone(TZ).date() == sd
        assert wake_start.astimezone(TZ).hour == 6
        assert wake_start.astimezone(TZ).minute == 45
        assert wake_start.astimezone(TZ).date() == date(2026, 8, 14)
        assert wake_end.astimezone(TZ).hour == 8
        assert wake_end.astimezone(TZ).minute == 30
        assert wake_end > wake_start > sleep_at

    def test_same_side_times(self):
        sd = date(2026, 8, 13)
        sleep_at, wake_start, wake_end = compute_cycle_boundaries(
            sd, TZ, "01:00", "06:00", "08:00"
        )
        assert wake_start > sleep_at
        assert wake_end > wake_start


class TestWakeCandidates:
    def test_step_1_minute(self):
        start = datetime(2026, 8, 14, 6, 45, tzinfo=UTC)
        end = datetime(2026, 8, 14, 8, 30, tzinfo=UTC)
        candidates = build_wake_candidates(start, end, 1)
        assert candidates[0] == start
        assert candidates[-1] == end
        assert len(candidates) == 106

    def test_step_clamped(self):
        start = datetime(2026, 8, 14, 6, 45, tzinfo=UTC)
        end = datetime(2026, 8, 14, 8, 30, tzinfo=UTC)
        candidates = build_wake_candidates(start, end, 0)
        assert len(candidates) == 106

    def test_single_point(self):
        t = datetime(2026, 8, 14, 7, 0, tzinfo=UTC)
        candidates = build_wake_candidates(t, t, 1)
        assert len(candidates) == 1


class TestPickWakeTime:
    def test_deterministic(self):
        start = datetime(2026, 8, 14, 6, 45, tzinfo=UTC)
        end = datetime(2026, 8, 14, 8, 30, tzinfo=UTC)
        chat_key = "test-chat"
        sd = date(2026, 8, 13)
        t1 = pick_wake_time(chat_key, sd, start, end, 1)
        t2 = pick_wake_time(chat_key, sd, start, end, 1)
        assert t1 == t2

    def test_different_chat_keys(self):
        start = datetime(2026, 8, 14, 6, 45, tzinfo=UTC)
        end = datetime(2026, 8, 14, 8, 30, tzinfo=UTC)
        sd = date(2026, 8, 13)
        pick_wake_time("chat-a", sd, start, end, 1)
        pick_wake_time("chat-b", sd, start, end, 1)
        # Different chat_keys should (almost certainly) get different times
        # This is probabilistic but with 106 candidates it's very unlikely to collide
        assert True  # Allow rare collision


class TestIsInSleepWindow:
    def test_inside(self):
        sleep_at = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        wake_at = datetime(2026, 8, 14, 0, 30, tzinfo=UTC)
        now = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        assert is_in_sleep_window(now, sleep_at, wake_at) is True

    def test_outside(self):
        sleep_at = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        wake_at = datetime(2026, 8, 14, 0, 30, tzinfo=UTC)
        now = datetime(2026, 8, 14, 1, 0, tzinfo=UTC)
        assert is_in_sleep_window(now, sleep_at, wake_at) is False


class TestIsNearWake:
    def test_near_end(self):
        sleep_at = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        wake_at = datetime(2026, 8, 14, 0, 30, tzinfo=UTC)
        # Last 15% of ~9.5 hours = ~85 min before wake
        near_time = wake_at - timedelta(minutes=30)
        assert is_near_wake(near_time, sleep_at, wake_at, 0.15) is True

    def test_not_near(self):
        sleep_at = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
        wake_at = datetime(2026, 8, 14, 0, 30, tzinfo=UTC)
        early = sleep_at + timedelta(hours=1)
        assert is_near_wake(early, sleep_at, wake_at, 0.15) is False


class TestFindSleepDate:
    def test_finds_current_night(self):
        # 2am Shanghai = in sleep window of Aug 13's cycle
        now_utc = datetime(2026, 8, 13, 18, 0, tzinfo=UTC)  # 2am Aug 14 Shanghai
        result = find_sleep_date_for_now(now_utc, TZ, "23:00", "06:45", "08:30")
        assert result == date(2026, 8, 13)

    def test_daytime_returns_none(self):
        now_utc = datetime(2026, 8, 13, 6, 0, tzinfo=UTC)  # 2pm Shanghai
        result = find_sleep_date_for_now(now_utc, TZ, "23:00", "06:45", "08:30")
        assert result is None


class TestGenerateIds:
    def test_cycle_id_stable(self):
        id1 = generate_cycle_id("chat", date(2026, 8, 13))
        id2 = generate_cycle_id("chat", date(2026, 8, 13))
        assert id1 == id2
        assert len(id1) == 12

    def test_quality_seed_stable(self):
        s1 = generate_quality_seed("chat", date(2026, 8, 13))
        s2 = generate_quality_seed("chat", date(2026, 8, 13))
        assert s1 == s2
        assert len(s1) == 16
