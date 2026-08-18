"""Continuous sleep-quality scoring model (spec §11).

Role-play metric, not medical. No per-event fixed deductions.
Pure functions — no host imports.
"""

from __future__ import annotations

import hashlib
import math
import struct
from datetime import datetime, timedelta

from .models import SleepCycle, SleepSegment, TimerInterval, WakeAttempt


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# §11.1 Coverage & fragmentation
# ---------------------------------------------------------------------------


def compute_coverage(actual_sleep: float, target_sleep: float) -> float:
    if target_sleep <= 0:
        return 1.0
    return _clamp(actual_sleep / target_sleep, 0.0, 1.0)


def compute_fragmentation(segments: list[SleepSegment], actual_sleep: float) -> float:
    """fragmentation = 1 - sum(seg_dur²) / actual_sleep² when actual_sleep > 0, else 1."""
    if actual_sleep <= 0:
        return 1.0
    sum_sq = 0.0
    for seg in segments:
        if seg.close_at is None:
            continue
        dur = (seg.close_at - seg.open_at).total_seconds()
        if dur > 0:
            sum_sq += dur * dur
    return 1.0 - sum_sq / (actual_sleep * actual_sleep)


# ---------------------------------------------------------------------------
# §11.2 Continuous burden integration
# ---------------------------------------------------------------------------


def _stage_weight(t: float, sleep_at: float, wake_at: float) -> float:
    """Early-sleep has higher weight, near-wake smoothly drops."""
    total = wake_at - sleep_at
    if total <= 0:
        return 1.0
    progress = _clamp((t - sleep_at) / total, 0.0, 1.0)
    return 1.0 - 0.4 * progress


def _exp_kernel(t: float, event_t: float, half_life: float = 1800.0) -> float:
    """Exponential decay kernel centered at event_t with given half-life in seconds."""
    dt = t - event_t
    if dt < 0:
        return 0.0
    decay_rate = math.log(2) / half_life
    return math.exp(-decay_rate * dt)


def compute_user_burden(
    cycle: SleepCycle,
    target_sleep: float,
    step: float = 60.0,
) -> float:
    """Integrate user disturbance burden across the target sleep interval."""
    if target_sleep <= 0:
        return 0.0

    sleep_at_ts = cycle.sleep_at.timestamp()
    wake_at_ts = cycle.planned_wake_at.timestamp()

    attempt_times = [wa.attempted_at.timestamp() for wa in cycle.wake_attempts]

    early_awake_intervals: list[tuple[float, float]] = []
    segments = cycle.sleep_segments
    for i, seg in enumerate(segments):
        if seg.close_at is None:
            continue
        close_ts = seg.close_at.timestamp()
        if i + 1 < len(segments):
            next_open_ts = segments[i + 1].open_at.timestamp()
            early_awake_intervals.append((close_ts, next_open_ts))

    weighted_integral = 0.0
    t = sleep_at_ts
    while t < wake_at_ts:
        w = _stage_weight(t, sleep_at_ts, wake_at_ts)

        kernel_sum = 0.0
        for at in attempt_times:
            kernel_sum += _exp_kernel(t, at)

        for start, end in early_awake_intervals:
            if start <= t < end:
                kernel_sum += 1.0

        saturated = 1.0 - math.exp(-kernel_sum) if kernel_sum > 0 else 0.0
        weighted_integral += saturated * w * step
        t += step

    return weighted_integral / target_sleep


def compute_timer_burden(
    cycle: SleepCycle,
    target_sleep: float,
    step: float = 60.0,
) -> float:
    """Integrate timer-task burden across the target sleep interval."""
    if target_sleep <= 0:
        return 0.0

    sleep_at_ts = cycle.sleep_at.timestamp()
    wake_at_ts = cycle.planned_wake_at.timestamp()

    timer_intervals: list[tuple[float, float]] = []
    for ti in cycle.timer_intervals:
        start = ti.start_at.timestamp()
        end = ti.end_at.timestamp() if ti.end_at else wake_at_ts
        timer_intervals.append((start, end))

    if not timer_intervals:
        return 0.0

    weighted_integral = 0.0
    timer_weight = 0.3
    t = sleep_at_ts
    while t < wake_at_ts:
        w = _stage_weight(t, sleep_at_ts, wake_at_ts)

        active = 0.0
        for start, end in timer_intervals:
            if start <= t < end:
                active = 1.0
                break

        weighted_integral += active * w * timer_weight * step
        t += step

    return weighted_integral / target_sleep


# ---------------------------------------------------------------------------
# §11.3 Stable jitter
# ---------------------------------------------------------------------------


def compute_stable_jitter(
    chat_key: str,
    sleep_date: str,
    quality_seed: str,
    jitter_points: float,
) -> float:
    """Deterministic jitter in [-jitter_points, +jitter_points].

    Same inputs always produce the same jitter, across restarts.
    """
    h = hashlib.sha256(
        f"{chat_key}:{sleep_date}:{quality_seed}:jitter".encode()
    ).digest()
    raw = struct.unpack(">I", h[:4])[0]
    normalized = raw / 0xFFFFFFFF
    return (normalized * 2.0 - 1.0) * jitter_points


# ---------------------------------------------------------------------------
# Final quality score
# ---------------------------------------------------------------------------


def compute_quality(
    cycle: SleepCycle,
    actual_sleep_seconds: float,
) -> int:
    """Compute final quality percentage (spec §11.3)."""
    snap = cycle.config_snapshot
    target_sleep = (cycle.planned_wake_at - cycle.sleep_at).total_seconds()

    coverage = compute_coverage(actual_sleep_seconds, target_sleep)
    fragmentation = compute_fragmentation(cycle.sleep_segments, actual_sleep_seconds)
    user_burden = compute_user_burden(cycle, target_sleep)
    timer_burden = compute_timer_burden(cycle, target_sleep)
    jitter = compute_stable_jitter(
        cycle.cycle_id,
        cycle.sleep_date,
        cycle.quality_seed,
        snap.quality_jitter_points,
    )

    raw = (
        100.0
        - 40.0 * (1.0 - coverage) ** 1.6
        - 18.0 * fragmentation ** 1.3
        - 32.0 * (1.0 - math.exp(-3.0 * user_burden))
        - 8.0 * (1.0 - math.exp(-3.0 * timer_burden))
        + jitter
    )

    return round(_clamp(raw, snap.quality_min, snap.quality_max))
