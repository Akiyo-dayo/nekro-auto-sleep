"""Sleep-quality scoring.

A role-play metric, not a medical one, with three properties the first model
did not have:

- **The percentage means something.** 100% is "slept the configured target with
  nothing bothering me". Above 100% means the night was longer than the target,
  which is why a number like 103% can show up at all — in the previous model
  anything over 100 came purely from the random jitter, and the configured
  ceiling of 120 was unreachable because the raw score topped out at 100.
- **The scale is used.** Penalties are additive and capped per category instead
  of exponentially saturating, so a rough night lands in the 60s and a wrecked
  one in the 20s rather than everything piling up near the floor.
- **Timing matters the way it does for people.** Being woken just after bedtime
  or right before the alarm is cheap; being woken in the middle of the night is
  expensive. The old stage weight decreased monotonically from bedtime, which
  scored a 23:30 ping as the worst possible moment and contradicted the
  plugin's own near-wake leniency.

Pure functions — no host imports.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import asdict, dataclass
from datetime import datetime

from .models import SleepCycle, SleepSegment

# Penalty scale. Tuned against the golden table in tests/test_quality.py; move
# these and that table moves with them.
FRAGMENT_COST = 6.0  # per extra sleep segment
FRAGMENT_CAP = 24.0
CALL_COST = 4.0  # an unanswered wake-up call
WAKE_COST = 10.0  # a call that actually got the bot out of bed
MAX_COVERAGE_RATIO = 1.25  # a night cannot count as more than 125% of target
NIGHT_DUTY_CREDIT = 0.5  # a timer round counts as half sleep, not as being awake


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class QualityBreakdown:
    """Every term behind the reported percentage, kept for diagnosis."""

    score: int
    target_hours: float
    effective_hours: float
    coverage_ratio: float
    base: float
    penalty_fragmentation: float
    penalty_calls: float
    penalty_wakes: float
    jitter: float
    raw: float
    segments: int
    calls: int
    wakes: int

    def as_dict(self) -> dict[str, float]:
        return {k: (v if isinstance(v, (int, float)) else float(v)) for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def stage_weight(moment: datetime, sleep_at: datetime, wake_at: datetime) -> float:
    """How costly a disturbance is at `moment`, in [0.35, 1.0].

    Inverted U: cheapest right after falling asleep and right before waking,
    most expensive in the middle of the night.
    """
    total = (wake_at - sleep_at).total_seconds()
    if total <= 0:
        return 1.0
    progress = _clamp((moment - sleep_at).total_seconds() / total, 0.0, 1.0)
    return 0.35 + 0.65 * math.sin(math.pi * progress) ** 1.2


def count_segments(segments: list[SleepSegment]) -> int:
    return sum(1 for seg in segments if seg.close_at is not None)


def night_duty_seconds(cycle: SleepCycle) -> float:
    """Time spent running scheduled tasks during the night."""
    total = 0.0
    for interval in cycle.timer_intervals:
        end = interval.end_at or cycle.planned_wake_at
        duration = (end - interval.start_at).total_seconds()
        if duration > 0:
            total += duration
    return total


def compute_stable_jitter(
    chat_key: str,
    sleep_date: str,
    quality_seed: str,
    jitter_points: float,
) -> float:
    """Deterministic jitter in [-jitter_points, +jitter_points].

    Same inputs always produce the same jitter, across restarts.
    """
    h = hashlib.sha256(f"{chat_key}:{sleep_date}:{quality_seed}:jitter".encode()).digest()
    raw = struct.unpack(">I", h[:4])[0]
    normalized = raw / 0xFFFFFFFF
    return (normalized * 2.0 - 1.0) * jitter_points


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def compute_quality_detail(
    cycle: SleepCycle,
    actual_sleep_seconds: float,
) -> QualityBreakdown:
    snap = cycle.config_snapshot

    if snap.sleep_target_hours > 0:
        target_seconds = snap.sleep_target_hours * 3600
    elif cycle.target_sleep_seconds > 0:
        # Auto: the midpoint of the configured wake range. A wake-up later than
        # the midpoint scores above 100%, earlier below.
        target_seconds = cycle.target_sleep_seconds
    else:
        # Cycles written before the target was recorded.
        target_seconds = max(1800.0, (cycle.planned_wake_at - cycle.sleep_at).total_seconds())
    target_hours = target_seconds / 3600

    duty = night_duty_seconds(cycle)
    effective = max(0.0, actual_sleep_seconds) + NIGHT_DUTY_CREDIT * duty

    coverage_ratio = _clamp(effective / target_seconds, 0.0, MAX_COVERAGE_RATIO)
    base = 100.0 * coverage_ratio

    segments = count_segments(cycle.sleep_segments)
    penalty_fragmentation = min(FRAGMENT_CAP, FRAGMENT_COST * max(0, segments - 1))

    penalty_calls = 0.0
    penalty_wakes = 0.0
    calls = 0
    wakes = 0
    for attempt in cycle.wake_attempts:
        weight = stage_weight(attempt.attempted_at, cycle.sleep_at, cycle.planned_wake_at)
        if attempt.is_confirmed:
            wakes += 1
            penalty_wakes += WAKE_COST * weight
        else:
            calls += 1
            penalty_calls += CALL_COST * weight

    jitter = compute_stable_jitter(
        cycle.cycle_id,
        cycle.sleep_date,
        cycle.quality_seed,
        snap.quality_jitter_points,
    )

    raw = base - penalty_fragmentation - penalty_calls - penalty_wakes + jitter
    score = round(_clamp(raw, snap.quality_min, snap.quality_max))

    return QualityBreakdown(
        score=score,
        target_hours=round(target_hours, 3),
        effective_hours=round(effective / 3600, 3),
        coverage_ratio=round(coverage_ratio, 4),
        base=round(base, 2),
        penalty_fragmentation=round(penalty_fragmentation, 2),
        penalty_calls=round(penalty_calls, 2),
        penalty_wakes=round(penalty_wakes, 2),
        jitter=round(jitter, 3),
        raw=round(raw, 2),
        segments=segments,
        calls=calls,
        wakes=wakes,
    )


def compute_quality(cycle: SleepCycle, actual_sleep_seconds: float) -> int:
    """Final quality percentage. See `compute_quality_detail` for the terms."""
    return compute_quality_detail(cycle, actual_sleep_seconds).score
