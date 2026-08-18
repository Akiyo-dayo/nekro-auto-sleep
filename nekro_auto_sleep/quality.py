"""Sleep-quality scoring.

A role-play metric, not a medical one, with three properties the first model
did not have:

- **The percentage means something.** 100% is "slept the night I had planned".
  The reference is *this night's own plan*, not a fixed number of hours, because
  the wake-up point is drawn at random from a range: scoring against a fixed
  target would swing the number by fifteen points for no reason other than the
  plugin's own dice. Anything above 100% comes from a clean night — nobody
  called, nothing interrupted — which is a fact about the night rather than
  about the random draw. (The original model could only exceed 100 through
  random jitter, and its configured ceiling of 120 was unreachable.)
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
MAX_COVERAGE_RATIO = 1.25  # only reachable when an explicit target is configured
CLEAN_NIGHT_BONUS = 3.0  # nobody called, nothing interrupted, one unbroken stretch
NIGHT_DUTY_DEBIT = 0.5  # half of a night-duty stretch does not count as rest
MAX_DUTY_RATIO = 0.25  # night duty can never eat more than a quarter of the night


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
    bonus_clean_night: float
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
        # Explicit operator override: measure against a fixed number of hours.
        target_seconds = snap.sleep_target_hours * 3600
    else:
        # Default: this night's own plan. The wake-up point is drawn at random
        # from a range, so measuring against anything fixed would make an early
        # draw look like a bad night when nothing bad happened.
        target_seconds = max(
            1800.0, (cycle.planned_wake_at - cycle.sleep_at).total_seconds()
        )
    target_hours = target_seconds / 3600

    # Night duty overlaps the sleep segment rather than breaking it: the bot did
    # not get out of bed, so half the stretch is charged as lost rest instead of
    # the whole thing being counted either way. The total is capped because the
    # plugin books an estimate per system message rather than a measured
    # duration, and a chatty night should not be able to run away with the score.
    duty = min(night_duty_seconds(cycle), MAX_DUTY_RATIO * target_seconds)
    effective = max(0.0, max(0.0, actual_sleep_seconds) - NIGHT_DUTY_DEBIT * duty)

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

    undisturbed = (
        calls == 0 and wakes == 0 and segments <= 1 and duty <= 0 and coverage_ratio >= 0.995
    )
    bonus_clean_night = CLEAN_NIGHT_BONUS if undisturbed else 0.0

    jitter = compute_stable_jitter(
        cycle.cycle_id,
        cycle.sleep_date,
        cycle.quality_seed,
        snap.quality_jitter_points,
    )

    raw = (
        base
        - penalty_fragmentation
        - penalty_calls
        - penalty_wakes
        + bonus_clean_night
        + jitter
    )
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
        bonus_clean_night=round(bonus_clean_night, 2),
        jitter=round(jitter, 3),
        raw=round(raw, 2),
        segments=segments,
        calls=calls,
        wakes=wakes,
    )


def compute_quality(cycle: SleepCycle, actual_sleep_seconds: float) -> int:
    """Final quality percentage. See `compute_quality_detail` for the terms."""
    return compute_quality_detail(cycle, actual_sleep_seconds).score
