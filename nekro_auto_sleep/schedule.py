"""Time-zone handling, cycle boundary computation, and random wake-up selection.

Pure functions — no host imports, no side effects beyond the returned values.
"""

from __future__ import annotations

import hashlib
import random
from datetime import date, datetime, time, timedelta

from zoneinfo import ZoneInfo

from .models import ConfigSnapshot


def parse_hhmm(s: str) -> time:
    """Parse 'HH:MM' into a time object."""
    parts = s.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM format: {s!r}")
    return time(int(parts[0]), int(parts[1]))


def compute_cycle_boundaries(
    sleep_date: date,
    tz: ZoneInfo,
    sleep_time_str: str,
    wake_start_str: str,
    wake_end_str: str,
) -> tuple[datetime, datetime, datetime]:
    """Return (sleep_at, wake_range_start, wake_range_end) as aware UTC datetimes.

    Rules from spec §4.1:
    - sleep_at = sleep_date + SLEEP_TIME in tz
    - wake range endpoints are the first local datetime strictly after sleep_at
    - if wake_range_end <= wake_range_start, add one more day to end
    """
    st = parse_hhmm(sleep_time_str)
    ws = parse_hhmm(wake_start_str)
    we = parse_hhmm(wake_end_str)

    sleep_at_local = datetime.combine(sleep_date, st, tzinfo=tz)
    sleep_at_utc = sleep_at_local.astimezone(ZoneInfo("UTC"))

    wake_start_local = datetime.combine(sleep_date, ws, tzinfo=tz)
    if wake_start_local <= sleep_at_local:
        wake_start_local += timedelta(days=1)

    wake_end_local = datetime.combine(sleep_date, we, tzinfo=tz)
    if wake_end_local <= sleep_at_local:
        wake_end_local += timedelta(days=1)
    if wake_end_local <= wake_start_local:
        wake_end_local += timedelta(days=1)

    wake_start_utc = wake_start_local.astimezone(ZoneInfo("UTC"))
    wake_end_utc = wake_end_local.astimezone(ZoneInfo("UTC"))

    return sleep_at_utc, wake_start_utc, wake_end_utc


def build_wake_candidates(
    wake_start: datetime,
    wake_end: datetime,
    step_minutes: int,
) -> list[datetime]:
    """Build candidate wake-up times in [wake_start, wake_end] at step_minutes intervals."""
    step_minutes = max(1, min(60, step_minutes))
    step = timedelta(minutes=step_minutes)
    candidates: list[datetime] = []
    t = wake_start
    while t <= wake_end:
        candidates.append(t)
        t += step
    if not candidates:
        candidates.append(wake_start)
    return candidates


def pick_wake_time(
    chat_key: str,
    sleep_date: date,
    wake_start: datetime,
    wake_end: datetime,
    step_minutes: int,
) -> datetime:
    """Deterministically pick a random wake-up time for a given chat_key and sleep_date.

    Uses a seeded RNG so the result is reproducible for the same inputs,
    but must be persisted on first call — restarts must not re-pick.
    """
    candidates = build_wake_candidates(wake_start, wake_end, step_minutes)
    seed = hashlib.sha256(f"{chat_key}:{sleep_date.isoformat()}:wake".encode()).hexdigest()
    rng = random.Random(seed)
    return rng.choice(candidates)


def generate_quality_seed(chat_key: str, sleep_date: date) -> str:
    """Generate a deterministic quality jitter seed."""
    return hashlib.sha256(
        f"{chat_key}:{sleep_date.isoformat()}:quality".encode()
    ).hexdigest()[:16]


def generate_cycle_id(chat_key: str, sleep_date: date) -> str:
    """Generate a unique cycle ID."""
    return hashlib.sha256(
        f"{chat_key}:{sleep_date.isoformat()}:cycle".encode()
    ).hexdigest()[:12]


def is_in_sleep_window(
    now_utc: datetime,
    sleep_at_utc: datetime,
    wake_at_utc: datetime,
) -> bool:
    """Check if now_utc is within [sleep_at, wake_at)."""
    return sleep_at_utc <= now_utc < wake_at_utc


def is_near_wake(
    now_utc: datetime,
    sleep_at_utc: datetime,
    wake_at_utc: datetime,
    near_wake_minutes: int,
) -> bool:
    """Whether now_utc falls in the final stretch before the planned wake-up.

    An absolute window rather than a fraction of the night: with a fraction the
    "not up yet" wording drifted every night along with the random wake point,
    and a longer configured night silently widened it.
    """
    minutes = max(0, min(720, near_wake_minutes))
    threshold = wake_at_utc - timedelta(minutes=minutes)
    return sleep_at_utc <= now_utc and now_utc >= threshold


def current_local_date(tz: ZoneInfo, now_utc: datetime | None = None) -> date:
    """Get current local date in the given timezone."""
    if now_utc is None:
        now_utc = datetime.now(ZoneInfo("UTC"))
    return now_utc.astimezone(tz).date()


def find_sleep_date_for_now(
    now_utc: datetime,
    tz: ZoneInfo,
    sleep_time_str: str,
    wake_start_str: str,
    wake_end_str: str,
) -> date | None:
    """If now_utc falls within a sleep window, return the sleep_date that owns it.

    Checks today and yesterday (local) since the sleep window crosses midnight.
    Returns None if not in any sleep window.
    """
    local_now = now_utc.astimezone(tz)
    for offset in (0, -1):
        candidate_date = local_now.date() + timedelta(days=offset)
        sleep_at, _wake_start, wake_end = compute_cycle_boundaries(
            candidate_date, tz, sleep_time_str, wake_start_str, wake_end_str
        )
        wake_at = wake_end
        if sleep_at <= now_utc < wake_at:
            return candidate_date
    return None


def next_sleep_at(
    now_utc: datetime,
    tz: ZoneInfo,
    sleep_time_str: str,
) -> datetime:
    """Compute the next sleep_at time that is strictly in the future."""
    st = parse_hhmm(sleep_time_str)
    local_now = now_utc.astimezone(tz)
    candidate = datetime.combine(local_now.date(), st, tzinfo=tz)
    if candidate <= local_now:
        candidate += timedelta(days=1)
    return candidate.astimezone(ZoneInfo("UTC"))


def parse_keyword_list(raw: str | list[str]) -> list[str]:
    """Split a comma/newline separated keyword field into a clean list."""
    if isinstance(raw, str):
        return [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]
    return [str(k).strip() for k in raw if str(k).strip()]


def create_config_snapshot(
    timezone: str,
    sleep_time: str,
    wake_time_start: str,
    wake_time_end: str,
    wake_random_step_minutes: int,
    near_wake_ratio: float,
    wake_confirm_window_seconds: int,
    history_mode: str,
    call_keywords: str | list[str],
    fallback_persona_name: str,
    early_wake_idle_minutes: int,
    quality_min: int,
    quality_max: int,
    quality_jitter_points: float,
    *,
    near_wake_minutes: int = 60,
    sleep_target_hours: float = 8.0,
    affirmative_keywords: str | list[str] = (),
    negative_keywords: str | list[str] = (),
    urgent_keywords: str | list[str] = (),
    answer_scope: str = "offeree",
    unclear_answer: str = "ignore",
    max_offers_per_night: int = 3,
    offer_cooldown_minutes: int = 20,
    snooze_minutes: int = 30,
    asleep_prompt: str = "【{persona}已经睡了 要叫醒{persona}吗？】",
    near_wake_prompt: str = "【{persona}还没起床 要叫醒{persona}吗？】",
) -> ConfigSnapshot:
    """Create a ConfigSnapshot from current config values."""
    return ConfigSnapshot(
        timezone=timezone,
        sleep_time=sleep_time,
        wake_time_start=wake_time_start,
        wake_time_end=wake_time_end,
        wake_random_step_minutes=wake_random_step_minutes,
        near_wake_ratio=near_wake_ratio,
        wake_confirm_window_seconds=wake_confirm_window_seconds,
        history_mode=history_mode,  # type: ignore[arg-type]
        call_keywords=parse_keyword_list(call_keywords),
        fallback_persona_name=fallback_persona_name,
        early_wake_idle_minutes=early_wake_idle_minutes,
        quality_min=quality_min,
        quality_max=quality_max,
        quality_jitter_points=quality_jitter_points,
        near_wake_minutes=near_wake_minutes,
        sleep_target_hours=sleep_target_hours,
        affirmative_keywords=parse_keyword_list(affirmative_keywords),
        negative_keywords=parse_keyword_list(negative_keywords),
        urgent_keywords=parse_keyword_list(urgent_keywords),
        answer_scope=answer_scope,  # type: ignore[arg-type]
        unclear_answer=unclear_answer,  # type: ignore[arg-type]
        max_offers_per_night=max_offers_per_night,
        offer_cooldown_minutes=offer_cooldown_minutes,
        snooze_minutes=snooze_minutes,
        asleep_prompt=asleep_prompt,
        near_wake_prompt=near_wake_prompt,
    )
