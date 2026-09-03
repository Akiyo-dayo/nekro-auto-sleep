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

    timer_intervals: list[tuple[float, float]] = []
    for ti in cycle.timer_intervals:
        start = ti.start_at.timestamp()
        end = ti.end_at.timestamp() if ti.end_at else wake_at_ts
        timer_intervals.append((start, end))

    early_awake_intervals: list[tuple[float, float]] = []
    segments = cycle.sleep_segments
    for i, seg in enumerate(segments):
        if seg.close_at is None:
            continue
        close_ts = seg.close_at.timestamp()
        if i + 1 < len(segments):
            next_open_ts = segments[i + 1].open_at.timestamp()
            early_awake_intervals.append((close_ts, next_open_ts))

    def _in_timer(t: float) -> bool:
        for s, e in timer_intervals:
            if s <= t < e:
                return True
        return False

    weighted_integral = 0.0
    t = sleep_at_ts
    while t < wake_at_ts:
        w = _stage_weight(t, sleep_at_ts, wake_at_ts)

        kernel_sum = 0.0
        for at in attempt_times:
            kernel_sum += _exp_kernel(t, at)

        # Timer-caused gaps are accounted by compute_timer_burden; counting them
        # here as well would double-charge the same disturbance.
        for start, end in early_awake_intervals:
            if start <= t < end and not _in_timer(t):
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


# ---------------------------------------------------------------------------
# §11.4 Fun layer: tier comments, deterministic dreams, streak notes
#
# All pickers are deterministic in (chat_key, sleep_date) so restarts and
# repeated settlements never change the story of a given night.
# ---------------------------------------------------------------------------

def stable_pick(seed: str, options: tuple[str, ...]) -> str:
    """Deterministic pick from ``options`` seeded by ``seed``."""
    h = hashlib.sha256(seed.encode()).digest()
    return options[struct.unpack(">I", h[:4])[0] % len(options)]


NICE_DREAMS: tuple[str, ...] = (
    "梦见自己变成一只猫，在洒满阳光的屋顶上睡了一整天",
    "梦见考试发现自己全都会，还提前交了卷",
    "梦见一顿火锅咕嘟咕嘟地冒着热气，就是够不着",
    "梦见在云端上散步，一脚踩进棉花糖里",
    "梦见钱包突然胖了一圈，数钱数到手抽筋",
    "梦见坐上了开往春天的慢火车，窗外全是花田",
    "梦见自己在海面上打水漂，一块钱漂出了十八个涟漪",
    "梦见回到小时候的夏天，蝉鸣和西瓜都很甜",
)
ODD_DREAMS: tuple[str, ...] = (
    "梦见自己永远排在一支看不见头的队伍里，前面的人还在原地转圈",
    "梦见手机永远充到 99%，怎么也充不满",
    "梦见自己在给一条鱼讲微积分，它听得直吐泡泡",
    "梦见客厅的沙发开口说话，还嫌我睡相不好",
    "梦见自己在一场重要会议里，但怎么也想不起来自己是谁",
    "梦见在洗澡突然想不起来刚才到底洗过没有",
)
BAD_DREAMS: tuple[str, ...] = (
    "梦见闹钟响了一遍又一遍，怎么按都按不掉（吓醒了发现没有闹钟）",
    "梦见被一只巨大的 Deadline 追着跑，跑鞋还丢了",
    "梦见手机屏幕裂成了蜘蛛网，弹出一条条红色的提醒",
    "梦见站在全班面前，突然发现自己一个字都念不出来",
    "梦见自己变成了一块正在被格式化的硬盘",
)

QUALITY_TIERS: tuple[tuple[int, str, str, tuple[str, ...]], ...] = (
    (
        110,
        "神清气爽",
        "✨",
        ("感觉像充了格电，随时能起飞！", "今天的世界格外清晰，元气满格！"),
    ),
    (
        95,
        "睡得不错",
        "😴",
        ("睡了个好觉，皮肤都在发光。", "一觉到天亮，神清气爽。"),
    ),
    (
        80,
        "睡得一般",
        "🌤",
        ("睡得还行，就是有点意犹未尽。", "马马虎虎，再睡五分钟就完美了。"),
    ),
    (
        70,
        "睡得迷糊",
        "🌀",
        ("脑袋还像浆糊，谁跟我说话都要缓冲一下。", "睡是睡了，但总感觉被谁偷走了半个梦。"),
    ),
    (
        0,
        "睡眼惺忪",
        "🥱",
        ("这觉跟没睡一样……让我先缓缓。", "浑身不得劲，今天谁也别惹我。"),
    ),
)


def quality_tier(percent: int) -> tuple[str, str, str]:
    """Return (tier_name, emoji, comment) for a quality percent."""
    for threshold, name, emoji, comments in QUALITY_TIERS:
        if percent >= threshold:
            return name, emoji, comments[percent % len(comments)]
    name, emoji, comments = QUALITY_TIERS[-1]
    return name, emoji, comments[percent % len(comments)]


def pick_dream(seed: str, percent: int) -> str | None:
    """Deterministic dream line for the night; None on exceptionally good sleep.

    Higher quality tends to nice dreams, poor quality to nightmares. A perfect
    night (>= 115) skips the dream line entirely — sleeping like a log means
    remembering nothing.
    """
    if percent >= 115:
        return None
    if percent >= 95:
        return stable_pick(seed, NICE_DREAMS)
    if percent >= 70:
        return stable_pick(seed, ODD_DREAMS)
    return stable_pick(seed, BAD_DREAMS)


def compute_streak_note(history: dict[str, int], sleep_date: str, good_threshold: int = 95) -> str | None:
    """One-line streak/trend note based on settled quality history.

    ``history`` maps sleep_date (YYYY-MM-DD) -> quality percent for the current
    night and all previous nights kept by the retention policy.
    """
    if sleep_date not in history:
        return None

    dates = sorted(d for d in history if d < sleep_date)
    if not dates:
        return "第一天记录睡眠打卡，开个好头～"

    yesterday = dates[-1]
    diff = history[sleep_date] - history[yesterday]

    streak = 0
    d = yesterday
    while d in history and history[d] >= good_threshold:
        streak += 1
        prev = (datetime.fromisoformat(d).date() - timedelta(days=1)).isoformat()
        if prev in history and history[prev] >= good_threshold:
            d = prev
        else:
            break

    notes: list[str] = []
    if streak >= 2:
        notes.append(f"已经连续 {streak + 1} 天睡得不错")
    elif history[sleep_date] >= good_threshold:
        notes.append("今晚算是个好开头")
    if diff >= 10:
        notes.append(f"比昨晚多睡了 {diff} 分的含金量")
    elif diff <= -10:
        notes.append(f"比昨晚掉了 {-diff} 分，今晚早点睡")
    if not notes:
        return None
    return "，".join(notes)
