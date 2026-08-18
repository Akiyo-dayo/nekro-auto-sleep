"""Tests for the host-facing wiring layer (`nekro_auto_sleep/__init__.py`).

Everything the domain-layer suites cannot see lives here: the order in which
settlement closes and scores a cycle, which MsgSignal each situation returns,
whether the persona name survives the round trip, what the prompt injection says
on the second round after a wake-up, and what happens after a restart.

Every v1 defect these cover was invisible to `test_engine.py` / `test_quality.py`
because those build their inputs by hand and call the domain functions directly.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

import nekro_auto_sleep as m
from nekro_auto_sleep.engine import (
    compute_actual_sleep_seconds,
    handle_valid_call_while_asleep,
    transition_to_sleep,
)
from nekro_auto_sleep.models import ChatSleepState, SleepStatus
from nekro_auto_sleep.persistence import SleepStateStore
from nekro_auto_sleep.quality import compute_quality
from tests.hoststub import (
    AgentCtx,
    ChatMessage,
    FakeChatChannel,
    MsgSignal,
    reset_ctx_factory,
    set_ctx_factory,
)
from tests.conftest import FakeStoreBackend

CHAT_KEY = "onebot_v11-group_123456789"
TZ = ZoneInfo("Asia/Shanghai")
UTC = ZoneInfo("UTC")
PERSONA = "阿绫"

# 2026-08-17 23:00 +08:00
BEDTIME = datetime(2026, 8, 17, 15, 0, tzinfo=UTC)


class FrozenClock:
    """Pinned clock for the wiring layer (`m._utcnow` is the only seam)."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock(monkeypatch):
    c = FrozenClock(BEDTIME + timedelta(hours=2))
    monkeypatch.setattr(m, "_utcnow", c)
    return c


@pytest.fixture
def wired(monkeypatch, clock):
    """A plugin wired to an in-memory store, with config restored afterwards."""
    backend = FakeStoreBackend()
    store = SleepStateStore(backend)
    monkeypatch.setattr(m, "_store", store)

    saved = m.config.model_dump()
    m.config.TIMEZONE = "Asia/Shanghai"
    m.config.SLEEP_TIME = "23:00"
    m.config.WAKE_TIME_START = "06:45"
    m.config.WAKE_TIME_END = "08:30"
    m.config.HISTORY_MODE = "preserve"
    m.config.WAKE_NOTICE_POLICY = "always"
    m.config.FALLBACK_PERSONA_NAME = "Bot"

    ctx = AgentCtx(CHAT_KEY, FakeChatChannel(CHAT_KEY, preset_name=PERSONA))
    set_ctx_factory(lambda chat_key: ctx)

    yield m, store, ctx

    reset_ctx_factory()
    for k, v in saved.items():
        setattr(m.config, k, v)


async def _put_asleep(store: SleepStateStore, now=BEDTIME) -> ChatSleepState:
    snap = m._make_config_snapshot()
    state = transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), now, snap)
    await store.save(state)
    return state


def _parse_notice(text: str) -> tuple[int, float]:
    """Pull (quality percent, duration in hours) out of a wake-up report."""
    quality = int(re.search(r"睡眠质量 (\d+)%", text).group(1))
    hours = 0.0
    h = re.search(r"(\d+) 小时", text)
    if h:
        hours += int(h.group(1))
    mins = re.search(r"(\d+) 分钟", text)
    if mins:
        hours += int(mins.group(1)) / 60
    return quality, hours


# ---------------------------------------------------------------------------
# Settlement: the score and the duration must describe the same night
# ---------------------------------------------------------------------------


class TestSettlement:
    async def test_quality_and_duration_come_from_the_same_snapshot(self, wired):
        plugin_mod, store, ctx = wired
        state = await _put_asleep(store)
        wake_at = state.cycle.planned_wake_at

        await plugin_mod._settle_wake(store, CHAT_KEY, wake_at)

        assert len(ctx.sent) == 1, ctx.sent
        reported_quality, reported_hours = _parse_notice(ctx.sent[0][0])

        settled = store.get_cached(CHAT_KEY)
        assert settled.status == SleepStatus.AWAKE
        actual_seconds = compute_actual_sleep_seconds(settled.cycle)

        # The number in the message must be the score of the cycle as settled.
        assert reported_quality == compute_quality(settled.cycle, actual_seconds)
        assert reported_hours == pytest.approx(actual_seconds / 3600, abs=0.02)

    async def test_undisturbed_night_does_not_report_the_floor(self, wired):
        """Scoring before the final segment closes measured 0s of sleep and
        pinned every quiet night to QUALITY_MIN while the duration stayed right."""
        plugin_mod, store, ctx = wired
        state = await _put_asleep(store)

        await plugin_mod._settle_wake(store, CHAT_KEY, state.cycle.planned_wake_at)

        quality, hours = _parse_notice(ctx.sent[0][0])
        assert hours > 8
        assert quality > m.config.QUALITY_MIN
        assert quality >= 95

    async def test_notice_suppressed_past_the_grace_period(self, wired):
        plugin_mod, store, ctx = wired
        state = await _put_asleep(store)
        m.config.WAKE_NOTICE_GRACE_MINUTES = 120

        very_late = state.cycle.planned_wake_at + timedelta(hours=5)
        await plugin_mod._settle_wake(store, CHAT_KEY, very_late)

        assert ctx.sent == []
        assert store.get_cached(CHAT_KEY).status == SleepStatus.AWAKE

    async def test_notice_sent_within_the_grace_period(self, wired):
        plugin_mod, store, ctx = wired
        state = await _put_asleep(store)
        m.config.WAKE_NOTICE_GRACE_MINUTES = 120

        slightly_late = state.cycle.planned_wake_at + timedelta(minutes=30)
        await plugin_mod._settle_wake(store, CHAT_KEY, slightly_late)

        assert len(ctx.sent) == 1
        assert PERSONA in ctx.sent[0][0]


# ---------------------------------------------------------------------------
# Message signals: the night must stay in the history
# ---------------------------------------------------------------------------


class TestMessageSignals:
    async def test_ordinary_night_message_is_still_recorded(self, wired):
        plugin_mod, store, ctx = wired
        await _put_asleep(store)

        signal = await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content="今天真累啊", chat_key=CHAT_KEY)
        )

        # BLOCK_ALL would make the host return before DBChatMessage.create,
        # leaving the bot with no memory of the night at all.
        assert signal == MsgSignal.BLOCK_TRIGGER
        assert ctx.sent == []

    async def test_strict_mode_still_drops_the_record(self, wired):
        plugin_mod, store, ctx = wired
        m.config.HISTORY_MODE = "strict"
        await _put_asleep(store)

        signal = await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content="今天真累啊", chat_key=CHAT_KEY)
        )
        assert signal == MsgSignal.BLOCK_ALL

    async def test_wake_offer_is_recorded_in_preserve_mode(self, wired):
        plugin_mod, store, ctx = wired
        await _put_asleep(store)

        signal = await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content="醒醒", chat_key=CHAT_KEY)
        )

        assert signal == MsgSignal.BLOCK_TRIGGER
        assert len(ctx.sent) == 1
        text, record = ctx.sent[0]
        assert f"【{PERSONA}已经睡了 要叫醒{PERSONA}吗？】" == text
        # Recorded, otherwise the woken bot sees the same user calling twice
        # with no turn of its own in between.
        assert record is True

    async def test_second_call_forces_the_agent_round(self, wired):
        plugin_mod, store, ctx = wired
        await _put_asleep(store)

        await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content="醒醒", chat_key=CHAT_KEY)
        )
        signal = await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content="醒醒", chat_key=CHAT_KEY)
        )

        assert signal == MsgSignal.FORCE_TRIGGER
        state = store.get_cached(CHAT_KEY)
        assert state.status == SleepStatus.AWAKE_EARLY
        assert state.woken_by == "u1"
        assert state.woken_at is not None


# ---------------------------------------------------------------------------
# Persona name
# ---------------------------------------------------------------------------


class TestPersonaName:
    async def test_resolves_the_preset_name(self, wired):
        plugin_mod, _store, ctx = wired
        # `db_chat_channel` is a plain property; awaiting it raised TypeError and
        # a bare except turned every persona into the configured fallback.
        assert await plugin_mod._get_persona_name(ctx) == PERSONA

    async def test_falls_back_only_when_there_is_no_channel(self, wired):
        plugin_mod, _store, _ctx = wired
        bare = AgentCtx(CHAT_KEY, None)
        assert await plugin_mod._get_persona_name(bare) == m.config.FALLBACK_PERSONA_NAME


# ---------------------------------------------------------------------------
# Wake context: the reply after a wake-up has to make sense
# ---------------------------------------------------------------------------


class TestWakeContext:
    async def _wake_up(self, plugin_mod, store, ctx):
        await _put_asleep(store)
        await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content="醒醒", chat_key=CHAT_KEY)
        )
        await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content="醒醒", chat_key=CHAT_KEY)
        )

    async def test_context_survives_more_than_one_round(self, wired):
        plugin_mod, store, ctx = wired
        await self._wake_up(plugin_mod, store, ctx)
        inject = plugin_mod.plugin.prompt_injects["sleep_status"]

        first = await inject(ctx)
        second = await inject(ctx)

        # The old one-shot cache was popped by the first round, so every later
        # reply had no idea it was the middle of the night.
        assert first.strip()
        assert second.strip()
        assert "resume_sleep" in second

    async def test_context_names_the_bedtime_and_the_waker(self, wired):
        plugin_mod, store, ctx = wired
        await self._wake_up(plugin_mod, store, ctx)

        text = await plugin_mod.plugin.prompt_injects["sleep_status"](ctx)

        assert "23:00" in text  # bedtime, rendered in the cycle timezone
        assert "u1" in text  # who did the waking
        assert "睡着时收到" in text  # night messages must not be treated as live

    async def test_just_woken_line_expires(self, wired):
        plugin_mod, store, ctx = wired
        await self._wake_up(plugin_mod, store, ctx)
        state = store.get_cached(CHAT_KEY)

        fresh = plugin_mod._render_sleep_context(state, state.woken_at + timedelta(seconds=5))
        later = plugin_mod._render_sleep_context(state, state.woken_at + timedelta(minutes=30))

        assert "刚刚被叫醒" in fresh
        assert "刚刚被叫醒" not in later
        assert later.strip()  # the rest of the context stays

    async def test_no_injection_when_awake(self, wired):
        plugin_mod, store, ctx = wired
        await store.save(ChatSleepState(chat_key=CHAT_KEY))
        assert await plugin_mod.plugin.prompt_injects["sleep_status"](ctx) == ""


# ---------------------------------------------------------------------------
# Restart behaviour
# ---------------------------------------------------------------------------


class TestBootReconcile:
    async def test_settles_a_wake_up_that_happened_while_down(self, wired, monkeypatch):
        plugin_mod, store, ctx = wired
        state = await _put_asleep(store)
        wake_at = state.cycle.planned_wake_at

        store.clear_all()  # a restart: nothing in the cache
        monkeypatch.setattr(
            plugin_mod, "_discover_chat_keys", lambda: _async_set({CHAT_KEY})
        )

        await plugin_mod._boot_reconcile(store, wake_at + timedelta(minutes=20))

        assert store.get_cached(CHAT_KEY).status == SleepStatus.AWAKE
        assert len(ctx.sent) == 1
        assert "已起床" in ctx.sent[0][0]

    async def test_goes_to_bed_for_a_night_that_started_while_down(
        self, wired, monkeypatch
    ):
        plugin_mod, store, ctx = wired
        monkeypatch.setattr(
            plugin_mod, "_discover_chat_keys", lambda: _async_set({CHAT_KEY})
        )

        # Booting at 02:00 local, three hours past bedtime.
        now = BEDTIME + timedelta(hours=3)
        await plugin_mod._boot_reconcile(store, now)

        state = store.get_cached(CHAT_KEY)
        assert state.status == SleepStatus.ASLEEP
        assert state.cycle.sleep_date == "2026-08-17"
        # Backdated, otherwise the morning report claims a three-hour night.
        assert state.cycle.sleep_segments[0].open_at == state.cycle.sleep_at

    async def test_hydrates_channels_that_never_spoke(self, wired, monkeypatch):
        plugin_mod, store, _ctx = wired
        monkeypatch.setattr(
            plugin_mod, "_discover_chat_keys", lambda: _async_set({CHAT_KEY})
        )

        # Daytime: nothing to settle, but the channel must still be watched or it
        # will never fall asleep tonight.
        await plugin_mod._boot_reconcile(store, BEDTIME - timedelta(hours=6))

        assert CHAT_KEY in store.known_chat_keys()


class TestBedtimeDetection:
    async def test_bedtime_is_not_missed_by_a_slow_maintenance_loop(self, wired):
        plugin_mod, store, _ctx = wired
        await store.hydrate(CHAT_KEY)

        # A tick 40 minutes after bedtime: the old 30-second look-back window
        # skipped the night entirely whenever the interval was raised.
        late_tick = BEDTIME + timedelta(minutes=40)
        await plugin_mod._check_sleep_transition(store, CHAT_KEY, late_tick, TZ)

        state = store.get_cached(CHAT_KEY)
        assert state.status == SleepStatus.ASLEEP
        assert state.cycle.sleep_date == "2026-08-17"

    async def test_does_not_restart_the_same_night_twice(self, wired):
        plugin_mod, store, _ctx = wired
        await store.hydrate(CHAT_KEY)

        first_tick = BEDTIME + timedelta(minutes=5)
        await plugin_mod._check_sleep_transition(store, CHAT_KEY, first_tick, TZ)
        opened_at = store.get_cached(CHAT_KEY).cycle.sleep_segments[0].open_at

        await plugin_mod._check_sleep_transition(
            store, CHAT_KEY, BEDTIME + timedelta(hours=2), TZ
        )

        state = store.get_cached(CHAT_KEY)
        assert len(state.cycle.sleep_segments) == 1
        assert state.cycle.sleep_segments[0].open_at == opened_at

    async def test_does_not_nap_after_this_nights_wake_up(self, wired):
        plugin_mod, store, _ctx = wired
        await store.hydrate(CHAT_KEY)

        # 08:15 local, inside the wake range but past most wake points.
        morning = datetime(2026, 8, 18, 0, 15, tzinfo=UTC)
        await plugin_mod._check_sleep_transition(store, CHAT_KEY, morning, TZ)

        state = store.get_cached(CHAT_KEY)
        if state.cycle is not None and morning >= state.cycle.planned_wake_at:
            assert state.status == SleepStatus.AWAKE


async def _async_set(value):
    return value
