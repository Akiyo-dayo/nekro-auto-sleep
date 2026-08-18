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
    transition_to_sleep,
)
from nekro_auto_sleep.models import ChatSleepState, ScheduleOverride, SleepStatus
from nekro_auto_sleep.persistence import ScheduleOverrideStore, SleepStateStore
from nekro_auto_sleep.quality import compute_quality
from tests.hoststub import (
    AgentCtx,
    clear_instance_config_overrides,
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
    monkeypatch.setattr(m, "_overrides", ScheduleOverrideStore(backend))
    store.test_backend = backend  # write counter, for the quiet-tick assertions
    m.invalidate_settings_cache()

    saved = m.config.model_dump()
    m.config.TIMEZONE = "Asia/Shanghai"
    m.config.SLEEP_TIME = "23:00"
    m.config.WAKE_TIME_START = "06:45"
    m.config.WAKE_TIME_END = "08:30"
    m.config.HISTORY_MODE = "preserve"
    m.config.WAKE_NOTICE_POLICY = "always"
    m.config.FALLBACK_PERSONA_NAME = "Bot"
    m.config.ANSWER_SCOPE = "offeree"
    m.config.URGENT_KEYWORDS = ""
    m.config.MAX_OFFERS_PER_NIGHT = 3
    m.config.OFFER_COOLDOWN_MINUTES = 20
    m.config.SNOOZE_MINUTES = 30

    ctx = AgentCtx(CHAT_KEY, FakeChatChannel(CHAT_KEY, preset_name=PERSONA))
    set_ctx_factory(lambda chat_key: ctx)

    yield m, store, ctx

    reset_ctx_factory()
    clear_instance_config_overrides()
    m.invalidate_settings_cache()
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
            ctx, ChatMessage(content="要", chat_key=CHAT_KEY)
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

    async def test_looks_the_channel_up_when_the_ctx_has_none(self, wired):
        """Upstream can hand back a ctx with no channel prefilled."""
        plugin_mod, _store, _ctx = wired
        bare = AgentCtx(CHAT_KEY, None)
        assert await plugin_mod._get_persona_name(bare) == "小助手"

    async def test_falls_back_when_the_lookup_fails(self, wired, monkeypatch):
        from nekro_agent.models import db_chat_channel

        plugin_mod, _store, _ctx = wired

        async def _boom(chat_key: str = ""):
            raise RuntimeError("no database here")

        monkeypatch.setattr(db_chat_channel.DBChatChannel, "get_channel", _boom)
        bare = AgentCtx(CHAT_KEY, None)
        assert await plugin_mod._get_persona_name(bare) == m.config.FALLBACK_PERSONA_NAME


# ---------------------------------------------------------------------------
# Wake context: the reply after a wake-up has to make sense
# ---------------------------------------------------------------------------


class TestWakeContext:
    async def _wake_up(self, plugin_mod, store, ctx):
        """Call, answer, and then carry on — so the bot is properly awake.

        The third message is what ends the decision round: the model replied,
        the user said something else, and from here the injection describes an
        awake bot in the middle of the night rather than one still deciding.
        """
        await _put_asleep(store)
        await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content="醒醒", chat_key=CHAT_KEY)
        )
        await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content="要", chat_key=CHAT_KEY)
        )
        await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content="帮我看个东西", chat_key=CHAT_KEY)
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
        await plugin_mod._check_sleep_transition(store, CHAT_KEY, late_tick)

        state = store.get_cached(CHAT_KEY)
        assert state.status == SleepStatus.ASLEEP
        assert state.cycle.sleep_date == "2026-08-17"

    async def test_does_not_restart_the_same_night_twice(self, wired):
        plugin_mod, store, _ctx = wired
        await store.hydrate(CHAT_KEY)

        first_tick = BEDTIME + timedelta(minutes=5)
        await plugin_mod._check_sleep_transition(store, CHAT_KEY, first_tick)
        opened_at = store.get_cached(CHAT_KEY).cycle.sleep_segments[0].open_at

        await plugin_mod._check_sleep_transition(
            store, CHAT_KEY, BEDTIME + timedelta(hours=2)
        )

        state = store.get_cached(CHAT_KEY)
        assert len(state.cycle.sleep_segments) == 1
        assert state.cycle.sleep_segments[0].open_at == opened_at

    async def test_does_not_nap_after_this_nights_wake_up(self, wired):
        plugin_mod, store, _ctx = wired
        await store.hydrate(CHAT_KEY)

        # 08:15 local, inside the wake range but past most wake points.
        morning = datetime(2026, 8, 18, 0, 15, tzinfo=UTC)
        await plugin_mod._check_sleep_transition(store, CHAT_KEY, morning)

        state = store.get_cached(CHAT_KEY)
        if state.cycle is not None and morning >= state.cycle.planned_wake_at:
            assert state.status == SleepStatus.AWAKE


async def _async_set(value):
    return value


# ---------------------------------------------------------------------------
# Wake protocol, end to end through the hook
# ---------------------------------------------------------------------------


class TestWakeProtocolWiring:
    async def _ask(self, plugin_mod, ctx, text="醒醒"):
        return await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content=text, chat_key=CHAT_KEY)
        )

    async def test_the_second_message_reaches_the_llm(self, wired):
        plugin_mod, store, ctx = wired
        await _put_asleep(store)

        await self._ask(plugin_mod, ctx)
        signal = await self._ask(plugin_mod, ctx, "要")

        assert signal == MsgSignal.FORCE_TRIGGER
        state = store.get_cached(CHAT_KEY)
        assert state.status == SleepStatus.AWAKE_EARLY
        assert state.wake_decision_pending is True

    async def test_a_refusal_also_reaches_the_llm(self, wired):
        """The plugin does not decide what 「算了你睡吧」 meant; the model does."""
        plugin_mod, store, ctx = wired
        await _put_asleep(store)

        await self._ask(plugin_mod, ctx)
        signal = await self._ask(plugin_mod, ctx, "算了你睡吧")

        assert signal == MsgSignal.FORCE_TRIGGER
        assert store.get_cached(CHAT_KEY).wake_decision_pending is True
        # Still only the question has been sent.
        assert len(ctx.sent) == 1

    async def test_the_model_declining_is_silent(self, wired):
        plugin_mod, store, ctx = wired
        await _put_asleep(store)
        await self._ask(plugin_mod, ctx)
        await self._ask(plugin_mod, ctx, "算了你睡吧")
        ctx.sent.clear()

        await plugin_mod.plugin.sandbox_methods["resume_sleep"](ctx)

        assert ctx.sent == [], "declining the wake must not send anything"
        state = store.get_cached(CHAT_KEY)
        assert state.status == SleepStatus.ASLEEP
        assert state.snooze_until is not None

    async def test_turning_in_after_a_real_conversation_announces(self, wired):
        plugin_mod, store, ctx = wired
        await _put_asleep(store)
        await self._ask(plugin_mod, ctx)
        await self._ask(plugin_mod, ctx, "要")
        # The model replied and the user carried on, ending the decision round.
        await self._ask(plugin_mod, ctx, "帮我查个东西")
        ctx.sent.clear()

        await plugin_mod.plugin.sandbox_methods["resume_sleep"](ctx)

        assert len(ctx.sent) == 1
        assert "已睡下" in ctx.sent[0][0]

    async def test_the_decision_round_gets_its_own_instructions(self, wired):
        plugin_mod, store, ctx = wired
        await _put_asleep(store)
        await self._ask(plugin_mod, ctx)
        await self._ask(plugin_mod, ctx, "嗯？")

        text = await plugin_mod.plugin.prompt_injects["sleep_status"](ctx)

        assert "要不要叫醒你" in text
        assert "resume_sleep" in text
        assert "不要输出任何内容" in text
        assert PERSONA in text

    async def test_repeated_calls_do_not_spam_the_chat(self, wired, clock):
        plugin_mod, store, ctx = wired
        await _put_asleep(store)

        for _ in range(8):
            await self._ask(plugin_mod, ctx, "在吗")
            clock.advance(minutes=4)

        # Eight valid calls across 32 minutes, two replies: one at the start and
        # one after the 20-minute cooldown. Before the rate limit every single
        # one of them got its own fixed reply and its own quality penalty.
        assert len(ctx.sent) == 2
        assert store.get_cached(CHAT_KEY).offers_sent_tonight == 2

    async def test_urgent_message_skips_the_handshake(self, wired):
        plugin_mod, store, ctx = wired
        m.config.URGENT_KEYWORDS = "紧急,急事,救命,出事了"
        await _put_asleep(store)

        signal = await self._ask(plugin_mod, ctx, "出事了 快醒醒")

        assert signal == MsgSignal.FORCE_TRIGGER
        state = store.get_cached(CHAT_KEY)
        assert state.status == SleepStatus.AWAKE_EARLY
        assert state.woken_reason == "urgent"
        assert ctx.sent == []

    async def test_urgent_wake_is_reflected_in_the_prompt(self, wired):
        plugin_mod, store, ctx = wired
        m.config.URGENT_KEYWORDS = "紧急,急事,救命,出事了"
        await _put_asleep(store)
        # Urgent still has to be aimed at the bot: an @ or a call keyword.
        await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content="救命", chat_key=CHAT_KEY, is_tome=True)
        )

        text = await plugin_mod.plugin.prompt_injects["sleep_status"](ctx)
        assert "紧急叫醒" in text


class TestQualityBreakdownPersisted:
    async def test_settlement_stores_every_term(self, wired):
        plugin_mod, store, ctx = wired
        state = await _put_asleep(store)

        await plugin_mod._settle_wake(store, CHAT_KEY, state.cycle.planned_wake_at)

        breakdown = store.get_cached(CHAT_KEY).cycle.quality_breakdown
        assert breakdown is not None
        reported, _hours = _parse_notice(ctx.sent[0][0])
        assert breakdown["score"] == reported
        assert set(breakdown) >= {
            "base",
            "penalty_fragmentation",
            "penalty_calls",
            "penalty_wakes",
            "jitter",
            "raw",
            "target_hours",
            "effective_hours",
        }


# ---------------------------------------------------------------------------
# Night-time scheduled tasks
# ---------------------------------------------------------------------------


class TestNightTimerPolicy:
    async def test_run_lets_the_reminder_through(self, wired):
        plugin_mod, store, ctx = wired
        m.config.NIGHT_TIMER_POLICY = "run"
        await _put_asleep(store)

        signal = await plugin_mod.plugin.on_system_message(ctx, "定时提醒：吃药")

        # The host only starts a round for callers passing trigger_agent=True,
        # so letting this through wakes the timer service and nothing else.
        assert signal == MsgSignal.CONTINUE

    async def test_run_charges_the_round_to_sleep_quality(self, wired):
        plugin_mod, store, ctx = wired
        m.config.NIGHT_TIMER_POLICY = "run"
        m.config.NIGHT_DUTY_ASSUMED_MINUTES = 6
        await _put_asleep(store)

        await plugin_mod.plugin.on_system_message(ctx, "定时提醒：吃药")

        cycle = store.get_cached(CHAT_KEY).cycle
        assert len(cycle.timer_intervals) == 1
        interval = cycle.timer_intervals[0]
        assert (interval.end_at - interval.start_at) == timedelta(minutes=6)
        # Night duty overlays the night; it does not break the sleep segment.
        assert cycle.sleep_segments[-1].close_at is None

    async def test_block_keeps_the_reminder_out_of_the_llm(self, wired):
        plugin_mod, store, ctx = wired
        m.config.NIGHT_TIMER_POLICY = "block"
        await _put_asleep(store)

        signal = await plugin_mod.plugin.on_system_message(ctx, "定时提醒：吃药")

        assert signal == MsgSignal.BLOCK_TRIGGER
        assert store.get_cached(CHAT_KEY).cycle.timer_intervals == []

    async def test_awake_channels_are_untouched(self, wired):
        plugin_mod, store, ctx = wired
        m.config.NIGHT_TIMER_POLICY = "block"
        await store.save(ChatSleepState(chat_key=CHAT_KEY))

        assert await plugin_mod.plugin.on_system_message(ctx, "x") == MsgSignal.CONTINUE

    async def test_the_gate_only_blocks_under_the_block_policy(self, wired):
        plugin_mod, store, _ctx = wired
        await _put_asleep(store)

        m.config.NIGHT_TIMER_POLICY = "run"
        assert plugin_mod._night_timer_blocked(CHAT_KEY) is False

        m.config.NIGHT_TIMER_POLICY = "block"
        assert plugin_mod._night_timer_blocked(CHAT_KEY) is True

    async def test_night_duty_prompt_tells_the_bot_it_is_on_duty(self, wired):
        plugin_mod, store, ctx = wired
        await _put_asleep(store)

        text = await plugin_mod.plugin.prompt_injects["sleep_status"](ctx)
        assert "夜里值班" in text
        assert "继续睡" in text


class TestScheduleGate:
    async def test_wrapper_is_actually_installed_and_reversible(self):
        """The first version registered the timer methods for restoration
        without ever wrapping them, so the gate never existed."""
        from nekro_auto_sleep.runtime import (
            is_wrapped,
            make_schedule_agent_task_wrapper,
            unwrap_callable,
            wrap_callable,
        )

        class FakeService:
            def __init__(self):
                self.calls = []

            async def schedule_agent_task(self, chat_key):
                self.calls.append(chat_key)
                return "ran"

        service = FakeService()
        blocked = {"value": False}
        wrapper = make_schedule_agent_task_wrapper(lambda ck: blocked["value"])

        assert wrap_callable(service, "schedule_agent_task", wrapper) is True
        assert is_wrapped(service, "schedule_agent_task")

        assert await service.schedule_agent_task(CHAT_KEY) == "ran"
        blocked["value"] = True
        assert await service.schedule_agent_task(CHAT_KEY) is None
        assert service.calls == [CHAT_KEY]

        assert unwrap_callable(service, "schedule_agent_task") is True
        assert not is_wrapped(service, "schedule_agent_task")
        assert await service.schedule_agent_task(CHAT_KEY) == "ran"


class TestInstallWraps:
    """`_install_wraps` must actually wrap, not just remember to unwrap.

    The first version appended `(timer_service, "_execute_task")` to the
    restore list without ever calling `wrap_callable`, so the gate it was
    supposed to install never existed while the bookkeeping said it did. A test
    that only exercises `wrap_callable` on a fake object cannot see that.
    """

    async def test_the_gate_is_installed_and_removed_for_real(self, wired):
        from nekro_agent.services.message_service import message_service as ms
        from nekro_auto_sleep.runtime import is_wrapped

        plugin_mod, _store, _ctx = wired
        plugin_mod._installed_wraps.clear()
        try:
            assert plugin_mod._install_wraps() is True
            assert is_wrapped(ms, "schedule_agent_task"), "gate reported but not installed"
            assert (ms, "schedule_agent_task") in plugin_mod._installed_wraps
        finally:
            plugin_mod._uninstall_wraps()

        assert not is_wrapped(ms, "schedule_agent_task")
        assert plugin_mod._installed_wraps == []

    async def test_the_installed_gate_honours_the_policy(self, wired):
        from nekro_agent.services.message_service import message_service as ms

        plugin_mod, store, _ctx = wired
        await _put_asleep(store)
        plugin_mod._installed_wraps.clear()
        try:
            plugin_mod._install_wraps()

            m.config.NIGHT_TIMER_POLICY = "run"
            assert await ms.schedule_agent_task(CHAT_KEY) == "scheduled"

            m.config.NIGHT_TIMER_POLICY = "block"
            assert await ms.schedule_agent_task(CHAT_KEY) is None
        finally:
            plugin_mod._uninstall_wraps()


# ---------------------------------------------------------------------------
# Schedule overrides and operator commands
# ---------------------------------------------------------------------------


def _cmd_context(chat_key: str = CHAT_KEY):
    from tests.hoststub import CommandExecutionContext

    return CommandExecutionContext(chat_key=chat_key)


class TestScheduleLayering:
    async def test_global_config_is_the_default(self, wired):
        plugin_mod, _store, _ctx = wired
        schedule = await plugin_mod._resolve_schedule_for(CHAT_KEY)

        assert schedule.timezone == "Asia/Shanghai"
        assert schedule.sleep_time == "23:00"
        assert set(schedule.sources.values()) == {"global"}

    async def test_channel_override_wins(self, wired):
        plugin_mod, _store, _ctx = wired
        await plugin_mod._get_overrides().set_channel(
            CHAT_KEY, ScheduleOverride(timezone="Asia/Tokyo", sleep_time="01:00")
        )
        plugin_mod.invalidate_settings_cache(CHAT_KEY)

        schedule = await plugin_mod._resolve_schedule_for(CHAT_KEY)
        assert schedule.timezone == "Asia/Tokyo"
        assert schedule.sleep_time == "01:00"
        # Untouched fields still come from the global config.
        assert schedule.wake_time_start == "06:45"
        assert schedule.sources["timezone"] == "channel"
        assert schedule.sources["wake_time_start"] == "global"

    async def test_channel_beats_persona_beats_global(self, wired):
        plugin_mod, _store, _ctx = wired
        overrides = plugin_mod._get_overrides()
        await overrides.set_preset(1, ScheduleOverride(sleep_time="22:00", timezone="Asia/Tokyo"))
        await overrides.set_channel(CHAT_KEY, ScheduleOverride(sleep_time="00:30"))
        plugin_mod.invalidate_settings_cache()

        schedule = await plugin_mod._resolve_schedule_for(CHAT_KEY)
        assert schedule.sleep_time == "00:30"  # channel
        assert schedule.timezone == "Asia/Tokyo"  # persona
        assert schedule.wake_time_end == "08:30"  # global
        assert schedule.sources == {
            "timezone": "preset",
            "sleep_time": "channel",
            "wake_time_start": "global",
            "wake_time_end": "global",
        }

    async def test_bedtime_follows_the_channel_override(self, wired, clock):
        plugin_mod, store, _ctx = wired
        await store.hydrate(CHAT_KEY)
        await plugin_mod._get_overrides().set_channel(
            CHAT_KEY, ScheduleOverride(sleep_time="01:00")
        )
        plugin_mod.invalidate_settings_cache(CHAT_KEY)

        # 23:30 local: past the global bedtime, before the one this channel set.
        await plugin_mod._check_sleep_transition(
            store, CHAT_KEY, BEDTIME + timedelta(minutes=30)
        )
        assert store.get_cached(CHAT_KEY).status == SleepStatus.AWAKE

        # 01:30 local.
        await plugin_mod._check_sleep_transition(
            store, CHAT_KEY, BEDTIME + timedelta(hours=2, minutes=30)
        )
        assert store.get_cached(CHAT_KEY).status == SleepStatus.ASLEEP

    async def test_the_cycle_records_the_schedule_it_actually_slept_on(self, wired):
        """The snapshot drives the cycle boundaries, not just the decision to sleep.

        Deciding with the channel schedule but building the cycle from the global
        one would put bedtime and the wake range in the wrong timezone for the
        whole night.
        """
        plugin_mod, store, _ctx = wired
        await store.hydrate(CHAT_KEY)
        await plugin_mod._get_overrides().set_channel(
            CHAT_KEY, ScheduleOverride(timezone="Asia/Tokyo", sleep_time="01:00")
        )
        plugin_mod.invalidate_settings_cache(CHAT_KEY)

        await plugin_mod._check_sleep_transition(
            store, CHAT_KEY, BEDTIME + timedelta(hours=2, minutes=30)
        )

        cycle = store.get_cached(CHAT_KEY).cycle
        assert cycle.config_snapshot.timezone == "Asia/Tokyo"
        assert cycle.config_snapshot.sleep_time == "01:00"
        assert cycle.sleep_at.astimezone(ZoneInfo("Asia/Tokyo")).strftime("%H:%M") == "01:00"


class TestOperatorCommands:
    async def test_status_reports_the_schedule_and_its_origin(self, wired):
        plugin_mod, _store, _ctx = wired
        await plugin_mod._get_overrides().set_channel(
            CHAT_KEY, ScheduleOverride(timezone="Asia/Tokyo")
        )
        plugin_mod.invalidate_settings_cache(CHAT_KEY)

        response = await plugin_mod.plugin.commands["sleep.status"](_cmd_context())

        assert response.status == "success"
        assert "Asia/Tokyo" in response.message
        assert "本频道" in response.message
        assert PERSONA in response.message

    async def test_status_shows_last_nights_scoring_terms(self, wired):
        plugin_mod, store, _ctx = wired
        state = await _put_asleep(store)
        await plugin_mod._settle_wake(store, CHAT_KEY, state.cycle.planned_wake_at)

        response = await plugin_mod.plugin.commands["sleep.status"](_cmd_context())

        assert "上一夜" in response.message
        assert "整夜无扰" in response.message

    async def test_now_puts_it_to_bed_immediately(self, wired, clock):
        plugin_mod, store, _ctx = wired
        clock.now = BEDTIME - timedelta(hours=1)  # 22:00, an hour early
        await store.hydrate(CHAT_KEY)

        response = await plugin_mod.plugin.commands["sleep.now"](_cmd_context())

        assert response.status == "success"
        state = store.get_cached(CHAT_KEY)
        assert state.status == SleepStatus.ASLEEP
        # The night is measured from when it actually turned in, not from 23:00.
        assert state.cycle.sleep_at == clock.now

    async def test_wake_settles_and_reports(self, wired):
        plugin_mod, store, ctx = wired
        await _put_asleep(store)
        ctx.sent.clear()

        response = await plugin_mod.plugin.commands["sleep.wake"](_cmd_context())

        assert response.status == "success"
        assert store.get_cached(CHAT_KEY).status == SleepStatus.AWAKE
        assert any("已起床" in text for text, _ in ctx.sent)

    async def test_wake_refuses_when_already_awake(self, wired):
        plugin_mod, store, _ctx = wired
        await store.hydrate(CHAT_KEY)
        response = await plugin_mod.plugin.commands["sleep.wake"](_cmd_context())
        assert response.status == "error"

    async def test_skip_keeps_it_up_for_one_night(self, wired, clock):
        plugin_mod, store, _ctx = wired
        await store.hydrate(CHAT_KEY)

        await plugin_mod.plugin.commands["sleep.skip"](_cmd_context())
        await plugin_mod._check_sleep_transition(
            store, CHAT_KEY, BEDTIME + timedelta(minutes=30)
        )

        state = store.get_cached(CHAT_KEY)
        assert state.status == SleepStatus.AWAKE
        assert state.skip_sleep_date == "2026-08-17"

    async def test_a_skipped_night_does_not_rewrite_state_every_tick(self, wired):
        """The maintenance loop runs every 15 seconds all night long."""
        plugin_mod, store, _ctx = wired
        await store.hydrate(CHAT_KEY)
        await plugin_mod.plugin.commands["sleep.skip"](_cmd_context())

        writes_before = store.test_backend.set_calls
        for minutes in (30, 60, 90):
            await plugin_mod._check_sleep_transition(
                store, CHAT_KEY, BEDTIME + timedelta(minutes=minutes)
            )

        assert store.test_backend.set_calls == writes_before

    async def test_set_and_unset_a_channel_override(self, wired):
        plugin_mod, _store, _ctx = wired
        commands = plugin_mod.plugin.commands

        ok = await commands["sleep.set"](_cmd_context(), "tz", "Asia/Tokyo")
        assert ok.status == "success"
        assert (await plugin_mod._resolve_schedule_for(CHAT_KEY)).timezone == "Asia/Tokyo"

        await commands["sleep.unset"](_cmd_context())
        assert (await plugin_mod._resolve_schedule_for(CHAT_KEY)).timezone == "Asia/Shanghai"

    async def test_set_rejects_nonsense(self, wired):
        plugin_mod, _store, _ctx = wired
        commands = plugin_mod.plugin.commands

        assert (await commands["sleep.set"](_cmd_context(), "tz", "Mars/Olympus")).status == "error"
        assert (await commands["sleep.set"](_cmd_context(), "bed", "25 点")).status == "error"
        assert (await commands["sleep.set"](_cmd_context(), "nope", "x")).status == "error"
        # Nothing was written.
        assert (await plugin_mod._resolve_schedule_for(CHAT_KEY)).timezone == "Asia/Shanghai"

    async def test_set_can_target_the_persona(self, wired):
        plugin_mod, _store, _ctx = wired

        ok = await plugin_mod.plugin.commands["sleep.set"](
            _cmd_context(), "bed", "22:00", "preset"
        )
        assert ok.status == "success"

        schedule = await plugin_mod._resolve_schedule_for(CHAT_KEY)
        assert schedule.sleep_time == "22:00"
        assert schedule.sources["sleep_time"] == "preset"


class TestUpgradeWarnings:
    """Plugin config is persisted per field, so an upgrade keeps v1 values.

    Found on a real install: after deploying v2 the score still read 60% for
    every rough night, because the saved config still carried the v1 floor.
    """

    async def test_a_stale_v1_floor_is_called_out(self, wired):
        plugin_mod, _store, _ctx = wired
        m.config.QUALITY_MIN = 60

        warnings = plugin_mod.collect_upgrade_warnings()
        assert len(warnings) == 1
        assert "20" in warnings[0]

    async def test_a_sane_floor_says_nothing(self, wired):
        plugin_mod, _store, _ctx = wired
        m.config.QUALITY_MIN = 20
        assert plugin_mod.collect_upgrade_warnings() == []

    async def test_status_surfaces_the_warning(self, wired):
        plugin_mod, _store, _ctx = wired
        m.config.QUALITY_MIN = 60

        response = await plugin_mod.plugin.commands["sleep.status"](_cmd_context())
        assert "睡眠质量下限" in response.message

    async def test_status_marks_a_clipped_score(self, wired):
        plugin_mod, store, _ctx = wired
        m.config.QUALITY_MIN = 60
        state = await _put_asleep(store)
        # A night that barely happened: the raw score lands far below the floor.
        await plugin_mod._settle_wake(
            store, CHAT_KEY, state.cycle.sleep_at + timedelta(minutes=1)
        )

        response = await plugin_mod.plugin.commands["sleep.status"](_cmd_context())
        assert "被下限" in response.message


class TestToolExposure:
    async def test_resume_sleep_is_hidden_while_awake(self, wired):
        plugin_mod, store, ctx = wired
        await store.save(ChatSleepState(chat_key=CHAT_KEY))
        assert await plugin_mod.plugin.collect_methods(ctx) == []

    async def test_resume_sleep_is_hidden_while_asleep(self, wired):
        plugin_mod, store, ctx = wired
        await _put_asleep(store)
        assert await plugin_mod.plugin.collect_methods(ctx) == []

    async def test_resume_sleep_appears_once_woken(self, wired):
        plugin_mod, store, ctx = wired
        await _put_asleep(store)
        await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content="醒醒", chat_key=CHAT_KEY)
        )
        await plugin_mod.plugin.on_user_message(
            ctx, ChatMessage(content="要", chat_key=CHAT_KEY)
        )

        methods = await plugin_mod.plugin.collect_methods(ctx)
        assert len(methods) == 1
        assert methods[0] is plugin_mod.resume_sleep_tool


# ---------------------------------------------------------------------------
# Two instances sharing one group
# ---------------------------------------------------------------------------

INST_A = "onebot_v11-inst1-group_555000"
INST_B = "onebot_v11-inst2-group_555000"


class TestSharedGroupIsolation:
    """Same群, two accounts. In the fork the instance segment is part of the
    chat_key, so everything downstream is keyed apart — but only if nothing
    quietly falls back to the global config."""

    def test_the_wake_draw_is_independent_per_instance(self):
        from datetime import date

        from nekro_auto_sleep.schedule import pick_wake_time

        start = datetime(2026, 8, 17, 22, 45, tzinfo=UTC)
        end = datetime(2026, 8, 18, 0, 30, tzinfo=UTC)

        differing = 0
        for offset in range(14):
            day = date(2026, 8, 1) + timedelta(days=offset)
            a = pick_wake_time(INST_A, day, start, end, 1)
            b = pick_wake_time(INST_B, day, start, end, 1)
            if a != b:
                differing += 1
            # and each is stable for its own instance
            assert a == pick_wake_time(INST_A, day, start, end, 1)

        # Two independent draws over a 106-minute window; identical every night
        # would mean the instance segment stopped reaching the seed.
        assert differing >= 12, f"only {differing}/14 nights differed"

    async def test_each_instance_keeps_its_own_night(self, wired):
        plugin_mod, store, _ctx = wired
        snap = plugin_mod._make_config_snapshot()

        state_a = transition_to_sleep(ChatSleepState(chat_key=INST_A), BEDTIME, snap)
        await store.save(state_a)
        await store.hydrate(INST_B)

        assert store.get_cached(INST_A).status == SleepStatus.ASLEEP
        assert store.get_cached(INST_B).status == SleepStatus.AWAKE

    async def test_instance_config_overrides_reach_the_background_loop(self, wired):
        """The maintenance loop carries no inbound context.

        `ScopedPluginConfig` resolves per-instance config from a contextvar the
        adapter sets on the way in — a background task has none, so without an
        explicit lookup both accounts would sleep on the global schedule no
        matter what the operator configured.
        """
        from tests.hoststub import set_instance_config_override

        plugin_mod, _store, _ctx = wired
        set_instance_config_override(
            plugin_mod.plugin.key, "inst2", {"SLEEP_TIME": "01:30"}
        )
        plugin_mod.invalidate_settings_cache()

        a = await plugin_mod._settings_for(INST_A)
        b = await plugin_mod._settings_for(INST_B)

        assert a.instance_key == "inst1"
        assert b.instance_key == "inst2"
        assert a.schedule.sleep_time == "23:00"
        assert b.schedule.sleep_time == "01:30"
        assert b.schedule.sources["sleep_time"] == "instance"
        # Everything it did not override still comes from the global config.
        assert b.schedule.wake_time_start == "06:45"

    async def test_the_two_instances_go_to_bed_at_their_own_times(self, wired):
        from tests.hoststub import set_instance_config_override

        plugin_mod, store, _ctx = wired
        set_instance_config_override(
            plugin_mod.plugin.key, "inst2", {"SLEEP_TIME": "01:30"}
        )
        plugin_mod.invalidate_settings_cache()
        await store.hydrate(INST_A)
        await store.hydrate(INST_B)

        # 23:30 local: past inst1's bedtime, well before inst2's.
        tick = BEDTIME + timedelta(minutes=30)
        await plugin_mod._check_sleep_transition(store, INST_A, tick)
        await plugin_mod._check_sleep_transition(store, INST_B, tick)

        assert store.get_cached(INST_A).status == SleepStatus.ASLEEP
        assert store.get_cached(INST_B).status == SleepStatus.AWAKE

        # 02:00 local.
        later = BEDTIME + timedelta(hours=3)
        await plugin_mod._check_sleep_transition(store, INST_B, later)
        assert store.get_cached(INST_B).status == SleepStatus.ASLEEP

    async def test_the_cycle_records_the_instance_schedule(self, wired):
        from tests.hoststub import set_instance_config_override

        plugin_mod, store, _ctx = wired
        set_instance_config_override(
            plugin_mod.plugin.key, "inst2", {"SLEEP_TIME": "01:30", "QUALITY_MIN": 35}
        )
        plugin_mod.invalidate_settings_cache()
        await store.hydrate(INST_B)

        await plugin_mod._check_sleep_transition(
            store, INST_B, BEDTIME + timedelta(hours=3)
        )

        cycle = store.get_cached(INST_B).cycle
        assert cycle.config_snapshot.sleep_time == "01:30"
        # Not just the schedule: the whole snapshot has to be the instance's.
        assert cycle.config_snapshot.quality_min == 35
