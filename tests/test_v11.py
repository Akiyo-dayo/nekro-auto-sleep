"""Tests for v1.1.0 additions: timer intervals, burdern isolation, fun layer."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
import importlib.util
import pathlib

from zoneinfo import ZoneInfo

from nekro_auto_sleep.engine import (
    build_wake_inject_text,
    close_timer_interval,
    open_timer_interval,
    settle_natural_wake,
    should_send_wake_notice,
)
from nekro_auto_sleep.models import (
    ChatSleepState,
    SleepCycle,
    SleepSegment,
    SleepStatus,
    WakeAttempt,
)
from nekro_auto_sleep.quality import (
    compute_timer_burden,
    compute_user_burden,
    compute_streak_note,
    pick_dream,
    quality_tier,
    stable_pick,
    BAD_DREAMS,
)
from nekro_auto_sleep.schedule import create_config_snapshot

UTC = ZoneInfo("UTC")
CHAT_KEY = "onebot_v11-group_123456789"


def _make_snapshot():
    return create_config_snapshot(
        timezone="Asia/Shanghai",
        sleep_time="23:00",
        wake_time_start="06:45",
        wake_time_end="08:30",
        wake_random_step_minutes=1,
        near_wake_ratio=0.15,
        wake_confirm_window_seconds=180,
        history_mode="preserve",
        call_keywords="醒醒,起床,在吗",
        fallback_persona_name="Bot",
        early_wake_idle_minutes=10,
        quality_min=60,
        quality_max=120,
        quality_jitter_points=4.0,
    )


def _make_state(**cycle_overrides) -> ChatSleepState:
    snap = _make_snapshot()
    sleep_at = datetime(2026, 9, 1, 15, 0, tzinfo=UTC)
    wake_at = datetime(2026, 9, 2, 0, 30, tzinfo=UTC)
    cycle = SleepCycle(
        cycle_id="test-cycle",
        sleep_date="2026-09-01",
        timezone="Asia/Shanghai",
        sleep_at=sleep_at,
        planned_wake_at=wake_at,
        config_snapshot=snap,
        quality_seed="abcdef1234567890",
        sleep_segments=[SleepSegment(open_at=sleep_at)],
        **cycle_overrides,
    )
    return ChatSleepState(chat_key=CHAT_KEY, status=SleepStatus.ASLEEP, cycle=cycle)


class TestTimerIntervals:
    def test_open_ignores_when_awake(self):
        state = ChatSleepState(chat_key=CHAT_KEY, status=SleepStatus.AWAKE)
        now = datetime.now(UTC)
        out = open_timer_interval(state, "t1", now)
        assert out.cycle is None
        assert out.status == SleepStatus.AWAKE

    def test_open_closes_segment_and_records_interval(self):
        state = _make_state()
        now = datetime(2026, 9, 1, 16, 0, tzinfo=UTC)
        out = open_timer_interval(state, "t1", now)
        assert out.cycle is not None
        assert len(out.cycle.timer_intervals) == 1
        assert out.cycle.timer_intervals[0].end_at is None
        assert out.cycle.sleep_segments[-1].close_at == now

    def test_close_reopens_segment_while_asleep(self):
        state = _make_state()
        start = datetime(2026, 9, 1, 16, 0, tzinfo=UTC)
        end = datetime(2026, 9, 1, 16, 10, tzinfo=UTC)
        out = open_timer_interval(state, "t1", start)
        out = close_timer_interval(out, "t1", end)
        assert out.cycle is not None
        assert out.cycle.timer_intervals[0].end_at == end
        # a fresh open segment exists so later sleep still counts
        assert out.cycle.sleep_segments[-1].close_at is None
        assert out.cycle.sleep_segments[-1].open_at == end

    def test_close_keeps_segment_when_awake_early(self):
        state = _make_state()
        state = state.model_copy(update={"status": SleepStatus.AWAKE_EARLY})
        # simulate that the segment was already closed when the bot woke early
        closed = state.cycle.sleep_segments[0].model_copy(
            update={"close_at": datetime(2026, 9, 1, 15, 30, tzinfo=UTC)}
        )
        state.cycle = state.cycle.model_copy(update={"sleep_segments": [closed]})
        start = datetime(2026, 9, 1, 16, 0, tzinfo=UTC)
        end = datetime(2026, 9, 1, 16, 10, tzinfo=UTC)
        out = open_timer_interval(state, "t1", start)  # ignored: not ASLEEP
        out = close_timer_interval(out, "t1", end)
        assert out.cycle is not None
        assert not out.cycle.timer_intervals
        # no new segment opened while awake-early
        assert all(seg.close_at is not None for seg in out.cycle.sleep_segments)

    def test_double_close_does_not_leak_segments(self):
        state = _make_state()
        start = datetime(2026, 9, 1, 16, 0, tzinfo=UTC)
        end = datetime(2026, 9, 1, 16, 10, tzinfo=UTC)
        out = open_timer_interval(state, "t1", start)
        out = close_timer_interval(out, "t1", end)
        n = len(out.cycle.sleep_segments)
        out = close_timer_interval(out, "t1", end)  # second call: nothing to close
        assert len(out.cycle.sleep_segments) == n


class TestBurdenNoDoubleCount:
    def test_timer_gap_not_charged_to_user(self):
        state = _make_state()
        now = datetime(2026, 9, 1, 16, 0, tzinfo=UTC)
        end = datetime(2026, 9, 1, 17, 0, tzinfo=UTC)
        with_timer = open_timer_interval(state, "t1", now)
        with_timer = close_timer_interval(with_timer, "t1", end)
        assert with_timer.cycle is not None

        # same shape but only the wake gap, no timer bookkeeping
        manual = _make_state()
        seg1 = SleepSegment(open_at=manual.cycle.sleep_at, close_at=now)
        seg2 = SleepSegment(open_at=end, close_at=manual.cycle.planned_wake_at)
        manual.cycle = manual.cycle.model_copy(
            update={"sleep_segments": [seg1, seg2]}
        )

        target = (manual.cycle.planned_wake_at - manual.cycle.sleep_at).total_seconds()
        burden_timer = compute_user_burden(with_timer.cycle, target)
        burden_manual = compute_user_burden(manual.cycle, target)
        assert burden_timer < burden_manual * 0.5

    def test_timer_burden_counts_for_timer_only(self):
        state = _make_state()
        start = datetime(2026, 9, 1, 16, 0, tzinfo=UTC)
        end = datetime(2026, 9, 1, 17, 0, tzinfo=UTC)
        out = open_timer_interval(state, "t1", start)
        out = close_timer_interval(out, "t1", end)
        target = (out.cycle.planned_wake_at - out.cycle.sleep_at).total_seconds()
        assert compute_timer_burden(out.cycle, target) > 0


class TestWakeInjectText:
    def test_deep_night_gets_grumpy(self):
        state = _make_state()
        # 30 minutes after sleep_at -> deep night
        now = state.cycle.sleep_at + timedelta(minutes=30)
        text = build_wake_inject_text(state, now)
        assert "起床气" in text

    def test_near_wake_is_gentle(self):
        state = _make_state()
        now = state.cycle.planned_wake_at - timedelta(minutes=20)
        text = build_wake_inject_text(state, now)
        assert "临近自然醒" in text

    def test_mid_sleep_is_sleepy(self):
        state = _make_state()
        now = state.cycle.sleep_at + timedelta(hours=4)
        text = build_wake_inject_text(state, now)
        assert "困" in text


class TestWakeNoticeCondition:
    def test_default_requires_attempts(self):
        state = _make_state()
        state = settle_natural_wake(state, state.cycle.planned_wake_at, "Bot", 100)[0]
        assert not should_send_wake_notice(state.cycle)

    def test_always_sends_without_attempts(self):
        state = _make_state()
        state = settle_natural_wake(state, state.cycle.planned_wake_at, "Bot", 100)[0]
        assert should_send_wake_notice(state.cycle, always=True)

    def test_early_awake_settlement_never_notifies(self):
        state = _make_state()
        state.cycle = state.cycle.model_copy(
            update={
                "wake_attempts": [
                    WakeAttempt(
                        user_id="u1", chat_key=CHAT_KEY,
                        attempted_at=datetime(2026, 9, 1, 16, 0, tzinfo=UTC),
                    )
                ]
            }
        )
        state.cycle = state.cycle.model_copy(update={"ended_while_early_awake": True})
        assert not should_send_wake_notice(state.cycle, always=True)


class TestFunLayer:
    def test_tier_ordering(self):
        assert quality_tier(120)[0] == "神清气爽"
        assert quality_tier(100)[0] == "睡得不错"
        assert quality_tier(85)[0] == "睡得一般"
        assert quality_tier(72)[0] == "睡得迷糊"
        assert quality_tier(60)[0] == "睡眼惺忪"

    def test_dream_deterministic(self):
        d1 = pick_dream("chat:2026-09-01", 100)
        d2 = pick_dream("chat:2026-09-01", 100)
        assert d1 == d2
        assert d1 is not None

    def test_perfect_sleep_no_dream(self):
        assert pick_dream("chat:2026-09-01", 118) is None

    def test_bad_sleep_nightmare_pool(self):
        assert pick_dream("chat:2026-09-01", 62) in BAD_DREAMS

    def test_stable_pick_stable(self):
        assert stable_pick("x", ("a", "b", "c")) == stable_pick("x", ("a", "b", "c"))

    def test_streak_note_first_day(self):
        note = compute_streak_note({"2026-09-01": 97}, "2026-09-01")
        assert note is not None and "第一天" in note

    def test_streak_note_counts_consecutive_good_nights(self):
        history = {
            "2026-08-29": 96,
            "2026-08-30": 98,
            "2026-08-31": 97,
            "2026-09-01": 99,
        }
        note = compute_streak_note(history, "2026-09-01")
        assert note is not None and "连续 4 天" in note

    def test_streak_note_trend_down(self):
        history = {
            "2026-08-31": 99,
            "2026-09-01": 78,
        }
        note = compute_streak_note(history, "2026-09-01")
        assert note is not None and "掉了" in note


class TestHistoryPersistence:
    def test_state_defaults_empty_history(self):
        state = ChatSleepState(chat_key=CHAT_KEY)
        assert state.quality_history == {}

    def test_history_roundtrip(self):
        state = _make_state()
        state = state.model_copy(
            update={"quality_history": {"2026-09-01": 96, "2026-08-31": 88}}
        )
        raw = state.model_dump_json()
        from nekro_auto_sleep.models import ChatSleepState as CS

        restored = CS.model_validate_json(raw)
        assert restored.quality_history == {"2026-09-01": 96, "2026-08-31": 88}


class TestRuntimeFixes:
    def test_extract_timer_task_info_with_datetime(self):
        from nekro_auto_sleep.runtime import _extract_timer_task_info

        class FakeJob:
            chat_key = "test_chat"
            job_id = "job_123"

        job = FakeJob()
        fired_at = datetime.now()
        # Shape of _fire_job(job, fired_at, is_misfire)
        ck, tid = _extract_timer_task_info((job, fired_at, False), {})
        assert ck == "test_chat"
        assert tid == "job_123"

    def test_lease_ledger_ghost_cleanup(self):
        from nekro_auto_sleep.runtime import LeaseLedger
        from nekro_auto_sleep.models import SourceType

        ledger = LeaseLedger()
        ledger.create("l1", SourceType.TIMER_ONESHOT, "chat1", "task1", ttl=100.0)
        # Directly simulate _leases missing or expired
        del ledger._leases["l1"]
        # get_active_for_chat should clean up ghost from _by_chat_key
        active = ledger.get_active_for_chat("chat1")
        assert len(active) == 0
        assert "chat1" not in ledger._by_chat_key


def _setup_mock_nekro_agent() -> None:
    if "nekro_agent" in sys.modules:
        return

    nekro_agent = ModuleType("nekro_agent")
    sys.modules["nekro_agent"] = nekro_agent

    api = ModuleType("nekro_agent.api")
    sys.modules["nekro_agent.api"] = api
    nekro_agent.api = api

    i18n = MagicMock()
    i18n.t = lambda key, **kwargs: key
    i18n.i18n_text = lambda **kwargs: kwargs
    api.i18n = i18n

    plugin = ModuleType("nekro_agent.api.plugin")
    class ConfigBase(BaseModel):
        model_config = {"extra": "allow"}

    class ExtraField(BaseModel):
        model_config = {"extra": "allow"}

    class SandboxMethodType:
        TOOL = "tool"

    class NekroPlugin:
        def __init__(self, *args, **kwargs):
            self.module_name = kwargs.get("module_name", "nekro_auto_sleep")
            self.key = kwargs.get("key", "Akiyo_dayo.nekro_auto_sleep")
        def mount(self, *a, **kw): pass
        def mount_config(self, *a, **kw): return lambda cls: cls
        def get_config(self, config_cls): return config_cls()
        def hook_user_message(self, *a, **kw): return lambda f: f
        def hook_system_message(self, *a, **kw): return lambda f: f
        def hook_agent_prompt(self, *a, **kw): return lambda f: f
        def hook_agent_tool(self, *a, **kw): return lambda f: f
        def register_sandbox_method(self, *a, **kw): return lambda f: f
        def register_background_task(self, *a, **kw): return lambda f: f
        def mount_on_user_message(self, *a, **kw): return lambda f: f
        def mount_on_system_message(self, *a, **kw): return lambda f: f
        def mount_prompt_inject_method(self, *a, **kw): return lambda f: f
        def mount_sandbox_method(self, *a, **kw): return lambda f: f
        def mount_init_method(self, *a, **kw): return lambda f: f
        def on_enabled(self, *a, **kw): return lambda f: f
        def on_disabled(self, *a, **kw): return lambda f: f
        def mount_cleanup_method(self, *a, **kw): return lambda f: f
    plugin.ConfigBase = ConfigBase
    plugin.ExtraField = ExtraField
    plugin.SandboxMethodType = SandboxMethodType
    plugin.NekroPlugin = NekroPlugin
    sys.modules["nekro_agent.api.plugin"] = plugin
    api.plugin = plugin

    schemas = ModuleType("nekro_agent.api.schemas")
    class AgentCtx:
        @classmethod
        async def create_by_chat_key(cls, ck):
            pass
    schemas.AgentCtx = AgentCtx
    sys.modules["nekro_agent.api.schemas"] = schemas
    api.schemas = schemas

    signal = ModuleType("nekro_agent.api.signal")
    class MsgSignal:
        pass
    signal.MsgSignal = MsgSignal
    sys.modules["nekro_agent.api.signal"] = signal
    api.signal = signal

    chat_msg_mod = ModuleType("nekro_agent.schemas.chat_message")
    class ChatMessage:
        pass
    chat_msg_mod.ChatMessage = ChatMessage
    sys.modules["nekro_agent.schemas"] = ModuleType("nekro_agent.schemas")
    sys.modules["nekro_agent.schemas.chat_message"] = chat_msg_mod

    models_mod = ModuleType("nekro_agent.models")
    db_mod = ModuleType("nekro_agent.models.db_plugin_data")
    class DBPluginData:
        pass
    db_mod.DBPluginData = DBPluginData
    sys.modules["nekro_agent.models"] = models_mod
    sys.modules["nekro_agent.models.db_plugin_data"] = db_mod


class TestPersistenceAndToolEnhancements:
    import pytest

    @pytest.mark.asyncio
    async def test_with_state_skips_save_when_unchanged(self):
        from nekro_auto_sleep.persistence import SleepStateStore

        saved_count = 0

        class MockBackend:
            def __init__(self):
                self.data = {}

            async def get(self, chat_key=None, store_key=""):
                return self.data.get(store_key)

            async def set(self, chat_key=None, store_key="", value=""):
                nonlocal saved_count
                saved_count += 1
                self.data[store_key] = value

        backend = MockBackend()
        store = SleepStateStore(backend)

        from nekro_auto_sleep.models import SleepStatus

        # First run: state is modified, so save SHOULD be called
        async def _modify(s):
            return s.model_copy(update={"status": SleepStatus.ASLEEP})

        await store.with_state("chat_test", _modify)
        assert saved_count >= 1  # save was called

        count_before = saved_count
        # Second run: state is unchanged, save should NOT be called
        async def _noop(s):
            return s

        await store.with_state("chat_test", _noop)
        assert saved_count == count_before  # no new save!

    def test_clean_expired_offers(self):
        from nekro_auto_sleep.models import ChatSleepState, PendingWakeOffer
        from nekro_auto_sleep.engine import clean_expired_offers
        from datetime import timezone

        t0 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        state = ChatSleepState(
            chat_key="c1",
            pending_wake_offers={
                "u1": PendingWakeOffer(
                    user_id="u1",
                    offered_at=t0 - timedelta(seconds=20),
                    expires_at=t0 + timedelta(seconds=10),
                ),
                "u2": PendingWakeOffer(
                    user_id="u2",
                    offered_at=t0 - timedelta(seconds=20),
                    expires_at=t0 - timedelta(seconds=10),
                ),
            },
        )
        cleaned = clean_expired_offers(state, t0)
        assert "u1" in cleaned.pending_wake_offers
        assert "u2" not in cleaned.pending_wake_offers

    @pytest.mark.asyncio
    async def test_dynamic_bedtime_llm_dispatch(self, monkeypatch):
        _setup_mock_nekro_agent()
        pkg_root = pathlib.Path(__file__).resolve().parent.parent
        init_file = pkg_root / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "nekro_auto_sleep",
            init_file,
            submodule_search_locations=[str(pkg_root)],
        )
        nas_mod = sys.modules["nekro_auto_sleep"]
        spec.loader.exec_module(nas_mod)

        pushed_prompts = []

        class MockCtx:
            async def push_system(self, prompt, trigger_agent=False):
                pushed_prompts.append((prompt, trigger_agent))

        mock_ctx = MockCtx()
        monkeypatch.setattr(
            nas_mod.AgentCtx,
            "create_by_chat_key",
            AsyncMock(return_value=mock_ctx),
        )
        monkeypatch.setattr(nas_mod, "_deterministic_hit", lambda *_: True)
        monkeypatch.setattr(nas_mod.config, "LLM_GREETINGS_ENABLED", True)

        await nas_mod._maybe_send_bedtime("chat1", "2025-01-01")
        assert len(pushed_prompts) == 1
        prompt, trigger = pushed_prompts[0]
        assert trigger is True
        assert "就寝时间" in prompt
        assert "符合你人设口吻" in prompt

    @pytest.mark.asyncio
    async def test_dynamic_bedtime_fallback_on_error(self, monkeypatch):
        _setup_mock_nekro_agent()
        nas_mod = sys.modules["nekro_auto_sleep"]

        fallback_sent = []

        class FailingCtx:
            async def push_system(self, prompt, trigger_agent=False):
                raise RuntimeError("LLM Service Error")

        monkeypatch.setattr(
            nas_mod.AgentCtx,
            "create_by_chat_key",
            AsyncMock(return_value=FailingCtx()),
        )
        monkeypatch.setattr(nas_mod, "_deterministic_hit", lambda *_: True)
        monkeypatch.setattr(nas_mod.config, "LLM_GREETINGS_ENABLED", True)
        monkeypatch.setattr(
            nas_mod,
            "_send_quiet_text",
            AsyncMock(side_effect=lambda ck, text: fallback_sent.append((ck, text))),
        )

        await nas_mod._maybe_send_bedtime("chat1", "2025-01-01")
        assert len(fallback_sent) == 1
        assert fallback_sent[0][0] == "chat1"

    @pytest.mark.asyncio
    async def test_dynamic_wake_llm_dispatch(self, monkeypatch):
        _setup_mock_nekro_agent()
        nas_mod = sys.modules["nekro_auto_sleep"]
        from nekro_auto_sleep.persistence import SleepStateStore

        class MockBackend:
            def __init__(self):
                self.data = {}
            async def get(self, chat_key=None, store_key=""):
                return self.data.get(store_key)
            async def set(self, chat_key=None, store_key="", value=""):
                self.data[store_key] = value

        store = SleepStateStore(MockBackend())
        now = datetime(2026, 9, 2, 0, 30, tzinfo=UTC)
        init_state = _make_state().model_copy(
            update={"chat_key": "chat_wake"}
        )
        await store.save(init_state)

        pushed_prompts = []

        class MockCtx:
            chat_key = "chat_wake"
            async def push_system(self, prompt, trigger_agent=False):
                pushed_prompts.append((prompt, trigger_agent))
            async def send_text(self, text, record=False):
                pass

        monkeypatch.setattr(
            nas_mod.AgentCtx,
            "create_by_chat_key",
            AsyncMock(return_value=MockCtx()),
        )
        monkeypatch.setattr(nas_mod.config, "LLM_GREETINGS_ENABLED", True)
        monkeypatch.setattr(nas_mod.config, "WAKE_NOTICE_ALWAYS", True)
        monkeypatch.setattr(nas_mod.config, "WAKE_NOTICE_GRACE_MINUTES", 120)

        await nas_mod._settle_wake(store, "chat_wake", now)
        assert len(pushed_prompts) == 1
        prompt, trigger = pushed_prompts[0]
        assert trigger is True
        assert "自然睡醒" in prompt
        assert "睡眠质量" in prompt

    @pytest.mark.asyncio
    async def test_settle_wake_grace_suppression(self, monkeypatch):
        _setup_mock_nekro_agent()
        nas_mod = sys.modules["nekro_auto_sleep"]
        from nekro_auto_sleep.persistence import SleepStateStore

        class MockBackend:
            def __init__(self):
                self.data = {}
            async def get(self, chat_key=None, store_key=""):
                return self.data.get(store_key)
            async def set(self, chat_key=None, store_key="", value=""):
                self.data[store_key] = value

        store = SleepStateStore(MockBackend())
        planned_wake = datetime(2026, 9, 2, 0, 30, tzinfo=UTC)
        # Now is 3 hours after planned wake (180 min delay > 120 min grace)
        now = planned_wake + timedelta(hours=3)
        init_state = _make_state().model_copy(
            update={"chat_key": "chat_wake_delayed"}
        )
        await store.save(init_state)

        pushed_prompts = []

        class MockCtx:
            chat_key = "chat_wake_delayed"
            async def push_system(self, prompt, trigger_agent=False):
                pushed_prompts.append((prompt, trigger_agent))
            async def send_text(self, text, record=False):
                pass

        monkeypatch.setattr(
            nas_mod.AgentCtx,
            "create_by_chat_key",
            AsyncMock(return_value=MockCtx()),
        )
        monkeypatch.setattr(nas_mod.config, "LLM_GREETINGS_ENABLED", True)
        monkeypatch.setattr(nas_mod.config, "WAKE_NOTICE_ALWAYS", True)
        monkeypatch.setattr(nas_mod.config, "WAKE_NOTICE_GRACE_MINUTES", 120)

        await nas_mod._settle_wake(store, "chat_wake_delayed", now)
        # Delayed beyond grace limit, so notice should be suppressed!
        assert len(pushed_prompts) == 0

    @pytest.mark.asyncio
    async def test_resume_sleep_tool_error_handling(self, monkeypatch):
        _setup_mock_nekro_agent()
        nas_mod = sys.modules["nekro_auto_sleep"]
        from nekro_auto_sleep.persistence import SleepStateStore

        class MockBackend:
            def __init__(self):
                self.data = {}
            async def get(self, chat_key=None, store_key=""):
                return self.data.get(store_key)
            async def set(self, chat_key=None, store_key="", value=""):
                self.data[store_key] = value

        store = SleepStateStore(MockBackend())
        init_state = ChatSleepState(
            chat_key="c_awake",
            status=SleepStatus.AWAKE,
        )
        await store.save(init_state)
        monkeypatch.setattr(nas_mod, "_store", store)

        class MockCtx:
            chat_key = "c_awake"
            async def send_text(self, text, record=False): pass

        # Calling resume_sleep when state is AWAKE (not AWAKE_EARLY) triggers ValueError
        res = await nas_mod.resume_sleep_tool(MockCtx())
        assert "无法重新入睡" in res

    @pytest.mark.asyncio
    async def test_persistence_with_state_saves_in_place_mutation(self):
        from nekro_auto_sleep.persistence import SleepStateStore

        class MockBackend:
            def __init__(self):
                self.data = {}
            async def get(self, chat_key=None, store_key=""):
                return self.data.get(store_key)
            async def set(self, chat_key=None, store_key="", value=""):
                self.data[store_key] = value

        backend = MockBackend()
        store = SleepStateStore(backend)
        init_state = ChatSleepState(chat_key="test_save", status=SleepStatus.AWAKE)
        await store.save(init_state)

        t_now = datetime(2026, 9, 2, 10, 0, tzinfo=UTC)

        async def _mutate(s: ChatSleepState) -> ChatSleepState:
            s.last_seen_at = t_now
            return s

        await store.with_state("test_save", _mutate)
        # Verify it was saved to backend
        reloaded = await store.load_or_create("test_save")
        assert reloaded.last_seen_at == t_now

    @pytest.mark.asyncio
    async def test_layer3_run_agent_task_wrapper_allows_permission_lease(self):
        from nekro_auto_sleep.runtime import (
            make_run_agent_task_wrapper,
            lease_ledger,
            SourceType,
        )

        lease_ledger.clear()
        chat_key = "test_l3_chat"
        # Bot is sleeping
        is_sleeping_fn = lambda ck: True
        has_permission_fn = lambda ck: lease_ledger.has_active_for_chat(ck)

        called = []
        async def original_agent_task(*args, **kwargs):
            called.append(True)
            return "agent_result"

        wrapper = make_run_agent_task_wrapper(
            is_sleeping_fn,
            has_permission_fn,
            on_agent_start_fn=AsyncMock(),
            on_agent_end_fn=AsyncMock(),
        )

        # 1. No lease/permission -> blocked
        res = await wrapper(original_agent_task, chat_key=chat_key)
        assert res is None
        assert len(called) == 0

        # 2. Active lease exists -> allowed
        lease_ledger.create("lease_wake", SourceType.INTERNAL_WAKE_NOTICE, chat_key, "wake", ttl=10.0)
        res = await wrapper(original_agent_task, chat_key=chat_key)
        assert res == "agent_result"
        assert len(called) == 1
        lease_ledger.clear()

    def test_plugin_active_fail_open_when_disabled(self, monkeypatch):
        _setup_mock_nekro_agent()
        nas_mod = sys.modules["nekro_auto_sleep"]

        # 1. When runtime is not active, _is_plugin_active returns False
        monkeypatch.setattr(nas_mod, "_is_runtime_active", False)
        assert nas_mod._is_plugin_active() is False
        assert nas_mod._is_sleeping("any_chat") is False

        # 2. When runtime is active but host plugin.enabled is False, returns False
        monkeypatch.setattr(nas_mod, "_is_runtime_active", True)
        monkeypatch.setattr(nas_mod.plugin, "enabled", False, raising=False)
        assert nas_mod._is_plugin_active() is False
        assert nas_mod._is_sleeping("any_chat") is False

    def test_wake_intent_detection(self):
        _setup_mock_nekro_agent()
        nas_mod = sys.modules["nekro_auto_sleep"]

        class FakeMsg:
            def __init__(self, text: str, sender: str = "u1", is_tome: bool = False, reply_id: str | None = None):
                self.content_text = text
                self.sender = sender
                self.is_tome = is_tome
                self.reply_message_id = reply_id

        # 1. Direct positive keywords
        is_conf, is_canc = nas_mod._check_wake_intent(FakeMsg("要"), persona_name="bot")
        assert is_conf is True
        assert is_canc is False

        is_conf, is_canc = nas_mod._check_wake_intent(FakeMsg("醒来"), persona_name="bot")
        assert is_conf is True
        assert is_canc is False

        # 2. Negative cancellation takes priority
        is_conf, is_canc = nas_mod._check_wake_intent(FakeMsg("不要"), persona_name="bot")
        assert is_canc is True
        assert is_conf is False

        is_conf, is_canc = nas_mod._check_wake_intent(FakeMsg("算了吧，你继续睡"), persona_name="bot")
        assert is_canc is True
        assert is_conf is False

        # 3. Mentioning bot in confirmation window
        is_conf, is_canc = nas_mod._check_wake_intent(FakeMsg("bot 起来一下", is_tome=False), persona_name="bot")
        assert is_conf is True
        assert is_canc is False

        is_conf, is_canc = nas_mod._check_wake_intent(FakeMsg("帮我查个东西", is_tome=True), persona_name="bot")
        assert is_conf is True
        assert is_canc is False

        # 4. Unrelated group chatter without mention or confirm keyword
        is_conf, is_canc = nas_mod._check_wake_intent(FakeMsg("今天去吃什么"), persona_name="bot")
        assert is_conf is False
        assert is_canc is False

        # 5. Long sentence containing single character keyword '要' without mentioning bot
        # e.g. "我要出门了待会儿再回来" -> should NOT be treated as confirmation
        is_conf, is_canc = nas_mod._check_wake_intent(FakeMsg("我要出门了待会儿再回来"), persona_name="bot")
        assert is_conf is False
        assert is_canc is False

