"""The model calls ``resume_sleep()``, and only when it can mean something.

Two separate contracts:
1. The sandbox name comes from ``func.__name__`` (host: ``services/plugin/
   base.py`` renders it into the prompt, ``services/sandbox/ext_caller.py``
   registers the container proxy by it). A ``*_tool`` suffix therefore leaked
   into what the model was told to call.
2. ``resume_sleep`` is disclosed only while AWAKE_EARLY and before the planned
   wake; ``get_sleep_report`` is always disclosed. That needs
   ``allow_sleep=False``, because a dormant plugin discloses nothing at all.
"""

from __future__ import annotations

import ast
import pathlib
from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from conftest import FakeStoreBackend
from hoststub import AgentCtx
from nekro_auto_sleep.engine import transition_to_sleep
from nekro_auto_sleep.models import ChatSleepState, SleepStatus
from nekro_auto_sleep.persistence import SleepStateStore

UTC = ZoneInfo("UTC")
CHAT_KEY = "onebot_v11-group_777"
NIGHT = datetime(2026, 8, 17, 15, 0, tzinfo=UTC)

_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def wired(plugin_mod):
    store = SleepStateStore(FakeStoreBackend())
    plugin_mod._store = store
    plugin_mod._is_runtime_active = True
    plugin_mod._persona_cache.clear()
    plugin_mod.invalidate_settings_cache()
    yield plugin_mod, store


class TestSandboxNames:
    def test_functions_are_named_what_the_model_calls(self, plugin_mod):
        assert plugin_mod.resume_sleep.__name__ == "resume_sleep"
        assert plugin_mod.get_sleep_report.__name__ == "get_sleep_report"

    def test_mounted_names_match_the_functions(self, plugin_mod):
        mounted = plugin_mod.plugin.sandbox_methods
        assert mounted["resume_sleep"] is plugin_mod.resume_sleep
        assert mounted["get_sleep_report"] is plugin_mod.get_sleep_report
        for name, func in mounted.items():
            assert func.__name__ == name, f"{name} would render as {func.__name__}()"

    def test_old_names_are_aliases_not_wrappers(self, plugin_mod):
        """External docs/scripts still import the old names."""
        assert plugin_mod.resume_sleep_tool is plugin_mod.resume_sleep
        assert plugin_mod.get_sleep_report_tool is plugin_mod.get_sleep_report

    def test_plugin_does_not_go_dormant(self, plugin_mod):
        """A dormant plugin discloses no methods at all — the report must survive."""
        assert plugin_mod.plugin.init_kwargs["allow_sleep"] is False
        assert plugin_mod.plugin.init_kwargs["sleep_brief"].strip()

    def test_host_renders_method_name_from_func_name(self):
        """Recorded host contract, so a future rename cannot silently regress."""
        src = (pathlib.Path("C:/Users/xiaoqian/nr") / "nekro_agent/services/plugin/base.py").read_text(
            encoding="utf-8"
        )
        assert "method.func.__name__" in src
        assert "prompts.append(f\"* {method.func.__name__}" in src

    def test_docstring_examples_use_the_real_name(self):
        for name in ("__init__.py",):
            tree = ast.parse((_ROOT / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in {
                    "resume_sleep",
                    "get_sleep_report",
                }:
                    assert f"{node.name}()" in (ast.get_docstring(node) or "")


async def _early_awake(plugin_mod, store, *, planned_wake_delta: timedelta = timedelta(hours=4)):
    snap = plugin_mod._make_config_snapshot()
    state = transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), NIGHT, snap)
    cycle = state.cycle.model_copy(
        update={"planned_wake_at": datetime.now(UTC) + planned_wake_delta}
    )
    state = state.model_copy(
        update={
            "status": SleepStatus.AWAKE_EARLY,
            "cycle": cycle,
            "idle_sleep_deadline": datetime.now(UTC) + timedelta(minutes=10),
        }
    )
    await store.save(state)


class TestDisclosure:
    async def test_awake_discloses_only_the_report(self, wired):
        plugin_mod, store = wired
        ctx = AgentCtx(CHAT_KEY)

        methods = await plugin_mod.collect_available_methods(ctx)

        assert methods == [plugin_mod.get_sleep_report]

    async def test_asleep_discloses_only_the_report(self, wired):
        plugin_mod, store = wired
        snap = plugin_mod._make_config_snapshot()
        await store.save(transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), NIGHT, snap))

        methods = await plugin_mod.collect_available_methods(AgentCtx(CHAT_KEY))

        assert methods == [plugin_mod.get_sleep_report]

    async def test_early_awake_discloses_resume_sleep(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store)

        methods = await plugin_mod.collect_available_methods(AgentCtx(CHAT_KEY))

        assert plugin_mod.resume_sleep in methods
        assert plugin_mod.get_sleep_report in methods

    async def test_early_awake_after_planned_wake_hides_resume_sleep(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store, planned_wake_delta=timedelta(minutes=-1))

        methods = await plugin_mod.collect_available_methods(AgentCtx(CHAT_KEY))

        assert plugin_mod.resume_sleep not in methods
        assert plugin_mod.get_sleep_report in methods

    async def test_inactive_plugin_discloses_nothing(self, wired):
        plugin_mod, _store = wired
        plugin_mod._is_runtime_active = False

        assert await plugin_mod.collect_available_methods(AgentCtx(CHAT_KEY)) == []

    async def test_no_store_discloses_nothing(self, wired):
        plugin_mod, _store = wired
        plugin_mod._store = None

        assert await plugin_mod.collect_available_methods(AgentCtx(CHAT_KEY)) == []

    async def test_unknown_chat_still_gets_the_report(self, wired):
        plugin_mod, _store = wired
        methods = await plugin_mod.collect_available_methods(AgentCtx("onebot_v11-group_new"))
        assert methods == [plugin_mod.get_sleep_report]
