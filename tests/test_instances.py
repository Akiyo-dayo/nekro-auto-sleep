"""Per-instance behaviour: each channel follows its own instance's config.

Two bots of one deployment can sit in the same group. Their chat_keys differ, so
their nights already stay apart — but resolving the *config* per channel is what
lets them keep different hours. Everything here goes through the background-safe
entry point (``resolve_scoped_config`` with an explicit instance key), and every
failure mode must fall back to the global config, i.e. behave exactly like 1.2.5.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from conftest import FakeStoreBackend
from nekro_auto_sleep.models import ChatSleepState, SleepStatus
from nekro_auto_sleep.persistence import SleepStateStore

UTC = ZoneInfo("UTC")
TZ = ZoneInfo("Asia/Shanghai")

BOT_A = "onebot_v11-botA-group_777"
BOT_B = "onebot_v11-botB-group_777"

# 2026-08-17 22:30 +08:00
EVENING = datetime(2026, 8, 17, 14, 30, tzinfo=UTC)


@pytest.fixture
def wired(plugin_mod, monkeypatch):
    backend = FakeStoreBackend()
    store = SleepStateStore(backend)
    plugin_mod._store = store
    plugin_mod._is_runtime_active = True
    plugin_mod.invalidate_settings_cache()
    plugin_mod.config.SLEEP_TIME = "23:00"
    plugin_mod.config.WAKE_TIME_START = "06:45"
    plugin_mod.config.WAKE_TIME_END = "08:30"
    plugin_mod.config.TIMEZONE = "Asia/Shanghai"
    import hoststub

    hoststub.clear_instance_config_overrides()
    hoststub.set_channel_factory(None)
    return plugin_mod, store


async def _prime(plugin_mod, chat_key: str) -> None:
    """Materialize a chat_key so maintenance (which only sees cached state) sees it."""
    store = plugin_mod._store

    async def _touch(state: ChatSleepState) -> ChatSleepState:
        return state

    await store.with_state(chat_key, _touch)


class TestInstanceSettings:
    async def test_each_channel_uses_its_own_instance_config(self, wired, monkeypatch):
        plugin_mod, store = wired
        import hoststub

        hoststub.set_instance_config_override(
            plugin_mod.plugin.key, "botA", {"SLEEP_TIME": "22:00"}
        )
        hoststub.set_instance_config_override(
            plugin_mod.plugin.key, "botB", {"SLEEP_TIME": "23:30"}
        )

        settings_a = await plugin_mod._settings_for(BOT_A)
        settings_b = await plugin_mod._settings_for(BOT_B)
        assert settings_a.instance_key == "botA"
        assert settings_b.instance_key == "botB"
        assert settings_a.config.SLEEP_TIME == "22:00"
        assert settings_b.config.SLEEP_TIME == "23:30"

    async def test_each_channel_falls_asleep_on_its_own_schedule(self, wired):
        plugin_mod, store = wired
        import hoststub

        hoststub.set_instance_config_override(
            plugin_mod.plugin.key, "botA", {"SLEEP_TIME": "22:00"}
        )
        hoststub.set_instance_config_override(
            plugin_mod.plugin.key, "botB", {"SLEEP_TIME": "23:30"}
        )
        await _prime(plugin_mod, BOT_A)
        await _prime(plugin_mod, BOT_B)

        # 22:30 local: only botA's bedtime has passed.
        await plugin_mod._maintain_chat(store, BOT_A, EVENING)
        await plugin_mod._maintain_chat(store, BOT_B, EVENING)

        assert store.get_cached(BOT_A).status == SleepStatus.ASLEEP
        assert store.get_cached(BOT_B).status == SleepStatus.AWAKE

        # 23:45 local: botB follows.
        await plugin_mod._maintain_chat(store, BOT_B, EVENING + timedelta(hours=1, minutes=15))
        assert store.get_cached(BOT_B).status == SleepStatus.ASLEEP

    async def test_cycle_records_the_instance_schedule(self, wired):
        plugin_mod, store = wired
        import hoststub

        hoststub.set_instance_config_override(
            plugin_mod.plugin.key, "botA", {"SLEEP_TIME": "22:00", "TIMEZONE": "Asia/Tokyo"}
        )
        await _prime(plugin_mod, BOT_A)
        await plugin_mod._check_sleep_transition(store, BOT_A, EVENING, TZ)

        cycle = store.get_cached(BOT_A).cycle
        assert cycle is not None
        assert cycle.config_snapshot.sleep_time == "22:00"
        assert cycle.config_snapshot.timezone == "Asia/Tokyo"

    async def test_channel_without_instance_uses_the_global_config(self, wired):
        plugin_mod, _store = wired
        settings = await plugin_mod._settings_for("onebot_v11-group_777")
        assert settings.instance_key == ""
        assert settings.config.SLEEP_TIME == "23:00"

    async def test_resolve_failure_falls_back_to_global(self, wired, monkeypatch):
        plugin_mod, _store = wired
        import sys

        scope = sys.modules["nekro_agent.services.plugin.scope"]

        def _boom(*args, **kwargs):
            raise RuntimeError("host exploded")

        monkeypatch.setattr(scope, "resolve_scoped_config", _boom)

        settings = await plugin_mod._settings_for(BOT_A)

        assert settings.instance_key == "botA"
        assert settings.config is plugin_mod.config
        assert settings.config.SLEEP_TIME == "23:00"

    async def test_missing_scope_module_falls_back_to_global(self, wired, monkeypatch):
        plugin_mod, _store = wired
        import sys

        monkeypatch.setitem(sys.modules, "nekro_agent.services.plugin.scope", None)
        settings = await plugin_mod._settings_for(BOT_A)
        assert settings.config is plugin_mod.config

    async def test_channel_lookup_failure_falls_back_to_global(self, wired):
        plugin_mod, _store = wired
        import hoststub

        def _broken(_chat_key: str):
            raise RuntimeError("no channel today")

        hoststub.set_channel_factory(_broken)

        settings = await plugin_mod._settings_for(BOT_A)

        assert settings.instance_key == ""
        assert settings.config is plugin_mod.config

    async def test_settings_cache_is_keyed_per_channel(self, wired):
        plugin_mod, _store = wired
        import hoststub

        hoststub.set_instance_config_override(
            plugin_mod.plugin.key, "botA", {"SLEEP_TIME": "22:00"}
        )
        first = await plugin_mod._settings_for(BOT_A)
        second = await plugin_mod._settings_for(BOT_B)
        third = await plugin_mod._settings_for(BOT_A)
        assert first.config.SLEEP_TIME == "22:00"
        assert second.config.SLEEP_TIME == "23:00"
        assert third is first  # cached, not re-resolved

        plugin_mod.invalidate_settings_cache(BOT_A)
        assert (await plugin_mod._settings_for(BOT_A)) is not first
