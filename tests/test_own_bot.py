"""Messages sent by this deployment's own bots must not count as calls.

Two instances of one deployment can share a group. The wake-up prompt contains
the persona name, which is itself a call keyword, so unguarded the two bots call
each other all night. The account ids come from
``DBAdapterInstance.provider_account_id``; when that cannot be read the plugin
must behave exactly like 1.2.5 (no own-bot exclusions at all).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from conftest import FakeStoreBackend
from nekro_auto_sleep.engine import open_sleep_segment, transition_to_sleep
from nekro_auto_sleep.models import ChatSleepState, SleepStatus
from nekro_auto_sleep.persistence import SleepStateStore

UTC = ZoneInfo("UTC")
TZ = ZoneInfo("Asia/Shanghai")
CHAT_KEY = "onebot_v11-group_123456789"
BEDTIME = datetime(2026, 8, 17, 15, 0, tzinfo=UTC)


@pytest.fixture
def wired(plugin_mod):
    store = SleepStateStore(FakeStoreBackend())
    plugin_mod._store = store
    plugin_mod._is_runtime_active = True
    plugin_mod._persona_cache.clear()
    plugin_mod._wake_inject_cache.clear()
    plugin_mod.invalidate_own_accounts_cache()
    plugin_mod.invalidate_settings_cache()
    plugin_mod.config.HISTORY_MODE = "preserve"
    import hoststub

    hoststub.set_channel_factory(None)
    yield plugin_mod, store
    hoststub.set_own_bot_accounts([])
    plugin_mod.invalidate_own_accounts_cache()


async def _asleep(store: SleepStateStore, plugin_mod) -> None:
    snap = plugin_mod._make_config_snapshot()
    state = transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), BEDTIME, snap)
    state = open_sleep_segment(state, BEDTIME)
    await store.save(state)


def _msg(text: str, sender: str) -> "hoststub.ChatMessage":  # noqa: F821
    import hoststub

    return hoststub.ChatMessage(content=text, chat_key=CHAT_KEY, platform_userid=sender)


class TestOwnBotMessages:
    async def test_own_bot_does_not_get_a_wake_offer(self, wired):
        plugin_mod, store = wired
        import hoststub

        hoststub.set_own_bot_accounts(["bot_self"])
        plugin_mod.invalidate_own_accounts_cache()
        await _asleep(store, plugin_mod)
        ctx = hoststub.AgentCtx(CHAT_KEY)

        signal = await plugin_mod.on_user_message(ctx, _msg("醒醒", "bot_self"))

        assert ctx.sent == []  # no "要叫醒吗" question
        state = store.get_cached(CHAT_KEY)
        assert state is not None
        assert state.pending_wake_offers == {}
        assert signal is not hoststub.MsgSignal.FORCE_TRIGGER
        assert signal is hoststub.MsgSignal.BLOCK_TRIGGER  # master's asleep path

    async def test_normal_user_still_gets_the_question(self, wired):
        plugin_mod, store = wired
        import hoststub

        hoststub.set_own_bot_accounts(["bot_self"])
        plugin_mod.invalidate_own_accounts_cache()
        await _asleep(store, plugin_mod)
        ctx = hoststub.AgentCtx(CHAT_KEY)

        await plugin_mod.on_user_message(ctx, _msg("醒醒", "u_human"))

        assert len(ctx.sent) == 1
        assert store.get_cached(CHAT_KEY).pending_wake_offers

    async def test_own_bot_while_awake_is_continue_and_records_no_call(self, wired):
        plugin_mod, store = wired
        import hoststub

        hoststub.set_own_bot_accounts(["bot_self"])
        plugin_mod.invalidate_own_accounts_cache()
        ctx = hoststub.AgentCtx(CHAT_KEY)

        signal = await plugin_mod.on_user_message(ctx, _msg("醒醒 bot_self", "bot_self"))

        assert signal is hoststub.MsgSignal.CONTINUE
        assert ctx.sent == []
        state = store.get_cached(CHAT_KEY)
        assert state is not None and state.cycle is None

    async def test_own_bot_does_not_answer_an_open_question(self, wired):
        plugin_mod, store = wired
        import hoststub

        hoststub.set_own_bot_accounts(["bot_self"])
        plugin_mod.invalidate_own_accounts_cache()
        await _asleep(store, plugin_mod)
        # Somebody else called first: an offer is outstanding.
        human_ctx = hoststub.AgentCtx(CHAT_KEY)
        await plugin_mod.on_user_message(human_ctx, _msg("醒醒", "u_human"))
        assert store.get_cached(CHAT_KEY).pending_wake_offers

        bot_ctx = hoststub.AgentCtx(CHAT_KEY)
        signal = await plugin_mod.on_user_message(bot_ctx, _msg("要", "bot_self"))

        assert signal is not hoststub.MsgSignal.FORCE_TRIGGER
        assert store.get_cached(CHAT_KEY).status == SleepStatus.ASLEEP
        assert bot_ctx.sent == []

    async def test_strict_history_mode_blocks_own_bot_completely(self, wired):
        plugin_mod, store = wired
        import hoststub

        hoststub.set_own_bot_accounts(["bot_self"])
        plugin_mod.invalidate_own_accounts_cache()
        await _asleep(store, plugin_mod)
        plugin_mod.config.HISTORY_MODE = "strict"
        plugin_mod.invalidate_settings_cache()

        signal = await plugin_mod.on_user_message(
            hoststub.AgentCtx(CHAT_KEY), _msg("醒醒", "bot_self")
        )

        assert signal is hoststub.MsgSignal.BLOCK_ALL

    async def test_own_bot_does_not_refresh_early_awake_idle(self, wired):
        plugin_mod, store = wired
        import hoststub

        hoststub.set_own_bot_accounts(["bot_self"])
        plugin_mod.invalidate_own_accounts_cache()
        await _asleep(store, plugin_mod)

        async def _wake(state):
            return state.model_copy(
                update={
                    "status": SleepStatus.AWAKE_EARLY,
                    "woken_by": "u_human",
                    "idle_sleep_deadline": BEDTIME + timedelta(minutes=10),
                }
            )

        await store.with_state(CHAT_KEY, _wake)
        deadline = store.get_cached(CHAT_KEY).idle_sleep_deadline

        await plugin_mod.on_user_message(
            hoststub.AgentCtx(CHAT_KEY), _msg("还在吗", "bot_self")
        )

        assert store.get_cached(CHAT_KEY).idle_sleep_deadline == deadline

    async def test_account_lookup_failure_restores_1_2_5_behaviour(self, wired, monkeypatch):
        plugin_mod, store = wired
        import sys

        import hoststub

        class _Broken:
            @classmethod
            def all(cls):
                raise RuntimeError("db down")

        monkeypatch.setitem(
            sys.modules,
            "nekro_agent.models.db_adapter_instance",
            type(sys)("nekro_agent.models.db_adapter_instance"),
        )
        sys.modules["nekro_agent.models.db_adapter_instance"].DBAdapterInstance = _Broken
        plugin_mod.invalidate_own_accounts_cache()
        await _asleep(store, plugin_mod)
        ctx = hoststub.AgentCtx(CHAT_KEY)

        await plugin_mod.on_user_message(ctx, _msg("醒醒", "u_human"))

        # No own-bot knowledge -> the ordinary path asks the question.
        assert len(ctx.sent) == 1

    async def test_missing_model_means_no_own_bots(self, wired, monkeypatch):
        plugin_mod, _store = wired
        import sys

        monkeypatch.setitem(sys.modules, "nekro_agent.models.db_adapter_instance", None)
        plugin_mod.invalidate_own_accounts_cache()

        assert await plugin_mod._own_account_ids() == frozenset()
