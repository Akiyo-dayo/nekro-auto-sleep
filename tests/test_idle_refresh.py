"""After an early wake, only messages aimed at the bot keep it up.

1.2.5 refreshed the idle-sleep deadline on every message in the channel, so a
group chatting among itself at 2am kept the bot awake indefinitely — observed
live: woken at 23:01, still up three hours later with the deadline nine minutes
out. The deadline now moves only for a message aimed at the bot (mention, reply,
call keyword) or from the person who woke it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from conftest import FakeStoreBackend
from hoststub import AgentCtx, ChatMessage, MsgSignal
from nekro_auto_sleep.engine import transition_to_sleep
from nekro_auto_sleep.models import ChatSleepState, SleepStatus
from nekro_auto_sleep.persistence import SleepStateStore

UTC = ZoneInfo("UTC")
TZ = ZoneInfo("Asia/Shanghai")
CHAT_KEY = "onebot_v11-group_555"
NIGHT = datetime(2026, 8, 17, 15, 0, tzinfo=UTC)  # 23:00 local
WOKEN_AT = NIGHT + timedelta(minutes=30)


@pytest.fixture
def wired(plugin_mod):
    store = SleepStateStore(FakeStoreBackend())
    plugin_mod._store = store
    plugin_mod._is_runtime_active = True
    plugin_mod._persona_cache.clear()
    plugin_mod._wake_inject_cache.clear()
    plugin_mod.invalidate_settings_cache()
    plugin_mod.invalidate_own_accounts_cache()
    yield plugin_mod, store
    plugin_mod._persona_cache.clear()
    plugin_mod._wake_inject_cache.clear()


def _msg(text: str, sender: str) -> ChatMessage:
    return ChatMessage(content=text, chat_key=CHAT_KEY, platform_userid=sender)


async def _early_awake(plugin_mod, store, *, woken_by: str | None = "u_waker") -> None:
    snap = plugin_mod._make_config_snapshot()
    state = transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), NIGHT, snap)
    state = state.model_copy(
        update={
            "status": SleepStatus.AWAKE_EARLY,
            "woken_at": WOKEN_AT,
            "woken_by": woken_by,
            "idle_sleep_deadline": WOKEN_AT + timedelta(minutes=10),
        }
    )
    await store.save(state)


async def _send(plugin_mod, store, message: ChatMessage):
    ctx = AgentCtx(CHAT_KEY)
    signal = await plugin_mod.on_user_message(ctx, message)
    return signal, store.get_cached(CHAT_KEY)


class TestIdleRefresh:
    async def test_unrelated_group_chatter_does_not_refresh(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store)
        before = store.get_cached(CHAT_KEY).idle_sleep_deadline

        signal, state = await _send(plugin_mod, store, _msg("今晚吃啥", "u_bystander"))

        assert signal is MsgSignal.CONTINUE
        assert state.status == SleepStatus.AWAKE_EARLY
        assert state.idle_sleep_deadline == before

    async def test_message_from_woken_by_refreshes_without_mention(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store, woken_by="u_waker")

        _signal, state = await _send(plugin_mod, store, _msg("再陪我一会儿", "u_waker"))

        assert state.idle_sleep_deadline > WOKEN_AT + timedelta(minutes=10)

    async def test_mention_refreshes(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store, woken_by="u_waker")

        _signal, state = await _send(plugin_mod, store, _msg("小助手 帮我看下", "u_bystander"))

        assert state.idle_sleep_deadline > WOKEN_AT + timedelta(minutes=10)

    async def test_call_keyword_refreshes(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store, woken_by="u_waker")
        plugin_mod.config.CALL_KEYWORDS = "醒醒"
        plugin_mod.invalidate_settings_cache()

        _signal, state = await _send(plugin_mod, store, _msg("醒醒 别睡了", "u_bystander"))

        assert state.idle_sleep_deadline > WOKEN_AT + timedelta(minutes=10)

    async def test_reply_to_bot_refreshes(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store, woken_by="u_waker")
        message = _msg("嗯", "u_bystander")
        message.ext_data = {"reply_to_bot": True}

        _signal, state = await _send(plugin_mod, store, message)

        assert state.idle_sleep_deadline > WOKEN_AT + timedelta(minutes=10)

    async def test_no_woken_by_still_requires_aim(self, wired):
        """A row migrated from 1.2.5 has no woken_by: only aimed messages count."""
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store, woken_by=None)
        before = store.get_cached(CHAT_KEY).idle_sleep_deadline

        _signal, state = await _send(plugin_mod, store, _msg("随便聊聊", "u_anyone"))
        assert state.idle_sleep_deadline == before

        _signal, state = await _send(plugin_mod, store, _msg("小助手 在吗", "u_anyone"))
        assert state.idle_sleep_deadline > before

    async def test_a_stream_of_chatter_cannot_keep_it_awake_forever(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store, woken_by="u_waker")
        deadline = store.get_cached(CHAT_KEY).idle_sleep_deadline

        for i in range(20):
            _signal, state = await _send(
                plugin_mod, store, _msg(f"水群第 {i} 条", "u_bystander")
            )
            assert state.idle_sleep_deadline == deadline

        # ...and the idle timeout then puts it back to sleep.
        assert plugin_mod.is_idle_expired(state, deadline + timedelta(seconds=1))


class TestWokenFields:
    def test_early_wake_records_who_and_when(self, plugin_mod):
        from nekro_auto_sleep.engine import transition_to_awake_early

        snap = plugin_mod._make_config_snapshot()
        state = transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), NIGHT, snap)
        state = transition_to_awake_early(state, WOKEN_AT, "u_waker", 10)

        assert state.woken_at == WOKEN_AT
        assert state.woken_by == "u_waker"

    def test_new_cycle_clears_woken_fields(self, plugin_mod):
        snap = plugin_mod._make_config_snapshot()
        state = ChatSleepState(
            chat_key=CHAT_KEY,
            woken_at=WOKEN_AT,
            woken_by="u_waker",
            status=SleepStatus.AWAKE,
        )
        state = transition_to_sleep(state, NIGHT + timedelta(days=1), snap)
        assert state.woken_at is None
        assert state.woken_by is None

    def test_settlement_clears_woken_fields(self, plugin_mod):
        from nekro_auto_sleep.engine import transition_to_awake

        snap = plugin_mod._make_config_snapshot()
        state = transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), NIGHT, snap)
        state = state.model_copy(update={"woken_at": WOKEN_AT, "woken_by": "u_waker"})
        state = transition_to_awake(state, NIGHT + timedelta(hours=8))
        assert state.woken_at is None
        assert state.woken_by is None

    def test_sleeping_back_clears_woken_fields(self, plugin_mod):
        from nekro_auto_sleep.engine import transition_resume_sleep

        snap = plugin_mod._make_config_snapshot()
        state = transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), NIGHT, snap)
        state = state.model_copy(update={"woken_at": WOKEN_AT, "woken_by": "u_waker"})
        state = transition_resume_sleep(state, WOKEN_AT + timedelta(minutes=5))
        assert state.woken_at is None
        assert state.woken_by is None
