"""The sleep notice must follow the model's own line, not precede it.

``resume_sleep`` runs inside the agent round. Sending 【已睡下】 inline put it
*above* the reply the model was about to produce, because that reply only goes
out when the round ends. So the notice is deferred until the chat's running task
finishes — with a hard wait limit, a short fallback when there is no task handle,
and cancellation on shutdown.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from conftest import FakeStoreBackend
from hoststub import AgentCtx
from nekro_auto_sleep.engine import transition_to_sleep
from nekro_auto_sleep.models import ChatSleepState, SleepStatus
from nekro_auto_sleep.persistence import SleepStateStore

UTC = ZoneInfo("UTC")
CHAT_KEY = "onebot_v11-group_888"
NIGHT = datetime(2026, 8, 17, 15, 0, tzinfo=UTC)


@pytest.fixture
def wired(plugin_mod):
    store = SleepStateStore(FakeStoreBackend())
    plugin_mod._store = store
    plugin_mod._is_runtime_active = True
    plugin_mod._persona_cache.clear()
    plugin_mod.invalidate_settings_cache()
    plugin_mod._cancel_deferred_sends()
    yield plugin_mod, store
    plugin_mod._cancel_deferred_sends()


async def _early_awake(plugin_mod, store) -> None:
    snap = plugin_mod._make_config_snapshot()
    state = transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), NIGHT, snap)
    cycle = state.cycle.model_copy(
        update={"planned_wake_at": datetime.now(UTC) + timedelta(hours=4)}
    )
    state = state.model_copy(
        update={
            "status": SleepStatus.AWAKE_EARLY,
            "cycle": cycle,
            "woken_at": datetime.now(UTC),
            "woken_by": "u_waker",
            "idle_sleep_deadline": datetime.now(UTC) + timedelta(minutes=10),
        }
    )
    await store.save(state)


async def _drain() -> None:
    """Let the deferred task run to completion."""
    for _ in range(50):
        await asyncio.sleep(0.01)
        pending = [t for t in asyncio.all_tasks() if not t.done() and t is not asyncio.current_task()]
        if not pending:
            return


class TestDeferredSend:
    async def test_notice_is_not_sent_inline(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store)
        ctx = AgentCtx(CHAT_KEY)

        result = await plugin_mod.resume_sleep(ctx)

        assert result == "ok"
        assert ctx.sent == []  # nothing went out during the tool call
        plugin_mod._cancel_deferred_sends()

    async def test_notice_is_sent_after_the_round(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store)
        ctx = AgentCtx(CHAT_KEY)
        plugin_mod.set_ctx_factory(lambda _chat_key: ctx) if hasattr(
            plugin_mod, "set_ctx_factory"
        ) else None
        import hoststub

        hoststub.set_ctx_factory(lambda _chat_key: ctx)

        await plugin_mod.resume_sleep(ctx)
        await _drain()

        assert [text for text, _record in ctx.sent] == ["【小助手已睡下】"]
        assert store.get_cached(CHAT_KEY).status == SleepStatus.ASLEEP

    async def test_waits_for_a_running_round_before_sending(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store)
        ctx = AgentCtx(CHAT_KEY)
        import hoststub

        hoststub.set_ctx_factory(lambda _chat_key: ctx)
        message_service = hoststub.message_service()
        round_task = asyncio.create_task(asyncio.sleep(0.15))
        message_service.running_tasks[CHAT_KEY] = round_task

        await plugin_mod.resume_sleep(ctx)
        await asyncio.sleep(0.02)
        assert ctx.sent == []  # still waiting on the round

        await round_task
        await _drain()
        assert len(ctx.sent) == 1

    async def test_falls_back_to_a_short_pause_without_a_task(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store)
        ctx = AgentCtx(CHAT_KEY)
        import hoststub

        hoststub.set_ctx_factory(lambda _chat_key: ctx)

        await plugin_mod.resume_sleep(ctx)
        await _drain()

        assert len(ctx.sent) == 1

    async def test_a_finished_task_does_not_block(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store)
        ctx = AgentCtx(CHAT_KEY)
        import hoststub

        hoststub.set_ctx_factory(lambda _chat_key: ctx)
        done = asyncio.create_task(asyncio.sleep(0))
        await done
        hoststub.message_service().running_tasks[CHAT_KEY] = done

        await plugin_mod.resume_sleep(ctx)
        await _drain()

        assert len(ctx.sent) == 1

    async def test_round_wait_is_capped(self, wired):
        """A stuck round must not keep the deferred task alive forever."""
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store)
        ctx = AgentCtx(CHAT_KEY)
        import hoststub

        hoststub.set_ctx_factory(lambda _chat_key: ctx)
        stuck = asyncio.create_task(asyncio.sleep(3600))
        hoststub.message_service().running_tasks[CHAT_KEY] = stuck

        await plugin_mod.resume_sleep(ctx)
        task = next(iter(plugin_mod._deferred_sends))
        assert plugin_mod._ROUND_WAIT_SECONDS == 120.0

        plugin_mod._cancel_deferred_sends()
        stuck.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_failure_to_send_is_logged_not_raised(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store)

        class BrokenCtx(AgentCtx):
            async def send_text(self, content, *, record=True):
                raise RuntimeError("platform down")

        import hoststub

        hoststub.set_ctx_factory(lambda _chat_key: BrokenCtx(CHAT_KEY))

        await plugin_mod.resume_sleep(AgentCtx(CHAT_KEY))
        await _drain()

        # The state change stands: the bot did go back to sleep.
        assert store.get_cached(CHAT_KEY).status == SleepStatus.ASLEEP

    async def test_shutdown_cancels_pending_notices(self, wired, monkeypatch):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store)
        import hoststub

        hoststub.set_ctx_factory(lambda _chat_key: AgentCtx(CHAT_KEY))
        blocker = asyncio.create_task(asyncio.sleep(3600))
        hoststub.message_service().running_tasks[CHAT_KEY] = blocker

        await plugin_mod.resume_sleep(AgentCtx(CHAT_KEY))
        task = next(iter(plugin_mod._deferred_sends))
        assert not task.done()

        async def _noop(*args, **kwargs):
            return None

        monkeypatch.setattr(plugin_mod, "_uninstall_wraps", lambda: None)
        monkeypatch.setattr(plugin_mod, "_remove_log_bridge", lambda: None)
        await plugin_mod._stop_runtime()

        assert plugin_mod._deferred_sends == set()
        with pytest.raises(asyncio.CancelledError):
            await task
        blocker.cancel()

    async def test_task_set_is_emptied_when_the_notice_finishes(self, wired):
        plugin_mod, store = wired
        await _early_awake(plugin_mod, store)
        import hoststub

        hoststub.set_ctx_factory(lambda _chat_key: AgentCtx(CHAT_KEY))

        await plugin_mod.resume_sleep(AgentCtx(CHAT_KEY))
        assert len(plugin_mod._deferred_sends) == 1

        await _drain()
        assert plugin_mod._deferred_sends == set()


class TestResumeGuards:
    async def test_awake_chat_still_returns_an_error_text(self, wired):
        plugin_mod, store = wired
        result = await plugin_mod.resume_sleep(AgentCtx(CHAT_KEY))
        assert "无法重新入睡" in result
        assert plugin_mod._deferred_sends == set()
