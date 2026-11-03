"""Data-safety guarantees for the persisted state row.

The nightly state is the only thing the plugin owns, and it is per channel: one
bad row must never cost a channel its history. Two properties are pinned here:

* a p0-batch1 ``schema_version=2`` row is migrated on read and rewritten as v1;
* a row this build cannot read (bad JSON, a newer schema, a payload that still
  fails validation) is left byte-for-byte in place, the channel behaves like
  sleep is disabled, and the failure is logged exactly once.
"""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from conftest import FakeStoreBackend, hoststub
from nekro_auto_sleep.models import SCHEMA_VERSION, ChatSleepState, SleepStatus
from nekro_auto_sleep.persistence import StateProtectedError, SleepStateStore

CHAT_KEY = "onebot_v11-group_123456789"
TZ = ZoneInfo("Asia/Shanghai")
UTC = ZoneInfo("UTC")
FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "p0_v2_state.json"


def _raw(backend: FakeStoreBackend, chat_key: str) -> str:
    return backend._data[f"{chat_key}||state.v1"]


def _put(backend: FakeStoreBackend, chat_key: str, raw: str) -> None:
    backend._data[f"{chat_key}||state.v1"] = raw


# ---------------------------------------------------------------------------
# Reading data written by the 8 月 test build
# ---------------------------------------------------------------------------


class TestV2Migration:
    async def test_p0_dump_is_readable(self, fake_backend):
        _put(fake_backend, CHAT_KEY, FIXTURE.read_text(encoding="utf-8"))
        store = SleepStateStore(fake_backend)

        state = await store.load(CHAT_KEY)

        assert state is not None
        assert state.status == SleepStatus.ASLEEP
        # The single pending_offer became the per-user offer under its own id.
        assert set(state.pending_wake_offers) == {"u_42"}
        offer = state.pending_wake_offers["u_42"]
        assert offer.user_id == "u_42"
        assert offer.offered_at == datetime(2026, 8, 17, 15, 30, tzinfo=UTC)
        # Shared-by-name v2 fields survive.
        assert state.offers_sent_tonight == 2
        assert state.last_offer_at == datetime(2026, 8, 17, 15, 30, tzinfo=UTC)
        assert state.cycle is not None
        assert state.cycle.cycle_id == "abcdef123456"
        assert state.cycle.config_snapshot.call_keywords == ["醒醒", "起床"]
        # v2-only fields are ignored, not fatal.
        assert not hasattr(state, "snooze_until")

    async def test_p0_dump_is_rewritten_as_v1(self, fake_backend):
        _put(fake_backend, CHAT_KEY, FIXTURE.read_text(encoding="utf-8"))
        store = SleepStateStore(fake_backend)
        state = await store.load(CHAT_KEY)
        assert state is not None

        await store.save(state)

        written = json.loads(_raw(fake_backend, CHAT_KEY))
        assert written["schema_version"] == SCHEMA_VERSION == 1
        assert written["chat_key"] == CHAT_KEY
        assert "pending_offer" not in written
        assert set(written["pending_wake_offers"]) == {"u_42"}
        assert written["offers_sent_tonight"] == 2

        # And the v1 row reads back unchanged.
        fresh = SleepStateStore(fake_backend)
        reread = await fresh.load(CHAT_KEY)
        assert reread is not None and reread.status == SleepStatus.ASLEEP

    async def test_pending_offer_is_kept_when_offers_already_exist(self, fake_backend):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["pending_wake_offers"] = {
            "u_7": {
                "user_id": "u_7",
                "offered_at": "2026-08-17T15:10:00Z",
                "expires_at": "2026-08-17T15:13:00Z",
            }
        }
        payload["pending_offer"]["user_id"] = "u_42"
        _put(fake_backend, CHAT_KEY, json.dumps(payload))
        store = SleepStateStore(fake_backend)

        state = await store.load(CHAT_KEY)

        assert state is not None
        # An existing per-user offer map wins; the single v2 offer is dropped.
        assert set(state.pending_wake_offers) == {"u_7"}

    async def test_v1_row_without_schema_version_still_loads(self, fake_backend):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["schema_version"] = 2
        from nekro_auto_sleep.persistence import migrate_v2_payload

        v1_payload = migrate_v2_payload(payload)
        v1_payload.pop("schema_version")
        _put(fake_backend, CHAT_KEY, json.dumps(v1_payload))
        store = SleepStateStore(fake_backend)

        state = await store.load(CHAT_KEY)

        assert state is not None
        assert state.schema_version == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Unreadable data: never overwrite
# ---------------------------------------------------------------------------


BAD_ROWS = {
    "newer-schema": json.dumps({"schema_version": 3, "chat_key": CHAT_KEY}, ensure_ascii=False),
    "bad-json": '{"schema_version": 1, "chat_key": ',
    "invalid-payload": json.dumps(
        {"schema_version": 1, "chat_key": CHAT_KEY, "status": "SLEEPWALKING"}, ensure_ascii=False
    ),
}


@pytest.fixture
def wiring(plugin_mod):
    """A plugin module with a store, a clock and a captured handler."""
    backend = FakeStoreBackend()
    store = SleepStateStore(backend)
    plugin_mod._store = store
    plugin_mod._is_runtime_active = True
    return plugin_mod, store, backend


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured_logs():
    handler = _Capture()
    log = logging.getLogger("nekro_auto_sleep.persistence")
    log.addHandler(handler)
    level = log.level
    log.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        log.removeHandler(handler)
        log.setLevel(level)


class TestProtectedRows:
    @pytest.mark.parametrize("kind", sorted(BAD_ROWS))
    async def test_load_refuses_and_keeps_the_row(self, kind, fake_backend):
        raw = BAD_ROWS[kind]
        _put(fake_backend, CHAT_KEY, raw)
        store = SleepStateStore(fake_backend)

        with pytest.raises(StateProtectedError):
            await store.load(CHAT_KEY)

        assert store.is_protected(CHAT_KEY)
        assert _raw(fake_backend, CHAT_KEY) == raw
        assert fake_backend.set_calls == 0

    @pytest.mark.parametrize("kind", sorted(BAD_ROWS))
    async def test_with_state_never_writes(self, kind, fake_backend):
        raw = BAD_ROWS[kind]
        _put(fake_backend, CHAT_KEY, raw)
        store = SleepStateStore(fake_backend)
        called = False

        async def _mutate(state):
            nonlocal called
            called = True
            return state.model_copy(update={"status": SleepStatus.ASLEEP})

        with pytest.raises(StateProtectedError):
            await store.with_state(CHAT_KEY, _mutate)

        assert called is False
        assert _raw(fake_backend, CHAT_KEY) == raw
        assert fake_backend.set_calls == 0

    async def test_save_refuses_a_protected_chat_key(self, fake_backend):
        raw = BAD_ROWS["bad-json"]
        _put(fake_backend, CHAT_KEY, raw)
        store = SleepStateStore(fake_backend)
        with pytest.raises(StateProtectedError):
            await store.load(CHAT_KEY)

        with pytest.raises(StateProtectedError):
            await store.save(ChatSleepState(chat_key=CHAT_KEY))

        assert _raw(fake_backend, CHAT_KEY) == raw

    @pytest.mark.parametrize("kind", sorted(BAD_ROWS))
    async def test_message_then_maintenance_leaves_row_untouched(
        self, kind, wiring, captured_logs
    ):
        plugin_mod, store, backend = wiring
        raw = BAD_ROWS[kind]
        _put(backend, CHAT_KEY, raw)

        message = hoststub.ChatMessage(
            content="在吗", chat_key=CHAT_KEY, platform_userid="u1"
        )
        ctx = hoststub.AgentCtx(CHAT_KEY)

        signal = await plugin_mod.on_user_message(ctx, message)
        assert signal is hoststub.MsgSignal.CONTINUE
        assert ctx.sent == []

        now = datetime(2026, 8, 17, 15, 30, tzinfo=UTC)
        plugin_mod._store = store
        await plugin_mod._maintain_chat(store, CHAT_KEY, now, TZ)

        assert _raw(backend, CHAT_KEY) == raw
        assert backend.set_calls == 0
        errors = [r for r in captured_logs.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        assert "数据已原样保留，未覆盖" in errors[0].getMessage()

    async def test_error_is_logged_only_once_per_process(self, wiring, captured_logs):
        plugin_mod, store, backend = wiring
        _put(backend, CHAT_KEY, BAD_ROWS["newer-schema"])

        for _ in range(4):
            with pytest.raises(StateProtectedError):
                await store.load(CHAT_KEY)
            assert await plugin_mod.on_user_message(
                hoststub.AgentCtx(CHAT_KEY),
                hoststub.ChatMessage(content="醒醒", chat_key=CHAT_KEY),
            ) is hoststub.MsgSignal.CONTINUE

        errors = [r for r in captured_logs.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        assert "数据版本 3" in errors[0].getMessage()

    async def test_protected_channel_does_not_gate_the_agent(self, wiring):
        plugin_mod, store, backend = wiring
        _put(backend, CHAT_KEY, BAD_ROWS["bad-json"])
        with pytest.raises(StateProtectedError):
            await store.load(CHAT_KEY)

        assert plugin_mod._is_sleeping(CHAT_KEY) is False

    async def test_prewarm_skips_protected_rows(self, fake_backend, captured_logs):
        _put(fake_backend, CHAT_KEY, BAD_ROWS["bad-json"])
        good = ChatSleepState(chat_key="other")
        _put(fake_backend, "other", good.model_dump_json())
        await fake_backend.set(
            store_key="chat_keys.v1", value=json.dumps([CHAT_KEY, "other"])
        )
        store = SleepStateStore(fake_backend)

        await store.initialize()

        assert store.is_protected(CHAT_KEY)
        assert store.known_chat_keys() == {"other"}
        assert _raw(fake_backend, CHAT_KEY) == BAD_ROWS["bad-json"]
        errors = [r for r in captured_logs.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
