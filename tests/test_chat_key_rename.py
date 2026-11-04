"""A row whose key the host renamed must keep being read and written under the new key.

NekroAgent forks with multi-instance adapters upgrade chat_keys in place, for example
``onebot_v11-group_X`` to ``onebot_v11-ob10001-group_X``. They rewrite
``plugin_data.target_chat_key``, but the JSON payload still carries the old
``chat_key``. Before the fix, ``save()`` wrote by the payload's key: the new key's
cache stayed AWAKE, every message "fell asleep" again and fired the bedtime
greeting, and a second row piled up under the old key.
"""

from __future__ import annotations

import json
from datetime import datetime

from zoneinfo import ZoneInfo

from conftest import FakeStoreBackend
from nekro_auto_sleep.models import ChatSleepState, SleepStatus
from nekro_auto_sleep.persistence import SleepStateStore

OLD_KEY = "onebot_v11-group_1062716968"
NEW_KEY = "onebot_v11-ob3208809890-group_1062716968"
UTC = ZoneInfo("UTC")


def _row(backend: FakeStoreBackend, chat_key: str) -> str | None:
    return backend._data.get(f"{chat_key}||state.v1")


def _renamed_row(backend: FakeStoreBackend) -> None:
    """What the host leaves behind: new row key, old key inside the payload."""
    stale = ChatSleepState(chat_key=OLD_KEY, last_seen_at=datetime(2026, 11, 2, 8, 47, tzinfo=UTC))
    backend._data[f"{NEW_KEY}||state.v1"] = stale.model_dump_json(by_alias=True)


async def test_load_uses_the_row_key(fake_backend):
    _renamed_row(fake_backend)
    store = SleepStateStore(fake_backend)

    state = await store.load(NEW_KEY)

    assert state is not None
    assert state.chat_key == NEW_KEY


async def test_with_state_writes_back_under_the_row_key(fake_backend):
    _renamed_row(fake_backend)
    store = SleepStateStore(fake_backend)

    async def _fall_asleep(s: ChatSleepState) -> ChatSleepState:
        return s.model_copy(update={"status": SleepStatus.ASLEEP})

    await store.with_state(NEW_KEY, _fall_asleep)

    # No second row under the stale key.
    assert _row(fake_backend, OLD_KEY) is None
    written = json.loads(_row(fake_backend, NEW_KEY))
    assert written["chat_key"] == NEW_KEY
    assert written["status"] == SleepStatus.ASLEEP.value
    # The next message sees the chat asleep, so it does not "fall asleep" again.
    assert store.get_cached(NEW_KEY).status == SleepStatus.ASLEEP
    assert (await SleepStateStore(fake_backend).load(NEW_KEY)).status == SleepStatus.ASLEEP


async def test_with_state_ignores_a_foreign_key_returned_by_fn(fake_backend):
    store = SleepStateStore(fake_backend)

    async def _foreign(s: ChatSleepState) -> ChatSleepState:
        return s.model_copy(update={"chat_key": OLD_KEY, "status": SleepStatus.ASLEEP})

    await store.with_state(NEW_KEY, _foreign)

    assert _row(fake_backend, OLD_KEY) is None
    assert json.loads(_row(fake_backend, NEW_KEY))["chat_key"] == NEW_KEY
