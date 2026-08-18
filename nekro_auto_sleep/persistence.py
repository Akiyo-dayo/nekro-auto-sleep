"""Persistence adapter for ChatSleepState via DBPluginData / PluginStore.

Provides per-chat_key async locking, JSON serialization, schema version checks,
and corruption handling. Domain layer uses this instead of importing host singletons.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol
from collections.abc import Callable, Coroutine

from .models import SCHEMA_VERSION, ChatSleepState, ScheduleOverride, SleepStatus

logger = logging.getLogger("nekro_auto_sleep.persistence")

DATA_KEY = "state.v1"


class StoreBackend(Protocol):
    """Minimal interface matching PluginStore.get/set/delete."""

    async def get(
        self, chat_key: str = "", user_key: str = "", store_key: str = ""
    ) -> str | None: ...

    async def set(
        self, chat_key: str = "", user_key: str = "", store_key: str = "", value: str = ""
    ) -> int: ...

    async def delete(
        self, chat_key: str = "", user_key: str = "", store_key: str = ""
    ) -> int: ...


CHANNEL_OVERRIDE_KEY = "schedule_override.v1"
PRESET_OVERRIDE_PREFIX = "preset_schedule.v1."


class ScheduleOverrideStore:
    """Per-channel and per-persona schedule overrides.

    Kept apart from the nightly state on purpose: overrides outlive cycles, and
    losing one because a night's state failed to validate would silently move a
    channel back to the global bedtime.
    """

    def __init__(self, backend: StoreBackend) -> None:
        self._backend = backend

    @staticmethod
    def _preset_key(preset_id: object) -> str:
        return f"{PRESET_OVERRIDE_PREFIX}{preset_id}"

    async def _load(self, chat_key: str, store_key: str) -> ScheduleOverride | None:
        raw = await self._backend.get(chat_key=chat_key, store_key=store_key)
        if not raw:
            return None
        try:
            return ScheduleOverride.model_validate_json(raw)
        except Exception as exc:
            logger.error("Corrupted schedule override at %s/%s: %s", chat_key, store_key, exc)
            return None

    async def _save(
        self, chat_key: str, store_key: str, override: ScheduleOverride
    ) -> None:
        if override.is_empty():
            await self._backend.delete(chat_key=chat_key, store_key=store_key)
            return
        await self._backend.set(
            chat_key=chat_key, store_key=store_key, value=override.model_dump_json()
        )

    async def get_channel(self, chat_key: str) -> ScheduleOverride | None:
        return await self._load(chat_key, CHANNEL_OVERRIDE_KEY)

    async def set_channel(self, chat_key: str, override: ScheduleOverride) -> None:
        await self._save(chat_key, CHANNEL_OVERRIDE_KEY, override)

    async def get_preset(self, preset_id: object) -> ScheduleOverride | None:
        if preset_id is None:
            return None
        return await self._load("", self._preset_key(preset_id))

    async def set_preset(self, preset_id: object, override: ScheduleOverride) -> None:
        if preset_id is None:
            return
        await self._save("", self._preset_key(preset_id), override)


class SleepStateStore:
    """Manages loading and saving ChatSleepState per chat_key with async locks."""

    def __init__(self, backend: StoreBackend) -> None:
        self._backend = backend
        self._locks: dict[str, asyncio.Lock] = {}
        self._cache: dict[str, ChatSleepState] = {}
        self._corrupted: dict[str, str] = {}

    def _get_lock(self, chat_key: str) -> asyncio.Lock:
        if chat_key not in self._locks:
            self._locks[chat_key] = asyncio.Lock()
        return self._locks[chat_key]

    async def load(self, chat_key: str) -> ChatSleepState | None:
        """Load state from DB. Returns None if no state exists.

        Raises ValueError for unknown higher schema versions.
        Isolates corrupted JSON in memory and returns None.
        """
        if chat_key in self._corrupted:
            return None

        if chat_key in self._cache:
            return self._cache[chat_key]

        raw = await self._backend.get(chat_key=chat_key, store_key=DATA_KEY)
        if raw is None:
            return None

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(
                "Corrupted JSON for chat_key=%s, isolating in memory: %s", chat_key, exc
            )
            self._corrupted[chat_key] = raw
            return None

        version = data.get("schema_version", 1)
        if version > SCHEMA_VERSION:
            logger.error(
                "Unknown schema version %d for chat_key=%s; skipping to avoid data loss",
                version,
                chat_key,
            )
            return None

        try:
            state = ChatSleepState.model_validate(data)
        except Exception as exc:
            logger.error(
                "Validation failed for chat_key=%s, using defaults: %s", chat_key, exc
            )
            state = ChatSleepState(chat_key=chat_key)

        self._cache[chat_key] = state
        return state

    async def save(self, state: ChatSleepState) -> None:
        """Persist state to DB. Must be called within the chat_key's lock.

        Stamps the current schema version on the way out. Pydantic keeps
        whatever version was in the payload it validated, so a row migrated
        from v1 kept advertising v1 forever — the guard that refuses to load a
        *newer* schema was reading a number that no longer described the row.
        """
        chat_key = state.chat_key
        if state.schema_version != SCHEMA_VERSION:
            state = state.model_copy(update={"schema_version": SCHEMA_VERSION})
        raw = state.model_dump_json(by_alias=True)
        await self._backend.set(chat_key=chat_key, store_key=DATA_KEY, value=raw)
        self._cache[chat_key] = state

    async def load_or_create(self, chat_key: str) -> ChatSleepState:
        """Load existing state or create a fresh AWAKE state."""
        state = await self.load(chat_key)
        if state is None:
            state = ChatSleepState(chat_key=chat_key)
        return state

    async def with_state(
        self,
        chat_key: str,
        fn: Callable[[ChatSleepState], Coroutine[Any, Any, ChatSleepState]],
    ) -> ChatSleepState:
        """Execute fn under the chat_key's lock, save if state changed.

        fn receives the current state and must return the (possibly modified) state.
        """
        lock = self._get_lock(chat_key)
        async with lock:
            state = await self.load_or_create(chat_key)
            new_state = await fn(state)
            await self.save(new_state)
            return new_state

    async def hydrate(self, chat_key: str) -> ChatSleepState:
        """Pull a chat_key into the cache so background maintenance can see it.

        `known_chat_keys()` only reports cached keys, and the cache used to be
        filled exclusively by inbound messages — so after a restart no channel
        went to bed until somebody talked in it, and channels that were already
        asleep never got their wake-up settled. Boot reconciliation calls this
        for every discovered channel, including ones with no stored state yet.
        """
        state = await self.load_or_create(chat_key)
        self._cache[chat_key] = state
        return state

    def get_cached(self, chat_key: str) -> ChatSleepState | None:
        """Return cached state without DB access. For read-only checks."""
        return self._cache.get(chat_key)

    def known_chat_keys(self) -> set[str]:
        """Return all chat_keys that have been loaded or cached."""
        return set(self._cache.keys())

    def invalidate_cache(self, chat_key: str) -> None:
        """Remove a chat_key from cache, forcing next load to read from DB."""
        self._cache.pop(chat_key, None)
        self._corrupted.pop(chat_key, None)

    def clear_all(self) -> None:
        """Clear all caches and locks."""
        self._cache.clear()
        self._corrupted.clear()
        self._locks.clear()
