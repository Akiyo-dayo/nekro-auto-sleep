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

from .models import SCHEMA_VERSION, ChatSleepState, SleepStatus

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
        """Persist state to DB. Must be called within the chat_key's lock."""
        chat_key = state.chat_key
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
