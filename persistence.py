"""Persistence adapter for ChatSleepState via DBPluginData / PluginStore.

Provides per-chat_key async locking, JSON serialization, schema version checks,
and corruption handling. Domain layer uses this instead of importing host singletons.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from typing import Any, Protocol

from pydantic import ValidationError

from .models import READABLE_SCHEMA_VERSION, SCHEMA_VERSION, ChatSleepState

logger = logging.getLogger("nekro_auto_sleep.persistence")

DATA_KEY = "state.v1"
CHAT_KEYS_INDEX_KEY = "chat_keys.v1"


class StateProtectedError(RuntimeError):
    """Raised when a chat_key's persisted row must not be touched.

    The row's bytes are kept exactly as they are: the plugin behaves as if the
    channel had sleep disabled, and nothing writes to ``state.v1`` for it.
    """

    def __init__(self, chat_key: str, reason: str) -> None:
        super().__init__(f"{chat_key}: {reason}")
        self.chat_key = chat_key
        self.reason = reason


def migrate_v2_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Turn a p0-batch1 ``schema_version=2`` payload into something readable here.

    The only structural break is the wake offer: v2 keeps a single
    ``pending_offer`` for the whole chat, while this version keeps one offer per
    user. An outstanding question is still an outstanding question, so it moves
    into ``pending_wake_offers`` under its own user id. Fields the two branches
    share by name (``offers_sent_tonight``, ``last_offer_at``, ``woken_at``,
    ``woken_by``, ``skip_sleep_date``) are left alone; everything else that only
    p0 knows about is ignored by pydantic.
    """
    pending = data.get("pending_offer")
    if isinstance(pending, dict):
        existing = data.get("pending_wake_offers") or {}
        if not existing:
            user_id = pending.get("user_id")
            if isinstance(user_id, str) and user_id:
                data["pending_wake_offers"] = {user_id: pending}
    data.pop("pending_offer", None)
    return data


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
        # chat_key -> the untouched raw row. A row lands here when it cannot be
        # read; nothing in this process may write over it afterwards.
        self._protected: dict[str, str] = {}
        self._known_chat_keys: set[str] = set()
        self._index_lock = asyncio.Lock()

    def _get_lock(self, chat_key: str) -> asyncio.Lock:
        if chat_key not in self._locks:
            self._locks[chat_key] = asyncio.Lock()
        return self._locks[chat_key]

    def is_protected(self, chat_key: str) -> bool:
        """Whether this chat_key's row failed to load and must not be touched."""
        return chat_key in self._protected

    def protected_chat_keys(self) -> set[str]:
        """chat_keys currently in the protected (read-only, ignored) state."""
        return set(self._protected)

    def _protect(self, chat_key: str, reason: str) -> None:
        """Remember a chat_key whose row could not be read, and say so once.

        The row itself is deliberately never rewritten: keeping unreadable data
        beats replacing it with a fresh empty state. One log line per channel
        per process, because the message hook runs on every message.
        """
        if chat_key in self._protected:
            return
        self._protected[chat_key] = reason
        logger.error(
            "无法读取 chat_key=%s 的睡眠状态（%s）。数据已原样保留，未覆盖；"
            "本频道按未启用睡眠处理，直到下次重启。",
            chat_key,
            reason,
        )

    async def initialize(
        self,
        discover_legacy_chat_keys: Callable[[], Awaitable[Iterable[str]]] | None = None,
    ) -> None:
        """Restore the persistent chat index and prewarm all known states.

        Older plugin versions did not write an index. In that case an optional
        host-specific discovery callback may enumerate legacy ``state.v1`` rows.
        Discovery failures are isolated so plugin startup remains fail-open.
        """
        raw_index = await self._backend.get(store_key=CHAT_KEYS_INDEX_KEY)
        keys: set[str] = set()
        index_valid = raw_index is not None

        if raw_index is not None:
            try:
                parsed = json.loads(raw_index)
            except json.JSONDecodeError as exc:
                logger.error(
                    "Invalid persistent chat key index; rebuilding if possible: %s",
                    exc,
                )
                index_valid = False
            else:
                if isinstance(parsed, list):
                    keys = {
                        item for item in parsed if isinstance(item, str) and item
                    }
                else:
                    logger.error(
                        "Invalid persistent chat key index; rebuilding if possible: "
                        "expected a JSON list"
                    )
                    index_valid = False

        if not index_valid and discover_legacy_chat_keys is not None:
            try:
                discovered = await discover_legacy_chat_keys()
                keys.update(key for key in discovered if isinstance(key, str) and key)
            except Exception as exc:  # noqa: BLE001 - host ORM errors vary by fork
                logger.warning(
                    "Legacy chat key discovery failed; continuing without migration: %s",
                    exc,
                )

        self._known_chat_keys = set(keys)
        if not index_valid and keys:
            await self._persist_index()

        for chat_key in sorted(keys):
            try:
                await self.load(chat_key)
            except StateProtectedError:
                # Already logged once by ``_protect``; prewarming continues so a
                # single unreadable row cannot block the rest of the start-up.
                continue

    async def _persist_index(self) -> None:
        raw = json.dumps(
            sorted(self._known_chat_keys),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        await self._backend.set(store_key=CHAT_KEYS_INDEX_KEY, value=raw)

    async def _remember_chat_key(self, chat_key: str) -> None:
        if not chat_key or chat_key in self._known_chat_keys:
            return
        async with self._index_lock:
            if chat_key in self._known_chat_keys:
                return
            updated = set(self._known_chat_keys)
            updated.add(chat_key)
            raw = json.dumps(
                sorted(updated),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            await self._backend.set(store_key=CHAT_KEYS_INDEX_KEY, value=raw)
            self._known_chat_keys = updated

    async def load(self, chat_key: str) -> ChatSleepState | None:
        """Load state from DB. Returns None if no state exists.

        Three outcomes besides success:

        * missing row -> ``None`` (a fresh state may be created);
        * ``schema_version`` 2 (p0-batch1) -> migrated once, then validated;
        * unreadable (bad JSON, a newer schema, a payload that still fails
          validation) -> the chat_key becomes *protected*: the raw bytes stay in
          the DB untouched and the plugin ignores this channel like a disabled
          one, instead of replacing the row with an empty state.

        Raises ``StateProtectedError`` for a protected chat_key.
        """
        if chat_key in self._protected:
            raise StateProtectedError(chat_key, self._protected[chat_key])

        if chat_key in self._cache:
            return self._cache[chat_key]

        raw = await self._backend.get(chat_key=chat_key, store_key=DATA_KEY)
        if raw is None:
            return None

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            self._protect(chat_key, f"JSON 解析失败：{exc}")
            raise StateProtectedError(chat_key, self._protected[chat_key]) from exc

        if not isinstance(data, dict):
            self._protect(chat_key, "JSON 顶层不是对象")
            raise StateProtectedError(chat_key, self._protected[chat_key])

        version = data.get("schema_version", 1)
        if isinstance(version, bool) or not isinstance(version, int):
            self._protect(chat_key, f"schema_version 不是整数：{version!r}")
            raise StateProtectedError(chat_key, self._protected[chat_key])

        if version > READABLE_SCHEMA_VERSION:
            self._protect(chat_key, f"数据版本 {version} 高于本插件支持的 {READABLE_SCHEMA_VERSION}")
            raise StateProtectedError(chat_key, self._protected[chat_key])

        if version == 2:
            data = migrate_v2_payload(data)
            logger.info("已将 chat_key=%s 的 v2 数据迁移为当前格式", chat_key)

        try:
            state = ChatSleepState.model_validate(data)
        except ValidationError as exc:
            self._protect(chat_key, f"数据校验失败：{exc}")
            raise StateProtectedError(chat_key, self._protected[chat_key]) from exc

        # Whatever version the row claimed, what we hold now is a v1 shape.
        if state.schema_version != SCHEMA_VERSION:
            state = state.model_copy(update={"schema_version": SCHEMA_VERSION})

        self._cache[chat_key] = state
        await self._remember_chat_key(chat_key)
        return state

    async def save(self, state: ChatSleepState) -> None:
        """Persist state to DB. Must be called within the chat_key's lock.

        Always stamps the current schema version on the way out: a row migrated
        from the p0-batch1 v2 format (or from an older build) must not keep
        advertising a version it no longer has. A protected chat_key is never
        written, not even by a caller that rebuilt a state object by hand.
        """
        chat_key = state.chat_key
        if chat_key in self._protected:
            raise StateProtectedError(chat_key, self._protected[chat_key])

        if state.schema_version != SCHEMA_VERSION:
            state = state.model_copy(update={"schema_version": SCHEMA_VERSION})
        raw = state.model_dump_json(by_alias=True)
        await self._remember_chat_key(chat_key)
        await self._backend.set(chat_key=chat_key, store_key=DATA_KEY, value=raw)
        self._cache[chat_key] = state

    async def load_or_create(self, chat_key: str) -> ChatSleepState:
        """Load existing state or create a fresh AWAKE state."""
        state = await self.load(chat_key)
        if state is None:
            state = ChatSleepState(chat_key=chat_key)
        return state

    async def ensure_loaded(self, chat_key: str) -> ChatSleepState:
        """Load or create the state for chat_key under its lock.

        Public read-through for callers outside ``with_state`` that still need
        the state materialized (e.g. the sleep-report sandbox tool).
        """
        async with self._get_lock(chat_key):
            state = await self.load(chat_key)
            if state is None:
                state = ChatSleepState(chat_key=chat_key)
            self._cache[chat_key] = state
            return state

    async def with_state(
        self,
        chat_key: str,
        fn: Callable[[ChatSleepState], Coroutine[Any, Any, ChatSleepState]],
    ) -> ChatSleepState:
        """Execute fn under the chat_key's lock, save if state changed.

        fn receives the current state and must return the (possibly modified) state.
        A protected chat_key never reaches ``fn`` and is never saved.
        """
        lock = self._get_lock(chat_key)
        async with lock:
            state = await self.load_or_create(chat_key)
            old_state = state.model_copy(deep=True)
            new_state = await fn(state)
            if chat_key in self._protected:
                return new_state
            self._cache[chat_key] = new_state
            if new_state != old_state:
                await self.save(new_state)
            return new_state

    def get_cached(self, chat_key: str) -> ChatSleepState | None:
        """Return cached state without DB access. For read-only checks."""
        return self._cache.get(chat_key)

    def known_chat_keys(self) -> set[str]:
        """Return all chat_keys restored from or written to the persistent index."""
        return set(self._known_chat_keys) - set(self._protected)

    def invalidate_cache(self, chat_key: str) -> None:
        """Remove a chat_key from cache, forcing next load to read from DB."""
        self._cache.pop(chat_key, None)
        self._protected.pop(chat_key, None)

    def clear_all(self) -> None:
        """Clear all caches and locks."""
        self._cache.clear()
        self._protected.clear()
        self._locks.clear()
        self._known_chat_keys.clear()
