"""Source ledger, leases, and reversible runtime wrapping.

Wraps host callables (schedule_agent_task, _run_chat_agent_task, timer entry points)
to enforce the sleep gate at dispatch and execution layers.

Uses contextvars for synchronous call chains and TTL-based message/task lease
ledger for cross-task permission propagation (spec §7.3).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

from .models import SourceType

logger = logging.getLogger("nekro_auto_sleep.runtime")

# ---------------------------------------------------------------------------
# Plugin-private marker for wrapped callables
# ---------------------------------------------------------------------------

_WRAP_MARKER = "__nekro_auto_sleep_wrapped__"
_WRAP_ORIGINAL = "__nekro_auto_sleep_original__"

# ---------------------------------------------------------------------------
# Context variable for synchronous source propagation
# ---------------------------------------------------------------------------

current_source: contextvars.ContextVar[SourceType | None] = contextvars.ContextVar(
    "nekro_auto_sleep_source", default=None
)


# ---------------------------------------------------------------------------
# TTL-based lease ledger for cross-task permission
# ---------------------------------------------------------------------------


@dataclass
class Lease:
    source_type: SourceType
    chat_key: str
    task_id: str
    created_at: float
    ttl: float
    future: asyncio.Future[Any] | None = None
    claimed: bool = False


class LeaseLedger:
    """Cross-task permission ledger with TTL expiry."""

    def __init__(self) -> None:
        self._leases: dict[str, Lease] = {}
        self._by_chat_key: dict[str, set[str]] = {}

    def create(
        self,
        lease_id: str,
        source_type: SourceType,
        chat_key: str,
        task_id: str,
        ttl: float,
        future: asyncio.Future[Any] | None = None,
    ) -> Lease:
        lease = Lease(
            source_type=source_type,
            chat_key=chat_key,
            task_id=task_id,
            created_at=time.monotonic(),
            ttl=ttl,
            future=future,
        )
        self._leases[lease_id] = lease
        self._by_chat_key.setdefault(chat_key, set()).add(lease_id)
        return lease

    def get(self, lease_id: str) -> Lease | None:
        lease = self._leases.get(lease_id)
        if lease is None:
            return None
        if time.monotonic() - lease.created_at > lease.ttl:
            self.remove(lease_id)
            return None
        return lease

    def get_active_for_chat(self, chat_key: str) -> list[Lease]:
        ids = self._by_chat_key.get(chat_key, set())
        active = []
        expired = []
        for lid in ids:
            lease = self._leases.get(lid)
            if lease is None:
                expired.append(lid)
                continue
            if time.monotonic() - lease.created_at > lease.ttl:
                expired.append(lid)
                continue
            active.append(lease)
        for lid in expired:
            self.remove(lid)
        return active

    def has_active_for_chat(self, chat_key: str) -> bool:
        return len(self.get_active_for_chat(chat_key)) > 0

    def claim(self, lease_id: str) -> Lease | None:
        lease = self.get(lease_id)
        if lease is not None:
            lease.claimed = True
        return lease

    def remove(self, lease_id: str) -> Lease | None:
        lease = self._leases.pop(lease_id, None)
        if lease is not None:
            chat_ids = self._by_chat_key.get(lease.chat_key)
            if chat_ids is not None:
                chat_ids.discard(lease_id)
                if not chat_ids:
                    del self._by_chat_key[lease.chat_key]
        return lease

    def clear(self) -> None:
        for lease in self._leases.values():
            if lease.future and not lease.future.done():
                lease.future.cancel()
        self._leases.clear()
        self._by_chat_key.clear()


# Global ledger instance
lease_ledger = LeaseLedger()

# ---------------------------------------------------------------------------
# Reversible wrapping utilities
# ---------------------------------------------------------------------------


def wrap_callable(
    target_obj: Any,
    attr_name: str,
    wrapper_fn: Callable[..., Any],
) -> bool:
    """Wrap target_obj.attr_name with wrapper_fn, idempotently.

    Returns True if wrapping was applied, False if already wrapped by us.
    Stores the original callable for restoration.
    """
    original = getattr(target_obj, attr_name, None)
    if original is None:
        logger.error("Cannot wrap %s.%s: attribute not found", type(target_obj).__name__, attr_name)
        return False

    if not callable(original):
        logger.error("Cannot wrap %s.%s: not callable", type(target_obj).__name__, attr_name)
        return False

    if getattr(original, _WRAP_MARKER, False):
        logger.debug("Already wrapped %s.%s, skipping", type(target_obj).__name__, attr_name)
        return False

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        return await wrapper_fn(original, *args, **kwargs)

    setattr(wrapped, _WRAP_MARKER, True)
    setattr(wrapped, _WRAP_ORIGINAL, original)
    setattr(target_obj, attr_name, wrapped)
    logger.info("Wrapped %s.%s", type(target_obj).__name__, attr_name)
    return True


def unwrap_callable(target_obj: Any, attr_name: str) -> bool:
    """Restore original callable if current one is our wrapper.

    Returns True if restoration was done, False if not our wrapper.
    """
    current = getattr(target_obj, attr_name, None)
    if current is None:
        return False

    if not getattr(current, _WRAP_MARKER, False):
        logger.debug(
            "Not unwrapping %s.%s: current callable is not our wrapper",
            type(target_obj).__name__,
            attr_name,
        )
        return False

    original = getattr(current, _WRAP_ORIGINAL, None)
    if original is None:
        logger.warning(
            "Cannot unwrap %s.%s: original callable not found on wrapper",
            type(target_obj).__name__,
            attr_name,
        )
        return False

    setattr(target_obj, attr_name, original)
    logger.info("Unwrapped %s.%s", type(target_obj).__name__, attr_name)
    return True


# ---------------------------------------------------------------------------
# Wrapper factories (to be connected in __init__.py)
# ---------------------------------------------------------------------------


def make_schedule_agent_task_wrapper(
    is_sleeping_fn: Callable[[str], bool],
    has_permission_fn: Callable[[str], bool],
) -> Callable[..., Any]:
    """Create a wrapper for message_service.schedule_agent_task.

    When sleeping and no trusted permission exists, silently blocks the call.
    """

    async def wrapper(original: Callable[..., Any], chat_key: str, *args: Any, **kwargs: Any) -> Any:
        src = current_source.get()

        if src in (SourceType.USER_WAKE_CONFIRM, SourceType.USER_DIRECT,
                   SourceType.TIMER_ONESHOT, SourceType.TIMER_RECURRING,
                   SourceType.INTERNAL_WAKE_NOTICE):
            return await original(chat_key, *args, **kwargs)

        if is_sleeping_fn(chat_key) and not has_permission_fn(chat_key):
            logger.debug("Blocked schedule_agent_task for sleeping chat_key=%s", chat_key)
            return None

        return await original(chat_key, *args, **kwargs)

    return wrapper


def make_run_agent_task_wrapper(
    is_sleeping_fn: Callable[[str], bool],
    on_agent_start_fn: Callable[[str], Any],
    on_agent_end_fn: Callable[[str], Any],
) -> Callable[..., Any]:
    """Create a wrapper for message_service._run_chat_agent_task.

    Re-checks sleep state before execution (spec §7.2 layer 3).
    """

    async def wrapper(original: Callable[..., Any], chat_key: str, *args: Any, **kwargs: Any) -> Any:
        src = current_source.get()

        if src not in (SourceType.USER_WAKE_CONFIRM, SourceType.USER_DIRECT,
                       SourceType.TIMER_ONESHOT, SourceType.TIMER_RECURRING,
                       SourceType.INTERNAL_WAKE_NOTICE):
            if is_sleeping_fn(chat_key):
                logger.debug("Blocked _run_chat_agent_task for sleeping chat_key=%s", chat_key)
                return None

        await on_agent_start_fn(chat_key)
        try:
            result = await original(chat_key, *args, **kwargs)
            return result
        finally:
            await on_agent_end_fn(chat_key)

    return wrapper


# ---------------------------------------------------------------------------
# Chat-key serial locks for timer tasks
# ---------------------------------------------------------------------------


class ChatKeyLocks:
    """Per-chat_key asyncio locks for serializing timer task execution."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, chat_key: str) -> asyncio.Lock:
        if chat_key not in self._locks:
            self._locks[chat_key] = asyncio.Lock()
        return self._locks[chat_key]

    def clear(self) -> None:
        self._locks.clear()


chat_key_locks = ChatKeyLocks()
