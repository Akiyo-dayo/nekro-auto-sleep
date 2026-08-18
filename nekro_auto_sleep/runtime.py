"""Reversible wrapping of host callables, and the source contextvar.

Only what is actually load-bearing lives here. The first version also carried a
TTL lease ledger and a per-chat lock table, built so that a monkey-patched timer
service could hand a night-time agent round a permission ticket. None of it ever
ran: the install step registered the timer methods for later restoration without
ever wrapping them, so `lease_ledger.create` had no call sites anywhere in the
tree and every night-time scheduled task was silently dropped instead of being
allowed through. `NIGHT_TIMER_POLICY` replaced that whole mechanism with a
decision made in `on_system_message`, which needs no patching at all.

What remains: `wrap_callable` / `unwrap_callable`, used for the single optional
gate on `message_service.schedule_agent_task` (a public method), and the source
contextvar used to mark the plugin's own outbound wake-up notice.
"""

from __future__ import annotations

import contextvars
import logging
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


def is_wrapped(target_obj: Any, attr_name: str) -> bool:
    """Whether attr_name currently holds our wrapper (used by tests)."""
    return bool(getattr(getattr(target_obj, attr_name, None), _WRAP_MARKER, False))


# ---------------------------------------------------------------------------
# The one remaining gate
# ---------------------------------------------------------------------------


def make_schedule_agent_task_wrapper(
    should_block_fn: Callable[[str], bool],
) -> Callable[..., Any]:
    """Wrap `message_service.schedule_agent_task`.

    Inbound user messages never reach this while the chat is asleep — the
    `on_user_message` hook has already returned a blocking signal by then — so
    in practice this only sees callers that schedule a round directly, which
    means the timer service. It therefore does nothing unless the operator asked
    for night-time scheduled tasks to be blocked.
    """

    async def wrapper(original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        chat_key = kwargs.get("chat_key")
        if chat_key is None and args:
            chat_key = args[0]

        if isinstance(chat_key, str) and should_block_fn(chat_key):
            logger.info("Night timer policy blocked an agent round for %s", chat_key)
            return None

        return await original(*args, **kwargs)

    return wrapper
