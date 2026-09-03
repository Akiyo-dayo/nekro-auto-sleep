"""Shared fixtures for nekro_auto_sleep tests.

Domain-layer tests import submodules directly (models, schedule, engine, etc.).
The repository root *is* the package body, so register it under the package
name by loading each submodule file explicitly — ``__init__.py`` itself is
never executed because it needs a running NekroAgent host.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from zoneinfo import ZoneInfo

_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SUBMODULES = ("models", "schedule", "quality", "engine", "persistence", "runtime")


def _register_alias_package() -> None:
    """Expose the repo root as ``nekro_auto_sleep`` without running __init__."""
    if "nekro_auto_sleep" in sys.modules:
        return

    pkg_spec = importlib.util.spec_from_file_location(
        "nekro_auto_sleep",
        _PKG_ROOT / "__init__.py",
        submodule_search_locations=[str(_PKG_ROOT)],
    )
    pkg = importlib.util.module_from_spec(pkg_spec)
    pkg.__path__ = [str(_PKG_ROOT)]
    sys.modules["nekro_auto_sleep"] = pkg

    for name in _SUBMODULES:
        mod_spec = importlib.util.spec_from_file_location(
            f"nekro_auto_sleep.{name}", _PKG_ROOT / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(mod_spec)
        sys.modules[f"nekro_auto_sleep.{name}"] = mod
        mod_spec.loader.exec_module(mod)


_register_alias_package()

from nekro_auto_sleep.models import ChatSleepState, ConfigSnapshot, SleepStatus
from nekro_auto_sleep.persistence import SleepStateStore, StoreBackend
from nekro_auto_sleep.schedule import create_config_snapshot


class FakeStoreBackend:
    """In-memory PluginStore backend for testing."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def _key(self, chat_key: str, user_key: str, store_key: str) -> str:
        return f"{chat_key}|{user_key}|{store_key}"

    async def get(
        self, chat_key: str = "", user_key: str = "", store_key: str = ""
    ) -> str | None:
        return self._data.get(self._key(chat_key, user_key, store_key))

    async def set(
        self, chat_key: str = "", user_key: str = "", store_key: str = "", value: str = ""
    ) -> int:
        k = self._key(chat_key, user_key, store_key)
        existed = k in self._data
        self._data[k] = value
        return 1 if existed else 0

    async def delete(
        self, chat_key: str = "", user_key: str = "", store_key: str = ""
    ) -> int:
        k = self._key(chat_key, user_key, store_key)
        if k in self._data:
            del self._data[k]
            return 0
        return 1


@pytest.fixture
def fake_backend() -> FakeStoreBackend:
    return FakeStoreBackend()


@pytest.fixture
def store(fake_backend: FakeStoreBackend) -> SleepStateStore:
    return SleepStateStore(fake_backend)


@pytest.fixture
def default_snapshot() -> ConfigSnapshot:
    return create_config_snapshot(
        timezone="Asia/Shanghai",
        sleep_time="23:00",
        wake_time_start="06:45",
        wake_time_end="08:30",
        wake_random_step_minutes=1,
        near_wake_ratio=0.15,
        wake_confirm_window_seconds=180,
        history_mode="preserve",
        call_keywords="醒醒,起床,在吗",
        fallback_persona_name="Bot",
        early_wake_idle_minutes=10,
        quality_min=60,
        quality_max=120,
        quality_jitter_points=4.0,
    )


CHAT_KEY = "onebot_v11-group_123456789"
TZ = ZoneInfo("Asia/Shanghai")
UTC = ZoneInfo("UTC")
