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
from types import ModuleType
from unittest.mock import AsyncMock

import pytest
from zoneinfo import ZoneInfo

_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SUBMODULES = ("models", "schedule", "quality", "engine", "persistence", "runtime")


class _BlockRealHost:
    """Refuse to import a real NekroAgent checkout from inside the test run.

    Importing the real host creates its ``data/`` tree in the working copy and
    loads production config, neither of which a test may do. Every host-facing
    test installs its own stub tree in ``sys.modules`` first, so this finder
    only ever fires on an accidental import.
    """

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        if fullname == "nekro_agent" or fullname.startswith("nekro_agent."):
            raise ImportError(
                f"{fullname} must not be imported for real during tests; "
                "use tests/hoststub.py instead"
            )
        return None


if not any(isinstance(f, _BlockRealHost) for f in sys.meta_path):
    sys.meta_path.insert(0, _BlockRealHost())


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


def _load_hoststub() -> ModuleType:
    """Import ``tests/hoststub.py`` without depending on how pytest set sys.path."""
    existing = sys.modules.get("hoststub")
    if isinstance(existing, ModuleType) and hasattr(existing, "install"):
        return existing
    spec = importlib.util.spec_from_file_location(
        "hoststub", pathlib.Path(__file__).resolve().parent / "hoststub.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["hoststub"] = module
    spec.loader.exec_module(module)
    return module


hoststub = _load_hoststub()


def load_plugin_module():
    """Exec the real ``__init__.py`` against the stub host.

    The stub tree is installed for the duration of the call only: the module
    keeps its imported objects, and every other suite gets a pristine
    ``sys.modules``. Re-executing per test keeps each case on fresh globals.

    Tests that exercise lazy host imports (the instance-config path, the own-bot
    account lookup) must use the ``plugin_mod`` fixture instead, which keeps the
    stub installed for the whole test.
    """
    hoststub.set_channel_factory(None)
    hoststub.set_ctx_factory(None)
    restore = hoststub.install()
    try:
        nas_mod = sys.modules["nekro_auto_sleep"]
        spec = importlib.util.spec_from_file_location(
            "nekro_auto_sleep",
            _PKG_ROOT / "__init__.py",
            submodule_search_locations=[str(_PKG_ROOT)],
        )
        spec.loader.exec_module(nas_mod)
    finally:
        restore()
    return nas_mod


@pytest.fixture
def host_stub():
    """Install the stub host for the whole test and restore it afterwards."""
    hoststub.set_channel_factory(None)
    hoststub.set_ctx_factory(None)
    hoststub.clear_instance_config_overrides()
    hoststub.set_own_bot_accounts([])
    restore = hoststub.install()
    try:
        yield hoststub
    finally:
        hoststub.set_channel_factory(None)
        hoststub.set_ctx_factory(None)
        hoststub.clear_instance_config_overrides()
        hoststub.set_own_bot_accounts([])
        restore()


@pytest.fixture
def plugin_mod(host_stub):
    """The plugin wiring module loaded against a live stub host."""
    nas_mod = sys.modules["nekro_auto_sleep"]
    spec = importlib.util.spec_from_file_location(
        "nekro_auto_sleep",
        _PKG_ROOT / "__init__.py",
        submodule_search_locations=[str(_PKG_ROOT)],
    )
    spec.loader.exec_module(nas_mod)
    return nas_mod


@pytest.fixture
def wired_store(plugin_mod):
    """Point the freshly loaded plugin at an in-memory store."""
    backend = FakeStoreBackend()
    plugin_mod._store = SleepStateStore(backend)
    return plugin_mod._store


class FakeStoreBackend:
    """In-memory PluginStore backend for testing."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self.set_calls = 0

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
        self.set_calls += 1
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
