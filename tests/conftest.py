"""Shared fixtures for nekro_auto_sleep tests.

Domain-layer tests import submodules directly (models, schedule, engine, etc.)
without going through __init__.py, which depends on the NekroAgent host.
The root conftest.py registers a stub package to avoid triggering __init__.py.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from zoneinfo import ZoneInfo

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
        quality_min=20,
        quality_max=120,
        quality_jitter_points=2.0,
        near_wake_minutes=60,
        sleep_target_hours=0.0,
        urgent_keywords="紧急,急事,救命,出事了",
        answer_scope="offeree",
        max_offers_per_night=3,
        offer_cooldown_minutes=20,
        snooze_minutes=30,
    )


CHAT_KEY = "onebot_v11-group_123456789"
TZ = ZoneInfo("Asia/Shanghai")
UTC = ZoneInfo("UTC")
