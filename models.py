"""Pydantic models for the nekro_auto_sleep plugin.

State, cycle, segments, and config snapshot — pure data, no host imports.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = 1
PLUGIN_KEY = "Akiyo_dayo.nekro_auto_sleep"
DATA_KEY = "state.v1"


class SleepStatus(str, enum.Enum):
    AWAKE = "AWAKE"
    ASLEEP = "ASLEEP"
    AWAKE_EARLY = "AWAKE_EARLY"


class SourceType(str, enum.Enum):
    USER_WAKE_CONFIRM = "USER_WAKE_CONFIRM"
    USER_DIRECT = "USER_DIRECT"
    TIMER_ONESHOT = "TIMER_ONESHOT"
    TIMER_RECURRING = "TIMER_RECURRING"
    INTERNAL_WAKE_NOTICE = "INTERNAL_WAKE_NOTICE"
    UNTRUSTED = "UNTRUSTED"


class NotificationStatus(str, enum.Enum):
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"


class SleepSegment(BaseModel):
    """A continuous sleep interval. open_at is always set; close_at is None while sleeping."""

    open_at: datetime
    close_at: datetime | None = None

    @field_validator("open_at", "close_at", mode="before")
    @classmethod
    def _parse_utc(cls, v: object) -> object:
        if isinstance(v, str) and v.endswith("Z"):
            return v.replace("Z", "+00:00")
        return v


class WakeAttempt(BaseModel):
    """Record of a user's wake-up call attempt."""

    user_id: str
    chat_key: str
    attempted_at: datetime
    is_confirmed: bool = False
    confirmed_at: datetime | None = None

    @field_validator("attempted_at", "confirmed_at", mode="before")
    @classmethod
    def _parse_utc(cls, v: object) -> object:
        if isinstance(v, str) and v.endswith("Z"):
            return v.replace("Z", "+00:00")
        return v


class TimerInterval(BaseModel):
    """Duration of a timer-task execution during sleep."""

    task_id: str
    start_at: datetime
    end_at: datetime | None = None
    source_type: SourceType = SourceType.TIMER_ONESHOT

    @field_validator("start_at", "end_at", mode="before")
    @classmethod
    def _parse_utc(cls, v: object) -> object:
        if isinstance(v, str) and v.endswith("Z"):
            return v.replace("Z", "+00:00")
        return v


class ConfigSnapshot(BaseModel):
    """Snapshot of sleep-relevant config at cycle creation time.

    Changes mid-cycle only affect the next cycle.
    """

    timezone: str
    sleep_time: str
    wake_time_start: str
    wake_time_end: str
    wake_random_step_minutes: int
    near_wake_ratio: float
    wake_confirm_window_seconds: int
    history_mode: Literal["preserve", "strict"]
    call_keywords: list[str]
    fallback_persona_name: str
    early_wake_idle_minutes: int
    quality_min: int
    quality_max: int
    quality_jitter_points: float


class SleepCycle(BaseModel):
    """One sleep cycle for a chat_key, from bedtime to wake-up settlement."""

    cycle_id: str
    sleep_date: str = Field(description="Local date YYYY-MM-DD when sleep started")
    timezone: str
    sleep_at: datetime
    planned_wake_at: datetime
    config_snapshot: ConfigSnapshot
    quality_seed: str
    sleep_segments: list[SleepSegment] = Field(default_factory=list)
    wake_attempts: list[WakeAttempt] = Field(default_factory=list)
    timer_intervals: list[TimerInterval] = Field(default_factory=list)
    notification_status: NotificationStatus = NotificationStatus.PENDING
    settled_at: datetime | None = None
    ended_while_early_awake: bool = False

    @field_validator("sleep_at", "planned_wake_at", "settled_at", mode="before")
    @classmethod
    def _parse_utc(cls, v: object) -> object:
        if isinstance(v, str) and v.endswith("Z"):
            return v.replace("Z", "+00:00")
        return v


class PendingWakeOffer(BaseModel):
    """A pending first-call wake offer for a specific user in a chat."""

    user_id: str
    offered_at: datetime
    expires_at: datetime

    @field_validator("offered_at", "expires_at", mode="before")
    @classmethod
    def _parse_utc(cls, v: object) -> object:
        if isinstance(v, str) and v.endswith("Z"):
            return v.replace("Z", "+00:00")
        return v


class ChatSleepState(BaseModel):
    """Top-level persisted state for one chat_key."""

    schema_version: int = SCHEMA_VERSION
    chat_key: str
    status: SleepStatus = SleepStatus.AWAKE
    last_seen_at: datetime | None = None
    pending_wake_offers: dict[str, PendingWakeOffer] = Field(default_factory=dict)
    idle_sleep_deadline: datetime | None = None
    cycle: SleepCycle | None = None

    @field_validator("last_seen_at", "idle_sleep_deadline", mode="before")
    @classmethod
    def _parse_utc(cls, v: object) -> object:
        if isinstance(v, str) and v.endswith("Z"):
            return v.replace("Z", "+00:00")
        return v

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, v: int) -> int:
        if v > SCHEMA_VERSION:
            raise ValueError(
                f"Unknown schema version {v} > {SCHEMA_VERSION}; "
                "refusing to load to avoid data corruption"
            )
        return v
