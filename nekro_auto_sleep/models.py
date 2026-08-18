"""Pydantic models for the nekro_auto_sleep plugin.

State, cycle, segments, and config snapshot — pure data, no host imports.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = 2
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
    wake_confirm_window_seconds: int
    history_mode: Literal["preserve", "strict"]
    call_keywords: list[str]
    fallback_persona_name: str
    early_wake_idle_minutes: int
    quality_min: int
    quality_max: int
    quality_jitter_points: float

    # Schema v2 additions. All defaulted so a v1 payload still validates and
    # migrates in place instead of resetting the night.
    near_wake_minutes: int = 60
    sleep_target_hours: float = 8.0
    urgent_keywords: list[str] = Field(default_factory=list)
    answer_scope: Literal["offeree", "anyone"] = "offeree"
    max_offers_per_night: int = 3
    offer_cooldown_minutes: int = 20
    snooze_minutes: int = 30
    asleep_prompt: str = "【{persona}已经睡了 要叫醒{persona}吗？】"
    near_wake_prompt: str = "【{persona}还没起床 要叫醒{persona}吗？】"

    # Deprecated in v2, kept so v1 payloads keep validating; the near-wake test
    # now uses `near_wake_minutes`.
    near_wake_ratio: float = 0.15


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
    # Every term that produced the reported percentage. Persisted so a score
    # that looks wrong can be explained instead of reverse-engineered.
    quality_breakdown: dict[str, float] | None = None

    @field_validator("sleep_at", "planned_wake_at", "settled_at", mode="before")
    @classmethod
    def _parse_utc(cls, v: object) -> object:
        if isinstance(v, str) and v.endswith("Z"):
            return v.replace("Z", "+00:00")
        return v


class PendingWakeOffer(BaseModel):
    """The wake-up question currently awaiting an answer in a chat.

    One per chat, not one per user: the bot asked out loud, and whoever is
    allowed to answer (see `answer_scope`) answers the same question.
    """

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
    pending_offer: PendingWakeOffer | None = None
    # Anti-spam: without these a chat that says the wake word every few minutes
    # got a fixed reply every time, and every one of them cost sleep quality.
    offers_sent_tonight: int = 0
    last_offer_at: datetime | None = None
    snooze_until: datetime | None = None
    idle_sleep_deadline: datetime | None = None
    cycle: SleepCycle | None = None
    # Wake provenance, used to render the sleep-status prompt injection for the
    # whole AWAKE_EARLY stretch instead of a single in-memory one-shot string.
    # Additive optional fields: v1 payloads still validate, so SCHEMA_VERSION
    # stays at 1 and a rollback keeps reading state instead of resetting it.
    woken_at: datetime | None = None
    woken_by: str | None = None
    woken_reason: str | None = None
    # True for the single agent round that decides whether the caller actually
    # wanted the bot up. The model answers by either replying (stay awake) or
    # calling resume_sleep (go back to sleep, silently). Cleared as soon as the
    # conversation continues.
    wake_decision_pending: bool = False

    @field_validator(
        "last_seen_at",
        "idle_sleep_deadline",
        "woken_at",
        "last_offer_at",
        "snooze_until",
        mode="before",
    )
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
