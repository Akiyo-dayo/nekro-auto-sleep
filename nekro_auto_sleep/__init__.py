"""NekroAgent Auto-Sleep Plugin

Provides per-chat_key humanized sleep cycles without modifying NekroAgent core.
Compatible with both Akiyo and upstream versions via runtime capability probing.

Plugin key: Akiyo_dayo.nekro_auto_sleep
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any

from zoneinfo import ZoneInfo

# NekroAgent public API imports
from nekro_agent.api import i18n
from nekro_agent.api.plugin import (
    ConfigBase,
    ExtraField,
    NekroPlugin,
    SandboxMethodType,
)
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.api.signal import MsgSignal
from nekro_agent.schemas.chat_message import ChatMessage
from pydantic import Field

from .engine import (
    ActionForceWake,
    clear_wake_decision,
    close_timer_interval,
    open_timer_interval,
    ActionNone,
    ActionSendFixed,
    ActionSendResumeSleep,
    ActionSendWakeNotice,
    handle_idle_sleep_back,
    handle_resume_sleep,
    handle_message_while_asleep,
    has_active_timer_lease,
    is_idle_expired,
    mark_notice_failed,
    mark_notice_sent,
    refresh_idle_deadline,
    settle_natural_wake,
    transition_to_awake,
    transition_to_sleep,
)
from .models import (
    DATA_KEY,
    PLUGIN_KEY,
    ChatSleepState,
    NotificationStatus,
    SleepCycle,
    SleepStatus,
    SourceType,
)
from .persistence import SleepStateStore
from .quality import compute_quality, compute_quality_detail
from .runtime import (
    current_source,
    make_schedule_agent_task_wrapper,
    unwrap_callable,
    wrap_callable,
)
from .schedule import (
    compute_cycle_boundaries,
    create_config_snapshot,
    find_sleep_date_for_now,
)

logger = logging.getLogger("nekro_auto_sleep")

# ---------------------------------------------------------------------------
# Plugin instance
# ---------------------------------------------------------------------------

plugin = NekroPlugin(
    name="自动睡眠",
    module_name="nekro_auto_sleep",
    description="为每个会话提供独立的拟人化睡眠周期，支持叫醒协议和睡眠质量统计",
    version="1.0.0",
    author="Akiyo_dayo",
    url="https://github.com/Akiyo-dayo/NekroAgent_ByAkiyo",
    allow_sleep=True,
    i18n_name=i18n.i18n_text(
        zh_CN="自动睡眠",
        en_US="Auto Sleep",
    ),
    i18n_description=i18n.i18n_text(
        zh_CN="为每个会话提供独立的拟人化睡眠周期，支持叫醒协议和睡眠质量统计",
        en_US="Per-chat humanized sleep cycles with wake protocol and sleep quality tracking",
    ),
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@plugin.mount_config()
class SleepConfig(ConfigBase):
    """自动睡眠配置"""

    ENABLED: bool = Field(
        default=True,
        title="启用自动睡眠",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="启用自动睡眠", en_US="Enable Auto Sleep"),
            i18n_description=i18n.i18n_text(
                zh_CN="总开关，关闭后所有会话停止新的睡眠行为",
                en_US="Master switch; disabling stops new sleep behavior for all chats",
            ),
        ).model_dump(),
    )
    TIMEZONE: str = Field(
        default="Asia/Shanghai",
        title="时区",
        json_schema_extra=ExtraField(
            placeholder="Asia/Shanghai",
            i18n_title=i18n.i18n_text(zh_CN="时区", en_US="Timezone"),
            i18n_description=i18n.i18n_text(
                zh_CN="IANA 时区名称，所有睡眠时间按此时区解释",
                en_US="IANA timezone name; all sleep times are interpreted in this timezone",
            ),
        ).model_dump(),
    )
    SLEEP_TIME: str = Field(
        default="23:00",
        title="入睡时间",
        json_schema_extra=ExtraField(
            placeholder="23:00",
            i18n_title=i18n.i18n_text(zh_CN="入睡时间", en_US="Bedtime"),
            i18n_description=i18n.i18n_text(
                zh_CN="每日自动入睡的时间，格式 HH:MM",
                en_US="Daily automatic bedtime, format HH:MM",
            ),
        ).model_dump(),
    )
    WAKE_TIME_START: str = Field(
        default="06:45",
        title="起床时间范围（起始）",
        json_schema_extra=ExtraField(
            placeholder="06:45",
            i18n_title=i18n.i18n_text(zh_CN="起床时间范围（起始）", en_US="Wake Range Start"),
            i18n_description=i18n.i18n_text(
                zh_CN="随机起床时间的最早时刻，格式 HH:MM",
                en_US="Earliest possible random wake-up time, format HH:MM",
            ),
        ).model_dump(),
    )
    WAKE_TIME_END: str = Field(
        default="08:30",
        title="起床时间范围（结束）",
        json_schema_extra=ExtraField(
            placeholder="08:30",
            i18n_title=i18n.i18n_text(zh_CN="起床时间范围（结束）", en_US="Wake Range End"),
            i18n_description=i18n.i18n_text(
                zh_CN="随机起床时间的最晚时刻，格式 HH:MM",
                en_US="Latest possible random wake-up time, format HH:MM",
            ),
        ).model_dump(),
    )
    WAKE_RANDOM_STEP_MINUTES: int = Field(
        default=1,
        title="起床随机粒度（分钟）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="起床随机粒度（分钟）", en_US="Wake Random Step (min)"),
            i18n_description=i18n.i18n_text(
                zh_CN="在起床范围内按此分钟数生成候选起床点，1-60",
                en_US="Candidate wake times are generated at this minute interval within the range, 1-60",
            ),
        ).model_dump(),
    )
    NEAR_WAKE_MINUTES: int = Field(
        default=60,
        title="接近起床提前量（分钟）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="接近起床提前量（分钟）", en_US="Near Wake Window (min)"),
            i18n_description=i18n.i18n_text(
                zh_CN="距离计划起床还有这么多分钟时，提示语改为「还没起床」，0-720",
                en_US="Within this many minutes of the planned wake-up the prompt changes to 'not up yet', 0-720",
            ),
        ).model_dump(),
    )
    WAKE_CONFIRM_WINDOW_SECONDS: int = Field(
        default=180,
        title="叫醒确认窗口（秒）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="叫醒确认窗口（秒）", en_US="Wake Confirm Window (s)"),
            i18n_description=i18n.i18n_text(
                zh_CN="首次呼叫后，同一用户需在此秒数内再次呼叫才能叫醒，10-1800",
                en_US="After the first call, the same user must call again within this many seconds to wake up, 10-1800",
            ),
        ).model_dump(),
    )
    URGENT_KEYWORDS: str = Field(
        default="",
        title="紧急唤醒关键词",
        json_schema_extra=ExtraField(
            is_textarea=True,
            i18n_title=i18n.i18n_text(zh_CN="紧急唤醒关键词", en_US="Urgent Keywords"),
            i18n_description=i18n.i18n_text(
                zh_CN="可选的关键词直通：消息里出现这些词且是冲着 Bot 来的，就跳过「要叫醒吗」直接叫醒。"
                "默认留空关闭——是否需要叫醒本来是交给 LLM 判断的，关键词只会带来误判",
                en_US="Optional keyword shortcut: a message aimed at the bot containing one of these skips the question. "
                "Empty (default) disables it — the wake decision belongs to the LLM",
            ),
        ).model_dump(),
    )
    WAKE_PROMPT_ASLEEP: str = Field(
        default="【{persona}已经睡了 要叫醒{persona}吗？】",
        title="叫醒提示语",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="叫醒提示语", en_US="Wake Prompt"),
            i18n_description=i18n.i18n_text(
                zh_CN="睡眠中被呼叫时发出的固定提示（不经 LLM），{persona} 会替换为人设名。"
                "想让用户知道怎么回可以写成「…要叫醒吗？（回「叫醒」或「不用」）」",
                en_US="Fixed prompt sent when called during sleep (never goes through the LLM); "
                "{persona} is replaced with the preset name",
            ),
        ).model_dump(),
    )
    WAKE_PROMPT_NEAR_WAKE: str = Field(
        default="【{persona}还没起床 要叫醒{persona}吗？】",
        title="接近起床时的提示语",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(
                zh_CN="接近起床时的提示语", en_US="Near-Wake Prompt"
            ),
            i18n_description=i18n.i18n_text(
                zh_CN="在「接近起床提前量」内被呼叫时改用这条，{persona} 会替换为人设名",
                en_US="Used instead when called within the near-wake window; {persona} is replaced with the preset name",
            ),
        ).model_dump(),
    )
    ANSWER_SCOPE: str = Field(
        default="offeree",
        title="谁的回答算数",
        json_schema_extra=ExtraField(
            placeholder="offeree",
            i18n_title=i18n.i18n_text(zh_CN="谁的回答算数", en_US="Answer Scope"),
            i18n_description=i18n.i18n_text(
                zh_CN="offeree: 只有被问的那个人的回答算数；anyone: 群里任何人都能回答",
                en_US="offeree: only the person who was asked can answer; anyone: anybody in the chat can",
            ),
        ).model_dump(),
    )
    MAX_OFFERS_PER_NIGHT: int = Field(
        default=3,
        title="每夜最多提示次数",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="每夜最多提示次数", en_US="Max Offers Per Night"),
            i18n_description=i18n.i18n_text(
                zh_CN="同一晚最多发送几次「要叫醒吗」，超出后静默，1-20",
                en_US="How many wake-up questions the bot may ask in one night before staying silent, 1-20",
            ),
        ).model_dump(),
    )
    OFFER_COOLDOWN_MINUTES: int = Field(
        default=20,
        title="提示冷却（分钟）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="提示冷却（分钟）", en_US="Offer Cooldown (min)"),
            i18n_description=i18n.i18n_text(
                zh_CN="两次「要叫醒吗」之间的最短间隔，0-240",
                en_US="Minimum gap between two wake-up questions, 0-240",
            ),
        ).model_dump(),
    )
    SNOOZE_MINUTES: int = Field(
        default=30,
        title="拒绝后静默时长（分钟）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="拒绝后静默时长（分钟）", en_US="Snooze After Refusal (min)"),
            i18n_description=i18n.i18n_text(
                zh_CN="有人明确说不用叫醒后，这段时间内不再提示，0-480",
                en_US="After somebody declines, stay silent for this long, 0-480",
            ),
        ).model_dump(),
    )
    HISTORY_MODE: str = Field(
        default="preserve",
        title="历史记录模式",
        json_schema_extra=ExtraField(
            placeholder="preserve",
            i18n_title=i18n.i18n_text(zh_CN="历史记录模式", en_US="History Mode"),
            i18n_description=i18n.i18n_text(
                zh_CN="preserve: 首次叫醒消息保留在历史中；strict: 完全拦截不记录",
                en_US="preserve: first wake message kept in history; strict: fully blocked",
            ),
        ).model_dump(),
    )
    CALL_KEYWORDS: str = Field(
        default="醒醒,起床,在吗",
        title="呼叫关键词",
        json_schema_extra=ExtraField(
            is_textarea=True,
            i18n_title=i18n.i18n_text(zh_CN="呼叫关键词", en_US="Call Keywords"),
            i18n_description=i18n.i18n_text(
                zh_CN="触发叫醒的关键词，逗号或换行分隔",
                en_US="Keywords that trigger wake-up, separated by comma or newline",
            ),
        ).model_dump(),
    )
    FALLBACK_PERSONA_NAME: str = Field(
        default="Bot",
        title="默认人格名",
        json_schema_extra=ExtraField(
            placeholder="Bot",
            i18n_title=i18n.i18n_text(zh_CN="默认人格名", en_US="Fallback Persona Name"),
            i18n_description=i18n.i18n_text(
                zh_CN="无法获取当前预设名称时使用的回退名",
                en_US="Fallback name used when the current preset name cannot be retrieved",
            ),
        ).model_dump(),
    )
    EARLY_WAKE_IDLE_MINUTES: int = Field(
        default=10,
        title="提前叫醒空闲超时（分钟）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="提前叫醒空闲超时（分钟）", en_US="Early Wake Idle Timeout (min)"),
            i18n_description=i18n.i18n_text(
                zh_CN="被叫醒后无新互动的自动睡回时间，1-240",
                en_US="Auto sleep-back time after being woken with no new interaction, 1-240",
            ),
        ).model_dump(),
    )
    WAKE_NOTICE_GRACE_MINUTES: int = Field(
        default=120,
        title="离线补发宽限期（分钟）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="离线补发宽限期（分钟）", en_US="Wake Notice Grace Period (min)"),
            i18n_description=i18n.i18n_text(
                zh_CN="重启后补发自然醒通知的最大延迟，超过则静默结算，0-1440",
                en_US="Maximum delay for sending missed wake notices after restart, 0-1440",
            ),
        ).model_dump(),
    )
    WAKE_NOTICE_POLICY: str = Field(
        default="always",
        title="起床播报策略",
        json_schema_extra=ExtraField(
            placeholder="always",
            i18n_title=i18n.i18n_text(zh_CN="起床播报策略", en_US="Wake Notice Policy"),
            i18n_description=i18n.i18n_text(
                zh_CN="always: 自然醒即播报；if_disturbed: 仅当夜里有人叫过时播报；never: 从不播报",
                en_US="always: announce every natural wake-up; if_disturbed: only when someone called during the night; never: stay silent",
            ),
        ).model_dump(),
    )
    HYDRATE_ACTIVE_DAYS: int = Field(
        default=14,
        title="启动装载活跃天数",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="启动装载活跃天数", en_US="Hydrate Active Days"),
            i18n_description=i18n.i18n_text(
                zh_CN="启动时装载最近多少天内有消息的频道，装载后它们才会按时入睡，1-365",
                en_US="On startup, load channels active within this many days; only loaded channels fall asleep on schedule, 1-365",
            ),
        ).model_dump(),
    )
    MAINTENANCE_INTERVAL_SECONDS: int = Field(
        default=15,
        title="维护循环间隔（秒）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="维护循环间隔（秒）", en_US="Maintenance Interval (s)"),
            i18n_description=i18n.i18n_text(
                zh_CN="后台维护任务的检查间隔，2-300",
                en_US="Background maintenance task check interval, 2-300",
            ),
        ).model_dump(),
    )
    NIGHT_TIMER_POLICY: str = Field(
        default="run",
        title="夜间定时任务策略",
        json_schema_extra=ExtraField(
            placeholder="run",
            i18n_title=i18n.i18n_text(
                zh_CN="夜间定时任务策略", en_US="Night Timer Policy"
            ),
            i18n_description=i18n.i18n_text(
                zh_CN="run（默认）：定时提醒照常执行，Bot 知道自己是在夜里值班，"
                "值班时长按半睡计入睡眠质量；block：夜间定时提醒只入历史、不执行",
                en_US="run (default): scheduled reminders still fire at night and the bot is told it is on night duty; "
                "block: night-time reminders are recorded but never executed",
            ),
        ).model_dump(),
    )
    NIGHT_DUTY_ASSUMED_MINUTES: int = Field(
        default=3,
        title="夜间值班计入时长（分钟）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(
                zh_CN="夜间值班计入时长（分钟）", en_US="Assumed Night Duty (min)"
            ),
            i18n_description=i18n.i18n_text(
                zh_CN="每次夜间定时任务按这么久计入睡眠质量（其中一半算作损失的休息），0-120。"
                "插件拿不到轮次的真实耗时，取一个保守估计而不是去猴补丁宿主内部方法",
                en_US="Each night-time scheduled task is accounted for as this long (half of it charged as lost rest), 0-120. "
                "The plugin cannot see the real round duration without patching host internals, so it estimates",
            ),
        ).model_dump(),
    )
    SLEEP_TARGET_HOURS: float = Field(
        default=0.0,
        title="目标睡眠时长（小时）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="目标睡眠时长（小时）", en_US="Sleep Target (hours)"),
            i18n_description=i18n.i18n_text(
                zh_CN="固定的 100% 基准时长。填 0（默认）表示以当晚自己的计划睡眠时长为准——起床时间本来就是范围内随机的，拿固定值当基准会让「今天起得早」被当成睡得差，0-16",
                en_US="Fixed duration that counts as 100%. 0 (default) scores against this night's own planned window, so an early random wake-up is not mistaken for a bad night, 0-16",
            ),
        ).model_dump(),
    )
    QUALITY_MIN: int = Field(
        default=20,
        title="睡眠质量下限",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="睡眠质量下限", en_US="Quality Min"),
            i18n_description=i18n.i18n_text(
                zh_CN="睡眠质量百分比的最低值，0-100",
                en_US="Minimum sleep quality percentage, 0-100",
            ),
        ).model_dump(),
    )
    QUALITY_MAX: int = Field(
        default=120,
        title="睡眠质量上限",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="睡眠质量上限", en_US="Quality Max"),
            i18n_description=i18n.i18n_text(
                zh_CN="睡眠质量百分比的最高值，100-200，不小于下限",
                en_US="Maximum sleep quality percentage, 100-200, must not be less than min",
            ),
        ).model_dump(),
    )
    QUALITY_JITTER_POINTS: float = Field(
        default=2.0,
        title="质量稳定扰动幅度",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="质量稳定扰动幅度", en_US="Quality Jitter Points"),
            i18n_description=i18n.i18n_text(
                zh_CN="每次睡眠质量的随机扰动范围（正负），0-15",
                en_US="Random jitter range (plus/minus) applied to each sleep quality score, 0-15",
            ),
        ).model_dump(),
    )


config = plugin.get_config(SleepConfig)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_store: SleepStateStore | None = None
_maintenance_task: asyncio.Task[None] | None = None
_installed_wraps: list[tuple[Any, str]] = []

# How long after an early wake the bot is still told it *just* woke up. The wake
# context itself is rendered from persisted state for the whole AWAKE_EARLY
# stretch; this only gates the extra "you were just shaken awake" line. It
# replaces an in-memory pop-once dict that leaked its payload into an unrelated
# round whenever the triggered round never ran (quota, observe mode, debounce).
_JUST_WOKEN_WINDOW_SECONDS = 180


def _utcnow() -> datetime:
    """Single clock seam for the wiring layer, so tests can pin the time."""
    return datetime.now(ZoneInfo("UTC"))


def _get_store() -> SleepStateStore:
    assert _store is not None, "Plugin not initialized"
    return _store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_persona_name(ctx: AgentCtx) -> str:
    """Resolve the current persona name, falling back to the configured default.

    `ctx.db_chat_channel` is a plain property on both the fork and upstream, so
    it must not be awaited: awaiting the ORM object raises TypeError, and with a
    bare `except` around it every prompt silently rendered the fallback name
    instead of the persona.
    """
    try:
        db_channel = ctx.db_chat_channel
        if db_channel is None:
            # Upstream builds can hand back a ctx with no channel prefilled.
            from nekro_agent.models.db_chat_channel import DBChatChannel

            db_channel = await DBChatChannel.get_channel(chat_key=ctx.chat_key)
        preset = await db_channel.get_preset()
        name = getattr(preset, "name", "")
        if name:
            return name
        logger.warning("Preset for %s has no name; using fallback", ctx.chat_key)
    except Exception as exc:
        logger.warning("Cannot resolve persona name for %s: %s", ctx.chat_key, exc)
    return config.FALLBACK_PERSONA_NAME


def _parse_keywords() -> list[str]:
    raw = config.CALL_KEYWORDS
    return [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]


def _message_text(message: ChatMessage) -> str:
    for attr in ("content", "content_text"):
        value = getattr(message, attr, None)
        if isinstance(value, str):
            return value
    return str(message)


def _is_valid_call(message: ChatMessage, persona_name: str) -> bool:
    """Whether a message counts as calling for the bot (spec §6.1).

    Only decides whether the bot should *offer* to wake up. Once a question is
    outstanding, the reply is read by `classify_answer` instead, which is what
    lets a bare "要" work and a bare "算了" stop meaning yes.
    """
    if hasattr(message, "channel_type") and message.channel_type == "private":
        return True
    if hasattr(message, "is_tome") and message.is_tome:
        return True
    text = _message_text(message)
    if persona_name and persona_name in text:
        return True
    keywords = _parse_keywords()
    for kw in keywords:
        if kw in text:
            return True
    return bool(_is_reply_to_bot(message))


def _is_reply_to_bot(message: ChatMessage) -> bool:
    """Conservative reply-to-bot detection (spec §6.1)."""
    if not hasattr(message, "extra_data") or not message.extra_data:
        return False
    ed = message.extra_data
    if isinstance(ed, dict):
        reply = ed.get("reply_to_bot") or ed.get("is_reply_to_self")
        if isinstance(reply, bool):
            return reply
    return False


def _get_user_id(message: ChatMessage) -> str:
    """Extract user identifier (spec §6.1)."""
    if hasattr(message, "platform_userid") and message.platform_userid:
        return str(message.platform_userid)
    if hasattr(message, "sender_id") and message.sender_id:
        return str(message.sender_id)
    return "unknown"


def _is_sleeping(chat_key: str) -> bool:
    """Quick check if a chat_key is in sleep state (for runtime wrappers)."""
    store = _get_store()
    state = store.get_cached(chat_key)
    if state is None:
        return False
    return state.status == SleepStatus.ASLEEP


def _night_timer_blocked(chat_key: str) -> bool:
    """Whether a directly scheduled agent round should be dropped right now.

    Only true while the chat is asleep *and* the operator asked for night-time
    scheduled tasks to be blocked. With the default `run` policy this is always
    false, which is the whole point: a reminder the user scheduled for 03:00 is
    something they asked for.
    """
    if not config.ENABLED:
        return False
    if config.NIGHT_TIMER_POLICY != "block":
        return False
    return _is_sleeping(chat_key)


def _make_config_snapshot() -> Any:
    return create_config_snapshot(
        timezone=config.TIMEZONE,
        sleep_time=config.SLEEP_TIME,
        wake_time_start=config.WAKE_TIME_START,
        wake_time_end=config.WAKE_TIME_END,
        wake_random_step_minutes=config.WAKE_RANDOM_STEP_MINUTES,
        near_wake_ratio=0.15,  # deprecated in schema v2, kept for rollback
        wake_confirm_window_seconds=config.WAKE_CONFIRM_WINDOW_SECONDS,
        history_mode=config.HISTORY_MODE,
        call_keywords=config.CALL_KEYWORDS,
        fallback_persona_name=config.FALLBACK_PERSONA_NAME,
        early_wake_idle_minutes=config.EARLY_WAKE_IDLE_MINUTES,
        quality_min=config.QUALITY_MIN,
        quality_max=config.QUALITY_MAX,
        quality_jitter_points=config.QUALITY_JITTER_POINTS,
        near_wake_minutes=config.NEAR_WAKE_MINUTES,
        sleep_target_hours=config.SLEEP_TARGET_HOURS,
        urgent_keywords=config.URGENT_KEYWORDS,
        answer_scope=config.ANSWER_SCOPE,
        max_offers_per_night=config.MAX_OFFERS_PER_NIGHT,
        offer_cooldown_minutes=config.OFFER_COOLDOWN_MINUTES,
        snooze_minutes=config.SNOOZE_MINUTES,
        asleep_prompt=config.WAKE_PROMPT_ASLEEP,
        near_wake_prompt=config.WAKE_PROMPT_NEAR_WAKE,
    )


# ---------------------------------------------------------------------------
# Message hooks
# ---------------------------------------------------------------------------


@plugin.mount_on_user_message()
async def on_user_message(ctx: AgentCtx, message: ChatMessage) -> MsgSignal | None:
    if not config.ENABLED:
        return MsgSignal.CONTINUE

    chat_key = ctx.chat_key
    store = _get_store()
    now_utc = _utcnow()
    persona_name = await _get_persona_name(ctx)
    user_id = _get_user_id(message)

    async def _process(state: ChatSleepState) -> ChatSleepState:
        nonlocal _result_signal, _result_action
        state.last_seen_at = now_utc

        if state.status == SleepStatus.AWAKE:
            _result_signal = MsgSignal.CONTINUE
            return state

        if state.status == SleepStatus.AWAKE_EARLY:
            # Another message means the exchange carried on, so the round that
            # was deciding whether to stay up is over either way.
            state = clear_wake_decision(state)
            state = refresh_idle_deadline(state, now_utc)
            _result_signal = MsgSignal.CONTINUE
            return state

        if state.status == SleepStatus.ASLEEP:
            state, action = handle_message_while_asleep(
                state,
                now_utc,
                user_id,
                _message_text(message),
                persona_name,
                _is_valid_call(message, persona_name),
            )
            _result_action = action

            if isinstance(action, ActionForceWake):
                # No side-channel cache: `woken_at` / `woken_by` are persisted on
                # the state, so every round of this early-awake stretch can
                # rebuild the wake context instead of only the first one.
                _result_signal = MsgSignal.FORCE_TRIGGER
            else:
                # Everything else stays out of the LLM but stays in the history:
                # the host returns on BLOCK_ALL *before* writing the message to
                # DBChatMessage, so blocking everything meant the bot woke up
                # with no memory of the night at all.
                _result_signal = (
                    MsgSignal.BLOCK_ALL
                    if config.HISTORY_MODE == "strict"
                    else MsgSignal.BLOCK_TRIGGER
                )

            return state

        _result_signal = MsgSignal.CONTINUE
        return state

    _result_signal: MsgSignal = MsgSignal.CONTINUE
    _result_action: Any = ActionNone()

    await store.with_state(chat_key, _process)

    if isinstance(_result_action, ActionSendFixed):
        # Record the question in `preserve` mode. Without it the woken bot sees
        # the same user calling twice with no turn of its own in between, which
        # is why the reply after a wake-up read as a non sequitur.
        try:
            await ctx.send_text(
                _result_action.text,
                record=_result_action.block_mode != "strict",
            )
        except Exception as exc:
            logger.error("Failed to send wake offer: %s", exc)

    return _result_signal


@plugin.mount_on_system_message()
async def on_system_message(ctx: AgentCtx, message: str) -> MsgSignal | None:
    if not config.ENABLED:
        return MsgSignal.CONTINUE

    chat_key = ctx.chat_key
    store = _get_store()
    state = store.get_cached(chat_key)

    if state is None:
        return MsgSignal.CONTINUE

    if state.status != SleepStatus.ASLEEP:
        return MsgSignal.CONTINUE

    if current_source.get() == SourceType.INTERNAL_WAKE_NOTICE:
        return MsgSignal.CONTINUE

    if config.NIGHT_TIMER_POLICY == "block":
        # Recorded so the morning context is continuous, but no agent round.
        return MsgSignal.BLOCK_TRIGGER

    # `run`: let it through. The host only starts a round for callers that pass
    # trigger_agent=True — which is the timer service — so ordinary system
    # notices still stay silent, and a reminder the user scheduled for 03:00
    # actually fires. The bot is told it is on night duty by the prompt
    # injection, and the stretch is charged to sleep quality.
    await _record_night_duty(chat_key)
    return MsgSignal.CONTINUE


async def _record_night_duty(chat_key: str) -> None:
    """Charge a night-time scheduled round to the sleep record.

    The plugin cannot see how long the round actually takes without wrapping a
    private host method, so it books a configured estimate instead. Half of the
    stretch is charged as lost rest; the sleep segment itself is left open,
    because a timer is not the bot getting out of bed.
    """
    minutes = max(0, min(120, config.NIGHT_DUTY_ASSUMED_MINUTES))
    if minutes == 0:
        return

    now_utc = _utcnow()
    task_id = f"night-duty-{now_utc.isoformat()}"

    async def _mark(state: ChatSleepState) -> ChatSleepState:
        if state.status != SleepStatus.ASLEEP or state.cycle is None:
            return state
        state = open_timer_interval(state, task_id, now_utc)
        return close_timer_interval(state, task_id, now_utc + timedelta(minutes=minutes))

    try:
        await _get_store().with_state(chat_key, _mark)
    except Exception as exc:
        logger.warning("Cannot record night duty for %s: %s", chat_key, exc)


# ---------------------------------------------------------------------------
# Prompt injection (one-shot wake info)
# ---------------------------------------------------------------------------


def _fmt_local(dt: datetime, tz_name: str) -> str:
    """Render an aware UTC datetime as HH:MM in the cycle timezone."""
    try:
        return dt.astimezone(ZoneInfo(tz_name)).strftime("%H:%M")
    except Exception:
        return dt.astimezone(ZoneInfo("UTC")).strftime("%H:%M UTC")


def _render_sleep_context(
    state: ChatSleepState,
    now_utc: datetime,
    persona_name: str = "",
) -> str:
    """Describe the current sleep situation for the prompt.

    Rendered from persisted state on every round rather than popped once, so the
    second and third replies after a wake-up still know it is the middle of the
    night, that the visible night messages arrived while the bot was asleep, and
    that `resume_sleep` is the way back to bed.
    """
    cycle = state.cycle
    if cycle is None:
        return ""

    tz_name = cycle.config_snapshot.timezone
    bedtime = _fmt_local(cycle.sleep_at, tz_name)
    planned = _fmt_local(cycle.planned_wake_at, tz_name)
    persona_placeholder = persona_name or cycle.config_snapshot.fallback_persona_name

    if state.status == SleepStatus.AWAKE_EARLY and state.woken_at is not None:
        awake_seconds = max(0.0, (now_utc - state.woken_at).total_seconds())
        who = f"用户 {state.woken_by} " if state.woken_by else ""
        verb = "紧急叫醒" if state.woken_reason == "urgent" else "提前叫醒"
        lines = [
            f"[睡眠状态] 你今晚 {bedtime} 就寝，原定 {planned} 自然醒。",
            f"{_fmt_local(state.woken_at, tz_name)} {who}把你{verb}了，"
            f"到现在醒了约 {int(awake_seconds // 60)} 分钟"
            f"（当前 {_fmt_local(now_utc, tz_name)}）。",
            f"{bedtime} 之后的消息你都是在睡着时收到的，刚醒来才看见，"
            "不要表现得像你当时就在场。",
            f"如果对方不再需要你，可以调用 resume_sleep 回去继续睡；"
            f"否则到 {planned} 会自然醒。",
        ]
        if state.wake_decision_pending:
            # This round decides whether the caller actually wanted the bot up.
            # The plugin deliberately does not guess that from the wording.
            lines = [
                f"[睡眠状态] 你今晚 {bedtime} 就寝，原定 {planned} 自然醒，刚才还在睡。",
                f"有人叫你，你已经回过一句「要叫醒{persona_placeholder}吗？」，"
                "最后这条消息就是对方的回答。",
                "先判断对方到底要不要叫醒你：",
                "· 要 —— 就正常回复，你已经醒了，但还带着睡意；",
                "· 不要、只是随口一句、或者根本不是在找你 —— "
                "调用 resume_sleep 继续睡，并且**不要输出任何内容**。",
                f"{bedtime} 之后的消息你都是在睡着时收到的，刚醒来才看见，"
                "不要表现得像你当时就在场。",
            ]
            return "\n".join(lines)

        if awake_seconds <= _JUST_WOKEN_WINDOW_SECONDS:
            lines.insert(0, "你刚刚被叫醒，还带着睡意。")
        return "\n".join(lines)

    if state.status == SleepStatus.ASLEEP:
        return (
            f"[睡眠状态] 你从 {bedtime} 起一直在睡觉，计划 {planned} 自然醒。"
            "现在是夜里值班——有个定时任务把你临时叫起来处理，"
            "处理完就继续睡，不用寒暄，也不要表现得像已经起床了。"
        )

    return ""


@plugin.mount_prompt_inject_method(
    "sleep_status",
    "注入当前睡眠状态，让被叫醒后的回复接得上夜里的上下文",
)
async def inject_sleep_status(ctx: AgentCtx) -> str:
    if not config.ENABLED:
        return ""

    state = _get_store().get_cached(ctx.chat_key)
    if state is None:
        return ""

    persona_name = ""
    if state.status == SleepStatus.AWAKE_EARLY and state.wake_decision_pending:
        persona_name = await _get_persona_name(ctx)

    return _render_sleep_context(state, _utcnow(), persona_name)


# ---------------------------------------------------------------------------
# Sandbox method: resume_sleep
# ---------------------------------------------------------------------------


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    "resume_sleep",
    "继续睡（仅在被提前叫醒后、计划起床前可用）。"
    "刚被叫醒那一轮如果判断对方其实并不需要你，调用它并且不要输出任何内容，"
    "会静默睡回去；聊完之后再调用则会说一句「已睡下」",
)
async def resume_sleep_tool(_ctx: AgentCtx) -> str:
    chat_key = _ctx.chat_key
    store = _get_store()
    now_utc = _utcnow()
    persona_name = await _get_persona_name(_ctx)

    result_text = ""

    async def _process(state: ChatSleepState) -> ChatSleepState:
        nonlocal result_text
        declined = state.wake_decision_pending
        state, action = handle_resume_sleep(state, now_utc, persona_name)
        if isinstance(action, ActionSendResumeSleep):
            try:
                # Recorded like the wake offer: the transcript should show the
                # bot going back to bed, not jump straight to silence.
                await _ctx.send_text(
                    action.text, record=config.HISTORY_MODE != "strict"
                )
            except Exception as exc:
                logger.error("Failed to send resume sleep message: %s", exc)
        result_text = (
            "已确认对方并不需要你，继续睡；本轮不要再输出任何内容。"
            if declined
            else "ok"
        )
        return state

    try:
        await store.with_state(chat_key, _process)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    return result_text


# ---------------------------------------------------------------------------
# Maintenance loop
# ---------------------------------------------------------------------------


async def _maintenance_loop() -> None:
    """Periodic maintenance: auto-sleep, natural wake, idle sleep-back."""
    store = _get_store()
    while True:
        try:
            interval = max(2, min(300, config.MAINTENANCE_INTERVAL_SECONDS))
            await asyncio.sleep(interval)

            if not config.ENABLED:
                continue

            now_utc = _utcnow()
            tz = ZoneInfo(config.TIMEZONE)

            for chat_key in list(store.known_chat_keys()):
                try:
                    await _maintain_chat(store, chat_key, now_utc, tz)
                except Exception as exc:
                    logger.error("Maintenance error for %s: %s", chat_key, exc)

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Maintenance loop error: %s", exc)
            await asyncio.sleep(5)


async def _maintain_chat(
    store: SleepStateStore,
    chat_key: str,
    now_utc: datetime,
    tz: ZoneInfo,
) -> None:
    """Run maintenance checks for a single chat_key."""
    state = store.get_cached(chat_key)
    if state is None:
        return

    if state.status == SleepStatus.AWAKE:
        await _check_sleep_transition(store, chat_key, now_utc, tz)

    elif state.status == SleepStatus.ASLEEP:
        if state.cycle and now_utc >= state.cycle.planned_wake_at:
            if not has_active_timer_lease(state):
                await _settle_wake(store, chat_key, now_utc)

    elif state.status == SleepStatus.AWAKE_EARLY:
        if state.cycle and now_utc >= state.cycle.planned_wake_at:
            async def _settle(s: ChatSleepState) -> ChatSleepState:
                return transition_to_awake(s, now_utc)
            await store.with_state(chat_key, _settle)
        elif is_idle_expired(state, now_utc):
            if not has_active_timer_lease(state):
                async def _idle_back(s: ChatSleepState) -> ChatSleepState:
                    return handle_idle_sleep_back(s, now_utc)
                await store.with_state(chat_key, _idle_back)


async def _check_sleep_transition(
    store: SleepStateStore,
    chat_key: str,
    now_utc: datetime,
    tz: ZoneInfo,
    backdate_segment: bool = False,
) -> None:
    """AWAKE -> ASLEEP, at most once per local sleep_date.

    The previous check asked whether `now` fell in a 30-second window just after
    bedtime, which quietly stopped working as soon as an operator raised
    MAINTENANCE_INTERVAL_SECONDS above 30 (the field allows up to 300). Keying
    on the sleep_date instead means any tick during the night is enough, and a
    second tick cannot start the night over.

    `backdate_segment` is for boot reconciliation: when the process was down
    across bedtime, the sleep segment starts at the real bedtime rather than at
    startup, so the morning report does not claim a five-minute night.
    """
    sleep_date = find_sleep_date_for_now(
        now_utc, tz, config.SLEEP_TIME, config.WAKE_TIME_START, config.WAKE_TIME_END
    )
    if sleep_date is None:
        return

    sleep_date_iso = sleep_date.isoformat()
    cached = store.get_cached(chat_key)
    if cached is not None and cached.cycle is not None and cached.cycle.sleep_date == sleep_date_iso:
        return

    snap = _make_config_snapshot()
    segment_open_at = now_utc
    if backdate_segment:
        try:
            bedtime, _ws, _we = compute_cycle_boundaries(
                sleep_date, tz, config.SLEEP_TIME, config.WAKE_TIME_START, config.WAKE_TIME_END
            )
            segment_open_at = min(bedtime, now_utc)
        except Exception as exc:
            logger.warning("Cannot backdate bedtime for %s: %s", chat_key, exc)

    async def _sleep(s: ChatSleepState) -> ChatSleepState:
        if s.status != SleepStatus.AWAKE:
            return s
        if s.cycle is not None and s.cycle.sleep_date == sleep_date_iso:
            return s
        new_state = transition_to_sleep(
            s,
            now_utc,
            snap,
            sleep_date_local=sleep_date_iso,
            segment_open_at=segment_open_at,
        )
        if new_state.cycle is not None and now_utc >= new_state.cycle.planned_wake_at:
            # Enabled (or restarted) after this night's wake-up point: record the
            # night as done instead of going to sleep for the remaining minutes.
            return transition_to_awake(new_state, now_utc)
        return new_state

    await store.with_state(chat_key, _sleep)


async def _settle_wake(
    store: SleepStateStore,
    chat_key: str,
    now_utc: datetime,
) -> None:
    """Settle a natural wake-up and announce it.

    The score and the duration are both produced inside `settle_natural_wake`,
    after the final sleep segment has been closed. Computing either one out here
    is what made every morning report say "quality 60%, slept 8h55m": the score
    was measured against a still-open segment worth zero seconds, the duration
    was rebuilt afterwards from the closed one.
    """
    ctx: AgentCtx | None = None
    persona_name = config.FALLBACK_PERSONA_NAME
    try:
        ctx = await AgentCtx.create_by_chat_key(chat_key)
        persona_name = await _get_persona_name(ctx)
    except Exception as exc:
        logger.warning("Cannot build ctx for %s, using fallback name: %s", chat_key, exc)

    notice_action: ActionSendWakeNotice | None = None
    breakdown: dict[str, float] | None = None

    def _score(cycle: SleepCycle, seconds: float) -> int:
        nonlocal breakdown
        detail = compute_quality_detail(cycle, seconds)
        breakdown = detail.as_dict()
        return detail.score

    async def _wake(s: ChatSleepState) -> ChatSleepState:
        nonlocal notice_action
        policy = config.WAKE_NOTICE_POLICY
        if s.cycle is not None:
            late_seconds = (now_utc - s.cycle.planned_wake_at).total_seconds()
            grace_seconds = max(0, config.WAKE_NOTICE_GRACE_MINUTES) * 60
            if late_seconds > grace_seconds:
                logger.info(
                    "Wake-up for %s is %.0f min late (grace %d min); settling silently",
                    chat_key,
                    late_seconds / 60,
                    config.WAKE_NOTICE_GRACE_MINUTES,
                )
                policy = "never"
        new_state, action = settle_natural_wake(
            s, now_utc, persona_name, _score, policy
        )
        if isinstance(action, ActionSendWakeNotice):
            notice_action = action
        if breakdown is not None and new_state.cycle is not None:
            # Keep every term behind the percentage, so a score that looks wrong
            # can be explained instead of reverse-engineered.
            new_state.cycle = new_state.cycle.model_copy(
                update={"quality_breakdown": breakdown}
            )
        return new_state

    await store.with_state(chat_key, _wake)

    if notice_action is None:
        return

    if ctx is None:
        logger.error("No ctx for %s; cannot deliver wake notice", chat_key)
        await store.with_state(chat_key, lambda s: _identity(mark_notice_failed(s)))
        return

    try:
        token = current_source.set(SourceType.INTERNAL_WAKE_NOTICE)
        try:
            await ctx.send_text(notice_action.text, record=False)
        finally:
            current_source.reset(token)

        async def _mark_sent(s: ChatSleepState) -> ChatSleepState:
            return mark_notice_sent(s)

        await store.with_state(chat_key, _mark_sent)
    except Exception as exc:
        logger.error("Failed to send wake notice for %s: %s", chat_key, exc)

        async def _mark_failed(s: ChatSleepState) -> ChatSleepState:
            return mark_notice_failed(s)

        await store.with_state(chat_key, _mark_failed)


async def _identity(state: ChatSleepState) -> ChatSleepState:
    return state


# ---------------------------------------------------------------------------
# Boot reconciliation
# ---------------------------------------------------------------------------


async def _discover_chat_keys() -> set[str]:
    """Every chat_key the plugin should be watching after a (re)start.

    Two sources, both best-effort: channels that already have persisted sleep
    state (they may be mid-night and need settling), and channels with recent
    traffic (they need to go to bed tonight even if nobody talks first).
    """
    keys: set[str] = set()

    try:
        from nekro_agent.models.db_plugin_data import DBPluginData

        plugin_key = getattr(plugin, "key", PLUGIN_KEY)
        rows = await DBPluginData.filter(plugin_key=plugin_key, data_key=DATA_KEY).values(
            "target_chat_key"
        )
        keys.update(str(r["target_chat_key"]) for r in rows if r.get("target_chat_key"))
    except Exception as exc:
        logger.error("Cannot enumerate persisted sleep state: %s", exc)

    try:
        from nekro_agent.models.db_chat_message import DBChatMessage

        days = max(1, min(365, config.HYDRATE_ACTIVE_DAYS))
        cutoff = int(time.time()) - days * 86400
        rows = (
            await DBChatMessage.filter(send_timestamp__gte=cutoff)
            .distinct()
            .values("chat_key")
        )
        keys.update(str(r["chat_key"]) for r in rows if r.get("chat_key"))
    except Exception as exc:
        logger.warning("Cannot enumerate recently active channels: %s", exc)

    return keys


async def _reconcile_chat(
    store: SleepStateStore,
    chat_key: str,
    now_utc: datetime,
    tz: ZoneInfo,
) -> None:
    """Align one chat_key with the wall clock after a restart."""
    state = store.get_cached(chat_key)
    if state is None:
        return

    if state.status in (SleepStatus.ASLEEP, SleepStatus.AWAKE_EARLY):
        if state.cycle is not None and now_utc >= state.cycle.planned_wake_at:
            await _settle_wake(store, chat_key, now_utc)
        return

    if state.status == SleepStatus.AWAKE:
        await _check_sleep_transition(store, chat_key, now_utc, tz, backdate_segment=True)


async def _boot_reconcile(store: SleepStateStore, now_utc: datetime) -> None:
    """Hydrate every known channel and settle whatever the downtime skipped."""
    try:
        tz = ZoneInfo(config.TIMEZONE)
    except Exception as exc:
        logger.error("Invalid timezone %r: %s", config.TIMEZONE, exc)
        return

    chat_keys = await _discover_chat_keys()
    logger.info("Boot reconciliation over %d chat_key(s)", len(chat_keys))

    for chat_key in sorted(chat_keys):
        try:
            await store.hydrate(chat_key)
            await _reconcile_chat(store, chat_key, now_utc, tz)
        except Exception as exc:
            logger.error("Boot reconciliation failed for %s: %s", chat_key, exc)


# ---------------------------------------------------------------------------
# Runtime wrapping (capability probing)
# ---------------------------------------------------------------------------


def _install_wraps() -> bool:
    """Install the one optional host gate, via capability probing.

    Only `schedule_agent_task` is wrapped, and only so that
    NIGHT_TIMER_POLICY=block can stop a directly scheduled round. It is a public
    method; nothing private is patched any more. Inbound user messages never get
    this far while asleep — `on_user_message` has already blocked them — so with
    the default `run` policy this wrapper is a pass-through.
    """
    success = True

    try:
        from nekro_agent.services.message_service import message_service as ms
        if ms is None:
            logger.error("message_service singleton not found")
            return False

        if hasattr(ms, "schedule_agent_task") and callable(ms.schedule_agent_task):
            wrapper = make_schedule_agent_task_wrapper(_night_timer_blocked)
            if wrap_callable(ms, "schedule_agent_task", wrapper):
                _installed_wraps.append((ms, "schedule_agent_task"))
        else:
            logger.error("schedule_agent_task not found on message_service")
            success = False

    except ImportError as exc:
        logger.error("Cannot import message_service: %s", exc)
        return False

    logger.info("Night timer policy: %s", config.NIGHT_TIMER_POLICY)
    return success


def _uninstall_wraps() -> None:
    """Restore all wrapped callables."""
    for obj, attr in reversed(_installed_wraps):
        unwrap_callable(obj, attr)
    _installed_wraps.clear()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@plugin.mount_init_method()
async def init() -> None:
    global _store, _maintenance_task

    _store = SleepStateStore(plugin.store)

    if not _install_wraps():
        logger.error("Some runtime wraps failed to install; plugin may not fully function")

    if config.ENABLED:
        try:
            await _boot_reconcile(_store, _utcnow())
        except Exception as exc:
            logger.error("Boot reconciliation aborted: %s", exc)

    _maintenance_task = asyncio.create_task(_maintenance_loop())
    logger.info("Auto-sleep plugin initialized")


@plugin.mount_cleanup_method()
async def cleanup() -> None:
    global _store, _maintenance_task

    if _maintenance_task is not None:
        _maintenance_task.cancel()
        try:
            await _maintenance_task
        except asyncio.CancelledError:
            pass
        _maintenance_task = None

    _uninstall_wraps()

    if _store:
        _store.clear_all()
        _store = None

    logger.info("Auto-sleep plugin cleaned up")
