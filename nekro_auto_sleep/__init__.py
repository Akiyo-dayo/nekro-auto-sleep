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
from dataclasses import dataclass
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
    ResolvedSchedule,
    ScheduleOverride,
    SleepCycle,
    SleepStatus,
    SourceType,
)
from .persistence import ScheduleOverrideStore, SleepStateStore
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
    resolve_schedule,
)

# The domain layer logs through stdlib `logging` so it stays host-free. The host
# logs through loguru and never bridged the two, so on a real install every
# INFO the plugin emitted went nowhere and every WARNING landed on bare stderr
# with no timestamp and no plugin tag — which is how "did boot reconciliation
# even run?" became unanswerable from the log. `_HostLogBridge` forwards the
# domain layer into the host logger; the wiring layer below uses it directly.
_domain_logger = logging.getLogger("nekro_auto_sleep")
logger = _domain_logger


class _HostLogBridge(logging.Handler):
    """Forward stdlib records from the domain layer into the host logger."""

    _METHODS = {"debug", "info", "warning", "error", "critical"}

    def __init__(self, host_logger: Any) -> None:
        super().__init__()
        self._host = host_logger

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = record.levelname.lower()
            if level not in self._METHODS:
                level = "info"

            # Everything goes through stdlib, including the wiring layer, so
            # `%s` placeholders are interpolated here by `getMessage()`. Handing
            # the raw template to loguru instead would print it literally: it
            # formats with `{}`, not `%`.
            target = self._host
            opt = getattr(target, "opt", None)
            if callable(opt):
                try:
                    candidate = opt(depth=6)
                    if hasattr(candidate, level):
                        target = candidate
                except Exception:
                    pass

            emit = getattr(target, level, None)
            if emit is not None:
                emit(record.getMessage())
        except Exception:  # pragma: no cover - logging must never raise
            pass


_log_bridge: logging.Handler | None = None


def _install_log_bridge() -> None:
    """Route this package into the host log, once."""
    global _log_bridge

    host_logger = getattr(plugin, "logger", None)
    if host_logger is None:
        return

    if _log_bridge is not None:
        return
    _log_bridge = _HostLogBridge(host_logger)
    _domain_logger.addHandler(_log_bridge)
    _domain_logger.setLevel(logging.INFO)
    _domain_logger.propagate = False


def _remove_log_bridge() -> None:
    global _log_bridge

    if _log_bridge is not None:
        _domain_logger.removeHandler(_log_bridge)
        _log_bridge = None

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
    # Without a brief the host treats the plugin as not dormant-capable under the
    # `auto` strategy, so `allow_sleep=True` alone kept its prompt resident all
    # day. (Host-side "sleep" here means plugin dormancy in the prompt — nothing
    # to do with the bot going to bed.)
    sleep_brief="让 Bot 按作息入睡与起床；仅在被提前叫醒、需要决定继续睡还是起来时才用得上。",
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
_overrides: ScheduleOverrideStore | None = None
_maintenance_task: asyncio.Task[None] | None = None
_installed_wraps: list[tuple[Any, str]] = []

# Resolved settings are cached because the maintenance loop asks for them per
# channel per tick and resolving costs a channel lookup. Overrides change by
# hand, so a long TTL plus explicit invalidation on write is plenty.
_SETTINGS_CACHE_TTL_SECONDS = 300.0
_settings_cache: dict[str, tuple[float, "ChatSettings"]] = {}

# Accounts belonging to this deployment's own bots. Two instances sharing a
# group see each other's messages as ordinary inbound traffic, and the wake-up
# prompt contains the persona name — which is a wake keyword. Left unguarded,
# one bot's 【X已经睡了 要叫醒X吗？】 is a valid call to the other, and they
# answer each other until the nightly cap runs out. Observed live on the test
# box the first night v2 ran.
_OWN_ACCOUNTS_TTL_SECONDS = 300.0
_own_accounts_cache: tuple[float, frozenset[str]] | None = None


@dataclass(frozen=True)
class ChatSettings:
    """Everything that varies per channel, resolved once.

    `config` is the plugin config as it applies to *this channel's instance*.
    In the Akiyo fork two accounts can sit in the same group; they get separate
    chat_keys, so they already keep separate nights — but their config would
    still have come from the global file, because background tasks carry no
    instance context for `ScopedPluginConfig` to pick up. Resolving it
    explicitly here is what makes two bots in one group able to keep genuinely
    different hours.
    """

    instance_key: str
    config: Any
    schedule: ResolvedSchedule

# How long after an early wake the bot is still told it *just* woke up. The wake
# context itself is rendered from persisted state for the whole AWAKE_EARLY
# stretch; this only gates the extra "you were just shaken awake" line. It
# replaces an in-memory pop-once dict that leaked its payload into an unrelated
# round whenever the triggered round never ran (quota, observe mode, debounce).
_JUST_WOKEN_WINDOW_SECONDS = 180


# The v1 default floor. It clipped everything the old model produced, and an
# upgraded install keeps it because plugin config is persisted per field: the
# new defaults only apply to fresh installs.
_V1_QUALITY_MIN = 60
_QUALITY_FLOOR_WARN_ABOVE = 40


def collect_upgrade_warnings() -> list[str]:
    """Config that was fine for v1 and quietly breaks v2.

    Returned rather than logged directly so it can be tested, and so
    `/sleep status` can show the same thing where operators actually look.
    """
    warnings: list[str] = []
    if config.QUALITY_MIN > _QUALITY_FLOOR_WARN_ABOVE:
        warnings.append(
            f"睡眠质量下限当前是 {config.QUALITY_MIN}"
            + ("，这正是 v1 的默认值。" if config.QUALITY_MIN == _V1_QUALITY_MIN else "。")
            +
            "新的评分量程是 20–110，下限设这么高会把所有糟糕的夜晚一律夹到同一个数字上，"
            "看起来就像分数不动——这正是 v1 那个 bug 的症状。建议改成 20。"
        )
    return warnings


def _utcnow() -> datetime:
    """Single clock seam for the wiring layer, so tests can pin the time."""
    return datetime.now(ZoneInfo("UTC"))


def _get_store() -> SleepStateStore:
    assert _store is not None, "Plugin not initialized"
    return _store


def _get_overrides() -> ScheduleOverrideStore:
    assert _overrides is not None, "Plugin not initialized"
    return _overrides


def _global_schedule() -> ResolvedSchedule:
    return resolve_schedule(
        timezone=config.TIMEZONE,
        sleep_time=config.SLEEP_TIME,
        wake_time_start=config.WAKE_TIME_START,
        wake_time_end=config.WAKE_TIME_END,
    )


async def _own_account_ids() -> frozenset[str]:
    """Platform account ids of every adapter instance in this deployment."""
    global _own_accounts_cache

    cached = _own_accounts_cache
    if cached is not None and (time.monotonic() - cached[0]) < _OWN_ACCOUNTS_TTL_SECONDS:
        return cached[1]

    accounts: set[str] = set()
    try:
        from nekro_agent.models.db_adapter_instance import DBAdapterInstance

        rows = await DBAdapterInstance.all().values("provider_account_id")
        accounts = {
            str(row["provider_account_id"]) for row in rows if row.get("provider_account_id")
        }
    except Exception as exc:
        logger.debug("Cannot enumerate own bot accounts: %s", exc)

    result = frozenset(accounts)
    _own_accounts_cache = (time.monotonic(), result)
    return result


def invalidate_own_accounts_cache() -> None:
    global _own_accounts_cache
    _own_accounts_cache = None


async def _is_own_bot(message: ChatMessage) -> bool:
    """Whether this message came from one of our own bot accounts."""
    sender = _get_user_id(message)
    if not sender or sender == "unknown":
        return False
    return sender in await _own_account_ids()


async def _channel_and_preset(chat_key: str) -> tuple[object, object]:
    """The channel row and the preset it is running, for override targeting."""
    channel = await _channel_for(chat_key)
    return channel, getattr(channel, "preset_id", None)


async def _channel_for(chat_key: str):
    """The channel row, or None when it cannot be looked up."""
    try:
        from nekro_agent.models.db_chat_channel import DBChatChannel

        return await DBChatChannel.get_channel(chat_key=chat_key)
    except Exception as exc:
        logger.debug("Cannot load channel %s: %s", chat_key, exc)
        return None


def _instance_config(instance_key: str):
    """The plugin config as it applies to one adapter instance.

    Fork-only: upstream has no per-instance config layer, and the probe below
    simply falls through to the global config there. Resolved by explicit
    instance_key rather than through the host contextvar, because the
    maintenance loop is a background task and carries no inbound context.
    """
    if not instance_key:
        return config
    try:
        from nekro_agent.services.plugin.scope import resolve_scoped_config
    except ImportError:
        return config
    try:
        return resolve_scoped_config(
            plugin.key, SleepConfig, plugin.get_global_config(SleepConfig), instance_key
        )
    except Exception as exc:
        logger.warning("Cannot resolve config for instance %r: %s", instance_key, exc)
        return config


def _instance_schedule_fields(instance_key: str) -> set[str]:
    """Which schedule fields this instance actually overrides (for provenance)."""
    if not instance_key:
        return set()
    try:
        from nekro_agent.services.plugin.scope import load_instance_config_overrides

        overrides = load_instance_config_overrides(plugin.key, instance_key)
    except Exception:
        return set()
    mapping = {
        "TIMEZONE": "timezone",
        "SLEEP_TIME": "sleep_time",
        "WAKE_TIME_START": "wake_time_start",
        "WAKE_TIME_END": "wake_time_end",
    }
    return {field for key, field in mapping.items() if key in overrides}


def invalidate_settings_cache(chat_key: str | None = None) -> None:
    if chat_key is None:
        _settings_cache.clear()
    else:
        _settings_cache.pop(chat_key, None)


# Kept under the old name so existing callers and docs stay valid.
invalidate_schedule_cache = invalidate_settings_cache


async def _settings_for(chat_key: str) -> ChatSettings:
    """Resolve one channel: channel > persona > instance > global."""
    cached = _settings_cache.get(chat_key)
    if cached is not None and (time.monotonic() - cached[0]) < _SETTINGS_CACHE_TTL_SECONDS:
        return cached[1]

    channel = await _channel_for(chat_key)
    instance_key = str(getattr(channel, "instance_key", "") or "")
    preset_id = getattr(channel, "preset_id", None)
    effective = _instance_config(instance_key)

    channel_override: ScheduleOverride | None = None
    preset_override: ScheduleOverride | None = None
    try:
        overrides = _get_overrides()
        channel_override = await overrides.get_channel(chat_key)
        preset_override = await overrides.get_preset(preset_id)
    except Exception as exc:
        logger.warning("Cannot load schedule overrides for %s: %s", chat_key, exc)

    schedule = resolve_schedule(
        timezone=effective.TIMEZONE,
        sleep_time=effective.SLEEP_TIME,
        wake_time_start=effective.WAKE_TIME_START,
        wake_time_end=effective.WAKE_TIME_END,
        preset_override=preset_override,
        channel_override=channel_override,
    )
    for field in _instance_schedule_fields(instance_key):
        if schedule.sources.get(field) == "global":
            schedule.sources[field] = "instance"

    settings = ChatSettings(instance_key=instance_key, config=effective, schedule=schedule)
    _settings_cache[chat_key] = (time.monotonic(), settings)
    return settings


async def _resolve_schedule_for(chat_key: str) -> ResolvedSchedule:
    return (await _settings_for(chat_key)).schedule


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


def _make_config_snapshot(
    schedule: ResolvedSchedule | None = None,
    cfg: Any = None,
) -> Any:
    schedule = schedule or _global_schedule()
    cfg = cfg if cfg is not None else config
    return create_config_snapshot(
        timezone=schedule.timezone,
        sleep_time=schedule.sleep_time,
        wake_time_start=schedule.wake_time_start,
        wake_time_end=schedule.wake_time_end,
        wake_random_step_minutes=cfg.WAKE_RANDOM_STEP_MINUTES,
        near_wake_ratio=0.15,  # deprecated in schema v2, kept for rollback
        wake_confirm_window_seconds=cfg.WAKE_CONFIRM_WINDOW_SECONDS,
        history_mode=cfg.HISTORY_MODE,
        call_keywords=cfg.CALL_KEYWORDS,
        fallback_persona_name=cfg.FALLBACK_PERSONA_NAME,
        early_wake_idle_minutes=cfg.EARLY_WAKE_IDLE_MINUTES,
        quality_min=cfg.QUALITY_MIN,
        quality_max=cfg.QUALITY_MAX,
        quality_jitter_points=cfg.QUALITY_JITTER_POINTS,
        near_wake_minutes=cfg.NEAR_WAKE_MINUTES,
        sleep_target_hours=cfg.SLEEP_TARGET_HOURS,
        urgent_keywords=cfg.URGENT_KEYWORDS,
        answer_scope=cfg.ANSWER_SCOPE,
        max_offers_per_night=cfg.MAX_OFFERS_PER_NIGHT,
        offer_cooldown_minutes=cfg.OFFER_COOLDOWN_MINUTES,
        snooze_minutes=cfg.SNOOZE_MINUTES,
        asleep_prompt=cfg.WAKE_PROMPT_ASLEEP,
        near_wake_prompt=cfg.WAKE_PROMPT_NEAR_WAKE,
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
    history_mode = (await _settings_for(chat_key)).config.HISTORY_MODE
    from_own_bot = await _is_own_bot(message)

    async def _process(state: ChatSleepState) -> ChatSleepState:
        nonlocal _result_signal, _result_action
        state.last_seen_at = now_utc

        if from_own_bot:
            # Another instance of ours talking in a shared group. Record it,
            # but never let it call, answer, or keep the bot awake.
            logger.debug("Ignoring own-bot message in %s", chat_key)
            if state.status == SleepStatus.AWAKE:
                _result_signal = MsgSignal.CONTINUE
            else:
                _result_signal = (
                    MsgSignal.BLOCK_ALL
                    if history_mode == "strict"
                    else MsgSignal.BLOCK_TRIGGER
                )
            return state

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
                    if history_mode == "strict"
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
                record=history_mode != "strict",
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

    cfg = (await _settings_for(chat_key)).config
    if cfg.NIGHT_TIMER_POLICY == "block":
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
    cfg = (await _settings_for(chat_key)).config
    minutes = max(0, min(120, cfg.NIGHT_DUTY_ASSUMED_MINUTES))
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


@plugin.mount_collect_methods()
async def collect_available_methods(_ctx: AgentCtx) -> list:
    """`resume_sleep` only means anything while the bot is up before its alarm.

    Registered unconditionally it sat in every prompt around the clock, and the
    model could call it at three in the afternoon and get a ValueError back.
    """
    if not config.ENABLED or _store is None:
        return []

    state = _store.get_cached(_ctx.chat_key)
    if state is None or state.status != SleepStatus.AWAKE_EARLY:
        return []
    if state.cycle is not None and _utcnow() >= state.cycle.planned_wake_at:
        return []
    return [resume_sleep_tool]


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

            for chat_key in list(store.known_chat_keys()):
                try:
                    await _maintain_chat(store, chat_key, now_utc)
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
) -> None:
    """Run maintenance checks for a single chat_key."""
    state = store.get_cached(chat_key)
    if state is None:
        return

    if state.status == SleepStatus.AWAKE:
        await _check_sleep_transition(store, chat_key, now_utc)

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
    settings = await _settings_for(chat_key)
    schedule = settings.schedule
    try:
        tz = ZoneInfo(schedule.timezone)
    except Exception as exc:
        logger.error("Invalid timezone %r for %s: %s", schedule.timezone, chat_key, exc)
        return

    sleep_date = find_sleep_date_for_now(
        now_utc,
        tz,
        schedule.sleep_time,
        schedule.wake_time_start,
        schedule.wake_time_end,
    )
    if sleep_date is None:
        return

    sleep_date_iso = sleep_date.isoformat()
    cached = store.get_cached(chat_key)
    if cached is not None and cached.cycle is not None and cached.cycle.sleep_date == sleep_date_iso:
        return
    if cached is not None and cached.skip_sleep_date == sleep_date_iso:
        logger.info("Staying up tonight for %s (skip requested)", chat_key)
        return

    snap = _make_config_snapshot(schedule, settings.config)
    segment_open_at = now_utc
    if backdate_segment:
        try:
            bedtime, _ws, _we = compute_cycle_boundaries(
                sleep_date,
                tz,
                schedule.sleep_time,
                schedule.wake_time_start,
                schedule.wake_time_end,
            )
            segment_open_at = min(bedtime, now_utc)
        except Exception as exc:
            logger.warning("Cannot backdate bedtime for %s: %s", chat_key, exc)

    async def _sleep(s: ChatSleepState) -> ChatSleepState:
        if s.status != SleepStatus.AWAKE:
            return s
        if s.cycle is not None and s.cycle.sleep_date == sleep_date_iso:
            return s
        if s.skip_sleep_date == sleep_date_iso:
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

    state = await store.with_state(chat_key, _sleep)
    if state.status == SleepStatus.ASLEEP and state.cycle is not None:
        logger.info(
            "%s 入睡（%s），计划 %s 起床",
            chat_key,
            _fmt_local(state.cycle.sleep_at, schedule.timezone),
            _fmt_local(state.cycle.planned_wake_at, schedule.timezone),
        )


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
    settings = await _settings_for(chat_key)
    cfg = settings.config
    ctx: AgentCtx | None = None
    persona_name = cfg.FALLBACK_PERSONA_NAME
    try:
        ctx = await AgentCtx.create_by_chat_key(chat_key)
        persona_name = await _get_persona_name(ctx)
    except Exception as exc:
        logger.warning("Cannot build ctx for %s, using fallback name: %s", chat_key, exc)

    notice_action: ActionSendWakeNotice | None = None
    breakdown: dict[str, float] | None = None
    policy_used: str = ""

    def _score(cycle: SleepCycle, seconds: float) -> int:
        nonlocal breakdown
        detail = compute_quality_detail(cycle, seconds)
        breakdown = detail.as_dict()
        return detail.score

    async def _wake(s: ChatSleepState) -> ChatSleepState:
        nonlocal notice_action
        policy = cfg.WAKE_NOTICE_POLICY
        if s.cycle is not None:
            late_seconds = (now_utc - s.cycle.planned_wake_at).total_seconds()
            grace_seconds = max(0, cfg.WAKE_NOTICE_GRACE_MINUTES) * 60
            if late_seconds > grace_seconds:
                logger.info(
                    "Wake-up for %s is %.0f min late (grace %d min); settling silently",
                    chat_key,
                    late_seconds / 60,
                    cfg.WAKE_NOTICE_GRACE_MINUTES,
                )
                policy = "never"
        nonlocal policy_used
        policy_used = policy
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
        logger.info("%s 自然醒，未播报（策略 %s）", chat_key, policy_used or cfg.WAKE_NOTICE_POLICY)
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
        logger.info("%s 已起床并播报：%s", chat_key, notice_action.text)
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
        await _check_sleep_transition(store, chat_key, now_utc, backdate_segment=True)


async def _boot_reconcile(store: SleepStateStore, now_utc: datetime) -> None:
    """Hydrate every known channel and settle whatever the downtime skipped."""
    chat_keys = await _discover_chat_keys()
    logger.info("启动对账：%d 个频道", len(chat_keys))

    for chat_key in sorted(chat_keys):
        try:
            await store.hydrate(chat_key)
            await _reconcile_chat(store, chat_key, now_utc)
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

    logger.info("夜间定时任务策略：%s", config.NIGHT_TIMER_POLICY)
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
    global _store, _overrides, _maintenance_task

    _install_log_bridge()
    logger.info("自动睡眠插件启动中")

    _store = SleepStateStore(plugin.store)
    _overrides = ScheduleOverrideStore(plugin.store)
    invalidate_schedule_cache()
    invalidate_own_accounts_cache()

    if not _install_wraps():
        logger.error("Some runtime wraps failed to install; plugin may not fully function")

    for warning in collect_upgrade_warnings():
        logger.warning("配置提醒：%s", warning)

    if config.ENABLED:
        try:
            await _boot_reconcile(_store, _utcnow())
        except Exception as exc:
            logger.error("Boot reconciliation aborted: %s", exc)

    _maintenance_task = asyncio.create_task(_maintenance_loop())
    logger.info(
        "自动睡眠已就绪：作息 %s → %s~%s（%s），起床播报 %s",
        config.SLEEP_TIME,
        config.WAKE_TIME_START,
        config.WAKE_TIME_END,
        config.TIMEZONE,
        config.WAKE_NOTICE_POLICY,
    )


@plugin.mount_cleanup_method()
async def cleanup() -> None:
    global _store, _overrides, _maintenance_task

    if _maintenance_task is not None:
        _maintenance_task.cancel()
        try:
            await _maintenance_task
        except asyncio.CancelledError:
            pass
        _maintenance_task = None

    _uninstall_wraps()

    invalidate_schedule_cache()
    _remove_log_bridge()
    _overrides = None

    if _store:
        _store.clear_all()
        _store = None

    logger.info("Auto-sleep plugin cleaned up")


# ---------------------------------------------------------------------------
# Operator commands
# ---------------------------------------------------------------------------

try:
    from nekro_agent.services.command.base import CommandPermission
    from nekro_agent.services.command.ctl import CmdCtl
    from nekro_agent.services.command.schemas import Arg, CommandExecutionContext

    _COMMANDS_AVAILABLE = True
except ImportError:  # pragma: no cover - hosts without the command system
    _COMMANDS_AVAILABLE = False
    logger.info("Command system unavailable; /sleep commands not registered")


def _fmt_duration(seconds: float) -> str:
    from .engine import format_sleep_duration

    return format_sleep_duration(max(0.0, seconds))


def _render_status(
    state: ChatSleepState | None,
    schedule: ResolvedSchedule,
    persona_name: str,
    now_utc: datetime,
) -> str:
    """Everything an operator needs to answer "why did it do that".

    Includes last night's scoring terms, because a percentage nobody can take
    apart is exactly how the previous version hid a bug for a whole release.
    """
    tz_name = schedule.timezone
    marks = {
        "global": "全局",
        "preset": "人设",
        "channel": "本频道",
    }
    origin = "、".join(
        f"{field}={marks.get(src, src)}"
        for field, src in schedule.sources.items()
        if src != "global"
    )
    lines = [
        f"【{persona_name} · 睡眠状态】",
        f"作息：{schedule.sleep_time} → {schedule.wake_time_start}~{schedule.wake_time_end}"
        f"（{tz_name}）" + (f"　覆盖：{origin}" if origin else ""),
    ]

    if state is None:
        lines.append("当前：尚未装载（本频道还没有睡眠记录）")
        return "\n".join(lines)

    status_text = {
        SleepStatus.AWAKE: "醒着",
        SleepStatus.ASLEEP: "睡着",
        SleepStatus.AWAKE_EARLY: "被提前叫醒",
    }.get(state.status, str(state.status))

    cycle = state.cycle
    if cycle is not None and state.status != SleepStatus.AWAKE:
        slept = _fmt_duration(
            sum(
                ((seg.close_at or now_utc) - seg.open_at).total_seconds()
                for seg in cycle.sleep_segments
            )
        )
        lines.append(
            f"当前：{status_text}（{_fmt_local(cycle.sleep_at, tz_name)} 就寝，"
            f"计划 {_fmt_local(cycle.planned_wake_at, tz_name)} 起床，已睡 {slept}）"
        )
    else:
        lines.append(f"当前：{status_text}")

    for warning in collect_upgrade_warnings():
        lines.append(f"⚠️ {warning}")

    if state.skip_sleep_date:
        lines.append(f"今夜：已设置不睡（{state.skip_sleep_date}）")
    elif cycle is not None:
        snooze = (
            f"，静默至 {_fmt_local(state.snooze_until, tz_name)}"
            if state.snooze_until and state.snooze_until > now_utc
            else ""
        )
        lines.append(
            f"今夜：已提示 {state.offers_sent_tonight}/"
            f"{cycle.config_snapshot.max_offers_per_night} 次{snooze}"
        )

    if cycle is not None:
        breakdown = cycle.quality_breakdown
        if breakdown:
            raw = breakdown.get("raw", 0.0)
            score = int(breakdown.get("score", 0))
            clipped = (
                f"（原始分 {raw:.1f}，被下限 {config.QUALITY_MIN} 夹住）"
                if raw < config.QUALITY_MIN - 0.5
                else ""
            )
            lines.append(
                f"上一夜：{score}%{clipped}，"
                f"睡了 {_fmt_duration(breakdown.get('effective_hours', 0) * 3600)}"
            )
            lines.append(
                "　　基准 {target:.2f}h · 覆盖 {base:.1f} · 碎片 -{frag:.1f} · "
                "呼叫 -{calls:.1f} · 叫醒 -{wakes:.1f} · 整夜无扰 +{bonus:.1f} · "
                "扰动 {jitter:+.1f}".format(
                    target=breakdown.get("target_hours", 0),
                    base=breakdown.get("base", 0),
                    frag=breakdown.get("penalty_fragmentation", 0),
                    calls=breakdown.get("penalty_calls", 0),
                    wakes=breakdown.get("penalty_wakes", 0),
                    bonus=breakdown.get("bonus_clean_night", 0),
                    jitter=breakdown.get("jitter", 0),
                )
            )
        calls = sum(1 for a in cycle.wake_attempts if not a.is_confirmed)
        wakes = sum(1 for a in cycle.wake_attempts if a.is_confirmed)
        lines.append(f"本夜记录：没理会的呼叫 {calls} 次，真被叫醒 {wakes} 次")

    return "\n".join(lines)


_OVERRIDE_FIELDS = {
    "tz": "timezone",
    "timezone": "timezone",
    "bed": "sleep_time",
    "sleep": "sleep_time",
    "wake-start": "wake_time_start",
    "wake-end": "wake_time_end",
}


def _validate_override(field: str, value: str) -> str | None:
    """Return an error message, or None when the value is usable."""
    if field == "timezone":
        try:
            ZoneInfo(value)
        except Exception:
            return f"时区 {value!r} 无法识别，需要 IANA 名称，例如 Asia/Tokyo"
        return None
    try:
        from .schedule import parse_hhmm

        parse_hhmm(value)
    except Exception:
        return f"{value!r} 不是 HH:MM 格式"
    return None


async def _current_sleep_date(chat_key: str, now_utc: datetime) -> str:
    """The local date of the night `now` belongs to (tonight, if it is daytime)."""
    schedule = await _resolve_schedule_for(chat_key)
    tz = ZoneInfo(schedule.timezone)
    found = find_sleep_date_for_now(
        now_utc, tz, schedule.sleep_time, schedule.wake_time_start, schedule.wake_time_end
    )
    if found is not None:
        return found.isoformat()
    return now_utc.astimezone(tz).date().isoformat()


if _COMMANDS_AVAILABLE:
    sleep_group = plugin.mount_command_group(
        name="sleep",
        description="自动睡眠：查看状态、临时调整作息",
        permission=CommandPermission.SUPER_USER,
        category="plugin",
    )

    @sleep_group.command(
        name="status",
        description="查看当前睡眠状态与上一夜的评分明细",
        permission=CommandPermission.ADVANCED,
    )
    async def sleep_status_command(context: CommandExecutionContext):
        chat_key = context.chat_key
        store = _get_store()
        state = store.get_cached(chat_key) or await store.hydrate(chat_key)
        schedule = await _resolve_schedule_for(chat_key)
        persona_name = config.FALLBACK_PERSONA_NAME
        try:
            ctx = await AgentCtx.create_by_chat_key(chat_key)
            persona_name = await _get_persona_name(ctx)
        except Exception as exc:
            logger.debug("Cannot resolve persona for status: %s", exc)
        return CmdCtl.success(_render_status(state, schedule, persona_name, _utcnow()))

    @sleep_group.command(name="now", description="立刻入睡")
    async def sleep_now_command(context: CommandExecutionContext):
        chat_key = context.chat_key
        store = _get_store()
        await store.hydrate(chat_key)
        now_utc = _utcnow()
        settings = await _settings_for(chat_key)
        schedule = settings.schedule
        sleep_date = await _current_sleep_date(chat_key, now_utc)
        snap = _make_config_snapshot(schedule, settings.config)

        async def _sleep(state: ChatSleepState) -> ChatSleepState:
            if state.status != SleepStatus.AWAKE:
                return state
            return transition_to_sleep(
                state,
                now_utc,
                snap,
                sleep_date_local=sleep_date,
                sleep_at_override=now_utc,
            )

        state = await store.with_state(chat_key, _sleep)
        if state.status != SleepStatus.ASLEEP:
            return CmdCtl.failed(f"当前状态是 {state.status.value}，不能直接入睡")
        wake_at = state.cycle.planned_wake_at if state.cycle else None
        when = _fmt_local(wake_at, schedule.timezone) if wake_at else "?"
        return CmdCtl.success(f"已入睡，计划 {when} 自然醒")

    @sleep_group.command(name="wake", description="立刻起床并结算这一夜")
    async def sleep_wake_command(context: CommandExecutionContext):
        chat_key = context.chat_key
        store = _get_store()
        state = store.get_cached(chat_key) or await store.hydrate(chat_key)
        if state.status == SleepStatus.AWAKE:
            return CmdCtl.failed("本来就是醒着的")
        await _settle_wake(store, chat_key, _utcnow())
        return CmdCtl.success("已起床并结算")

    @sleep_group.command(name="skip", description="今晚不入睡")
    async def sleep_skip_command(context: CommandExecutionContext):
        chat_key = context.chat_key
        store = _get_store()
        await store.hydrate(chat_key)
        sleep_date = await _current_sleep_date(chat_key, _utcnow())

        async def _skip(state: ChatSleepState) -> ChatSleepState:
            return state.model_copy(update={"skip_sleep_date": sleep_date})

        await store.with_state(chat_key, _skip)
        return CmdCtl.success(f"今晚（{sleep_date}）不入睡，明晚恢复")

    @sleep_group.command(
        name="set",
        description="设置本频道或当前人设的作息覆盖",
        usage="/sleep set <tz|bed|wake-start|wake-end> <值> [scope=channel|preset]",
    )
    async def sleep_set_command(
        context: CommandExecutionContext,
        field: str = Arg("要改的字段", positional=True, choices=sorted(_OVERRIDE_FIELDS)),
        value: str = Arg("新的值", positional=True),
        scope: str = Arg("作用范围", default="channel", choices=["channel", "preset"]),
    ):
        key = _OVERRIDE_FIELDS.get(field)
        if key is None:
            return CmdCtl.failed(f"未知字段 {field!r}，可用：{'、'.join(sorted(_OVERRIDE_FIELDS))}")
        error = _validate_override(key, value)
        if error:
            return CmdCtl.failed(error)

        chat_key = context.chat_key
        overrides = _get_overrides()
        if scope == "preset":
            preset_id = (await _channel_and_preset(chat_key))[1]
            if preset_id is None:
                return CmdCtl.failed("取不到当前人设，无法按人设设置")
            current = await overrides.get_preset(preset_id) or ScheduleOverride()
            await overrides.set_preset(preset_id, current.model_copy(update={key: value}))
            invalidate_schedule_cache()
            return CmdCtl.success(f"已为人设 {preset_id} 设置 {field} = {value}")

        current = await overrides.get_channel(chat_key) or ScheduleOverride()
        await overrides.set_channel(chat_key, current.model_copy(update={key: value}))
        invalidate_schedule_cache(chat_key)
        return CmdCtl.success(f"已为本频道设置 {field} = {value}")

    @sleep_group.command(
        name="unset",
        description="清除作息覆盖",
        usage="/sleep unset [scope=channel|preset]",
    )
    async def sleep_unset_command(
        context: CommandExecutionContext,
        scope: str = Arg("作用范围", default="channel", choices=["channel", "preset"]),
    ):
        chat_key = context.chat_key
        overrides = _get_overrides()
        if scope == "preset":
            preset_id = (await _channel_and_preset(chat_key))[1]
            if preset_id is None:
                return CmdCtl.failed("取不到当前人设")
            await overrides.set_preset(preset_id, ScheduleOverride())
            invalidate_schedule_cache()
            return CmdCtl.success(f"已清除人设 {preset_id} 的作息覆盖")

        await overrides.set_channel(chat_key, ScheduleOverride())
        invalidate_schedule_cache(chat_key)
        return CmdCtl.success("已清除本频道的作息覆盖")
