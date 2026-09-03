"""NekroAgent Auto-Sleep Plugin

Provides per-chat_key humanized sleep cycles without modifying NekroAgent core.
Compatible with both Akiyo and upstream versions via runtime capability probing.

Plugin key: Akiyo_dayo.nekro_auto_sleep
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

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
    ActionNone,
    ActionSendFixed,
    ActionSendResumeSleep,
    ActionSendWakeNotice,
    close_sleep_segment,
    close_timer_interval,
    compute_actual_sleep_seconds,
    handle_idle_sleep_back,
    handle_resume_sleep,
    handle_valid_call_while_asleep,
    has_active_timer_lease,
    is_idle_expired,
    mark_notice_failed,
    mark_notice_sent,
    open_sleep_segment,
    open_timer_interval,
    refresh_idle_deadline,
    settle_natural_wake,
    transition_to_awake,
    transition_to_sleep,
)
from .models import (
    PLUGIN_KEY,
    ChatSleepState,
    NotificationStatus,
    SleepStatus,
    SourceType,
)
from .persistence import DATA_KEY, SleepStateStore
from .quality import (
    compute_quality,
    compute_streak_note,
    pick_dream,
    quality_tier,
    stable_pick,
)
from .runtime import (
    ChatKeyLocks,
    LeaseLedger,
    chat_key_locks,
    current_source,
    lease_ledger,
    make_run_agent_task_wrapper,
    make_schedule_agent_task_wrapper,
    make_timer_task_wrapper,
    unwrap_callable,
    wrap_callable,
)
from .schedule import (
    compute_cycle_boundaries,
    create_config_snapshot,
    current_local_date,
    find_sleep_date_for_now,
)

logger = logging.getLogger("nekro_auto_sleep")

# ---------------------------------------------------------------------------
# Plugin instance
# ---------------------------------------------------------------------------

plugin = NekroPlugin(
    name="自动睡眠",
    module_name="nekro_auto_sleep",
    description="为每个会话提供独立的拟人化睡眠周期：叫醒协议、睡眠质量评分、梦境播报与连续打卡",
    version="1.1.2",
    author="Akiyo_dayo",
    url="https://github.com/Akiyo-dayo/NekroAgent_ByAkiyo",
    allow_sleep=True,
    i18n_name=i18n.i18n_text(
        zh_CN="自动睡眠",
        en_US="Auto Sleep",
    ),
    i18n_description=i18n.i18n_text(
        zh_CN="为每个会话提供独立的拟人化睡眠周期：叫醒协议、睡眠质量评分、梦境播报与连续打卡",
        en_US="Per-chat humanized sleep cycles with wake protocol, quality scoring, dream reports and streaks",
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
    NEAR_WAKE_RATIO: float = Field(
        default=0.15,
        title="接近起床判定比例",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="接近起床判定比例", en_US="Near Wake Ratio"),
            i18n_description=i18n.i18n_text(
                zh_CN="睡眠区间末尾的比例，在此范围内提示语改为「还没起床」，0-0.5",
                en_US="Ratio of sleep window end; within this range the prompt changes to 'not yet awake', 0-0.5",
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
    TIMER_AGENT_WAIT_TIMEOUT_SECONDS: int = Field(
        default=900,
        title="定时任务等待超时（秒）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="定时任务等待超时（秒）", en_US="Timer Agent Wait Timeout (s)"),
            i18n_description=i18n.i18n_text(
                zh_CN="插件等待定时任务完成的最大时间，超时后清理租约但不取消核心任务，30-7200",
                en_US="Maximum time the plugin waits for a timer task to complete; lease is cleaned on timeout but core task is not cancelled, 30-7200",
            ),
        ).model_dump(),
    )
    QUALITY_MIN: int = Field(
        default=60,
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
        default=4.0,
        title="质量稳定扰动幅度",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="质量稳定扰动幅度", en_US="Quality Jitter Points"),
            i18n_description=i18n.i18n_text(
                zh_CN="每次睡眠质量的随机扰动范围（正负），0-15",
                en_US="Random jitter range (plus/minus) applied to each sleep quality score, 0-15",
            ),
        ).model_dump(),
    )
    DREAM_ENABLED: bool = Field(
        default=True,
        title="晨间梦境播报",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="晨间梦境播报", en_US="Dream Report"),
            i18n_description=i18n.i18n_text(
                zh_CN="自然醒报告中附带一条当夜梦境（睡得越差越容易做噩梦）",
                en_US="Include a dream line in the morning report; worse sleep tends to nightmares",
            ),
        ).model_dump(),
    )
    BEDTIME_CHANCE: float = Field(
        default=0.35,
        title="晚安消息概率",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="晚安消息概率", en_US="Bedtime Greeting Chance"),
            i18n_description=i18n.i18n_text(
                zh_CN="入睡时主动说晚安的概率，0-1；每晚以固定随机种子判定，重启不会重复发送",
                en_US="Chance of saying goodnight when falling asleep, 0-1; decided per night with a fixed seed so restarts never duplicate",
            ),
        ).model_dump(),
    )
    WAKE_NOTICE_ALWAYS: bool = Field(
        default=True,
        title="无打扰也发晨间报告",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="无打扰也发晨间报告", en_US="Morning Report Always"),
            i18n_description=i18n.i18n_text(
                zh_CN="开启后即使夜里没人叫醒也发送晨间睡眠报告；关闭则仅在被叫醒过的夜晚发送",
                en_US="Send the morning report even when nobody tried to wake the bot; disable to report only after wake attempts",
            ),
        ).model_dump(),
    )
    QUALITY_HISTORY_DAYS: int = Field(
        default=14,
        title="质量历史保留天数",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="质量历史保留天数", en_US="Quality History Days"),
            i18n_description=i18n.i18n_text(
                zh_CN="每个会话保留的每日睡眠质量记录条数，用于连续打卡与趋势，1-90",
                en_US="Per-chat days of settled quality kept for streaks and trends, 1-90",
            ),
        ).model_dump(),
    )


config = plugin.get_config(SleepConfig)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_store: SleepStateStore | None = None
_maintenance_task: asyncio.Task[None] | None = None
_wake_inject_cache: dict[str, str] = {}
_installed_wraps: list[tuple[Any, str]] = []


def _get_store() -> SleepStateStore:
    assert _store is not None, "Plugin not initialized"
    return _store


def _check_install_dir_name() -> None:
    """Fail loudly when the install directory name diverges from ``module_name``.

    The framework's unload/reload path keys everything off the declared
    ``module_name`` (``loaded_module_names`` cleanup, ``sys.modules`` reuse,
    ``reload_plugin_by_module_name`` path building). With a mismatched
    directory name the module registry and sys.modules entry are left behind,
    which surfaces as a plugin that cannot be unloaded or hot-reloaded.
    """
    try:
        dir_name = Path(__file__).resolve().parent.name
        if dir_name != plugin.module_name:
            logger.error(
                "自动睡眠插件安装目录名 `%s` 与 module_name `%s` 不一致！"
                "框架按 module_name 卸载/重载，不一致会导致插件无法正常卸载或热重载。"
                "请把目录改名为 `%s` 后重启 NekroAgent。",
                dir_name,
                plugin.module_name,
                plugin.module_name,
            )
    except Exception as exc:  # noqa: BLE001 - self-check must never break init
        logger.warning("Install directory self-check failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_persona_name(ctx: AgentCtx) -> str:
    """Get persona name, falling back to config default."""
    try:
        db_channel = await ctx.db_chat_channel
        preset = await db_channel.get_preset()
        if preset and hasattr(preset, "name") and preset.name:
            return preset.name
    except Exception:
        pass
    return config.FALLBACK_PERSONA_NAME


def _parse_keywords() -> list[str]:
    raw = config.CALL_KEYWORDS
    return [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]


def _is_valid_call(message: ChatMessage, persona_name: str) -> bool:
    """Check if a message constitutes a valid wake-up call (spec §6.1)."""
    if hasattr(message, "channel_type") and message.channel_type == "private":
        return True
    if hasattr(message, "is_tome") and message.is_tome:
        return True
    text = message.content_text if hasattr(message, "content_text") else str(message)
    if persona_name and persona_name in text:
        return True
    keywords = _parse_keywords()
    for kw in keywords:
        if kw in text:
            return True
    return bool(_is_reply_to_bot(message))


def _is_reply_to_bot(message: ChatMessage) -> bool:
    """Conservative reply-to-bot detection (spec §6.1)."""
    if not hasattr(message, "ext_data") or not message.ext_data:
        return False
    ed = message.ext_data
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


def _has_permission(chat_key: str) -> bool:
    """Check if there's an active lease or contextvar permission."""
    src = current_source.get()
    if src in (SourceType.USER_WAKE_CONFIRM, SourceType.USER_DIRECT,
               SourceType.TIMER_ONESHOT, SourceType.TIMER_RECURRING,
               SourceType.INTERNAL_WAKE_NOTICE):
        return True
    return lease_ledger.has_active_for_chat(chat_key)


def _make_config_snapshot() -> Any:
    return create_config_snapshot(
        timezone=config.TIMEZONE,
        sleep_time=config.SLEEP_TIME,
        wake_time_start=config.WAKE_TIME_START,
        wake_time_end=config.WAKE_TIME_END,
        wake_random_step_minutes=config.WAKE_RANDOM_STEP_MINUTES,
        near_wake_ratio=config.NEAR_WAKE_RATIO,
        wake_confirm_window_seconds=config.WAKE_CONFIRM_WINDOW_SECONDS,
        history_mode=config.HISTORY_MODE,
        call_keywords=config.CALL_KEYWORDS,
        fallback_persona_name=config.FALLBACK_PERSONA_NAME,
        early_wake_idle_minutes=config.EARLY_WAKE_IDLE_MINUTES,
        quality_min=config.QUALITY_MIN,
        quality_max=config.QUALITY_MAX,
        quality_jitter_points=config.QUALITY_JITTER_POINTS,
    )


# ---------------------------------------------------------------------------
# Fun layer: bedtime greetings
# ---------------------------------------------------------------------------

BEDTIME_GREETS: tuple[str, ...] = (
    "夜深了，我先去睡啦，晚安～",
    "今天的营业时间到此结束，我去睡了，晚安！",
    "zzZ… 不聊了不聊了，我先睡为敬，晚安～",
    "眼睛已经开始打架了，我去睡了，晚安各位。",
    "晚安！梦里见（如果你们也来梦里的话）。",
)


def _deterministic_hit(seed: str, chance: float) -> bool:
    """Stable per-seed probability so a given night never flips on restart."""
    import hashlib

    if chance <= 0:
        return False
    h = hashlib.sha256(seed.encode()).digest()
    return (h[0] / 255.0) < max(0.0, min(1.0, chance))


async def _send_quiet_text(chat_key: str, text: str) -> None:
    """Send a message without recording it into chat history."""
    try:
        ctx = await AgentCtx.create_by_chat_key(chat_key)
    except Exception as exc:
        logger.warning("Cannot create ctx for %s: %s", chat_key, exc)
        return
    token = current_source.set(SourceType.INTERNAL_WAKE_NOTICE)
    try:
        await ctx.send_text(text, record=False)
    finally:
        current_source.reset(token)


async def _maybe_send_bedtime(chat_key: str, sleep_date: str) -> None:
    """Say goodnight with a stable per-night chance when falling asleep."""
    if not _deterministic_hit(f"bedtime:{chat_key}:{sleep_date}", config.BEDTIME_CHANCE):
        return
    text = stable_pick(f"bedtime-text:{chat_key}:{sleep_date}", BEDTIME_GREETS)
    try:
        await _send_quiet_text(chat_key, text)
    except Exception as exc:
        logger.warning("Failed to send bedtime greeting for %s: %s", chat_key, exc)


# ---------------------------------------------------------------------------
# Timer task hooks: record intervals + leases while a timer runs during sleep
# ---------------------------------------------------------------------------


async def _on_timer_task_start(chat_key: str, task_id: str) -> None:
    store = _get_store()
    now_utc = datetime.now(ZoneInfo("UTC"))

    async def _open(s: ChatSleepState) -> ChatSleepState:
        return open_timer_interval(s, task_id, now_utc)

    try:
        await store.with_state(chat_key, _open)
    except Exception as exc:  # noqa: BLE001 - timer flow must never break
        logger.warning("Failed to open timer interval for %s: %s", chat_key, exc)


async def _on_timer_task_end(chat_key: str, task_id: str) -> None:
    store = _get_store()
    now_utc = datetime.now(ZoneInfo("UTC"))

    async def _close(s: ChatSleepState) -> ChatSleepState:
        return close_timer_interval(s, task_id, now_utc)

    try:
        await store.with_state(chat_key, _close)
    except Exception as exc:  # noqa: BLE001 - timer flow must never break
        logger.warning("Failed to close timer interval for %s: %s", chat_key, exc)


def _compensate_sleep_if_due(
    state: ChatSleepState,
    now_utc: datetime,
    tz: ZoneInfo,
) -> ChatSleepState:
    """Enter the currently active sleep cycle even after a delayed check/reload."""
    if state.status != SleepStatus.AWAKE:
        return state

    snapshot = _make_config_snapshot()
    sleep_date = find_sleep_date_for_now(
        now_utc,
        tz,
        snapshot.sleep_time,
        snapshot.wake_time_start,
        snapshot.wake_time_end,
    )
    if sleep_date is None:
        return state

    candidate = transition_to_sleep(
        state,
        now_utc,
        snapshot,
        sleep_date_local=sleep_date.isoformat(),
    )
    if candidate.cycle is None or now_utc >= candidate.cycle.planned_wake_at:
        return state
    return candidate


# ---------------------------------------------------------------------------
# Message hooks
# ---------------------------------------------------------------------------


@plugin.mount_on_user_message()
async def on_user_message(ctx: AgentCtx, message: ChatMessage, *_args: Any, **_kwargs: Any) -> MsgSignal | None:
    # *_args/**_kwargs: upstream 2.4+ may extend hook signatures; extra
    # parameters are accepted and ignored so a signature change can never
    # TypeError through the framework's un-guarded dispatch loop.
    if not config.ENABLED:
        return MsgSignal.CONTINUE

    chat_key = ctx.chat_key
    store = _get_store()
    now_utc = datetime.now(ZoneInfo("UTC"))
    persona_name = await _get_persona_name(ctx)
    user_id = _get_user_id(message)

    async def _process(state: ChatSleepState) -> ChatSleepState:
        nonlocal _result_signal, _result_action, _went_to_sleep, _sleep_date
        state.last_seen_at = now_utc

        if state.status == SleepStatus.AWAKE:
            state = _compensate_sleep_if_due(
                state,
                now_utc,
                ZoneInfo(config.TIMEZONE),
            )
            if state.status == SleepStatus.AWAKE:
                _result_signal = MsgSignal.CONTINUE
                return state
            # compensate just put this chat to sleep
            _went_to_sleep = True
            if state.cycle is not None:
                _sleep_date = state.cycle.sleep_date

        if state.status == SleepStatus.AWAKE_EARLY:
            state = refresh_idle_deadline(state, now_utc)
            _result_signal = MsgSignal.CONTINUE
            return state

        if state.status == SleepStatus.ASLEEP:
            if not _is_valid_call(message, persona_name):
                _result_signal = MsgSignal.BLOCK_ALL
                return state

            state, action = handle_valid_call_while_asleep(
                state, now_utc, user_id, persona_name
            )
            _result_action = action

            if isinstance(action, ActionForceWake):
                _wake_inject_cache[chat_key] = action.inject_text
                _result_signal = MsgSignal.FORCE_TRIGGER
            elif isinstance(action, ActionSendFixed):
                if action.block_mode == "strict":
                    _result_signal = MsgSignal.BLOCK_ALL
                else:
                    _result_signal = MsgSignal.BLOCK_TRIGGER
            else:
                _result_signal = MsgSignal.CONTINUE

            return state

        _result_signal = MsgSignal.CONTINUE
        return state

    _result_signal: MsgSignal = MsgSignal.CONTINUE
    _result_action: Any = ActionNone()
    _went_to_sleep: bool = False
    _sleep_date: str = ""

    await store.with_state(chat_key, _process)

    if isinstance(_result_action, ActionSendFixed):
        try:
            await ctx.send_text(_result_action.text, record=False)
        except Exception as exc:
            logger.error("Failed to send wake offer: %s", exc)

    if _went_to_sleep and _sleep_date:
        await _maybe_send_bedtime(chat_key, _sleep_date)

    return _result_signal


@plugin.mount_on_system_message()
async def on_system_message(ctx: AgentCtx, message: str, *_args: Any, **_kwargs: Any) -> MsgSignal | None:
    if not config.ENABLED:
        return MsgSignal.CONTINUE

    chat_key = ctx.chat_key
    store = _get_store()
    state = store.get_cached(chat_key)

    if state is None:
        return MsgSignal.CONTINUE

    if state.status != SleepStatus.ASLEEP:
        return MsgSignal.CONTINUE

    src = current_source.get()
    if src in (SourceType.TIMER_ONESHOT, SourceType.TIMER_RECURRING,
               SourceType.INTERNAL_WAKE_NOTICE):
        return MsgSignal.CONTINUE

    if lease_ledger.has_active_for_chat(chat_key):
        return MsgSignal.CONTINUE

    return MsgSignal.BLOCK_ALL


# ---------------------------------------------------------------------------
# Prompt injection (one-shot wake info)
# ---------------------------------------------------------------------------


@plugin.mount_prompt_inject_method(
    "sleep_status",
    "注入当前睡眠状态信息（仅在叫醒时瞬时注入）",
)
async def inject_sleep_status(ctx: AgentCtx, *_args: Any, **_kwargs: Any) -> str:
    chat_key = ctx.chat_key

    inject = _wake_inject_cache.pop(chat_key, None)
    if inject:
        return inject

    src = current_source.get()
    if src in (SourceType.TIMER_ONESHOT, SourceType.TIMER_RECURRING):
        return "当前为夜间定时任务执行，任务完成后将自动恢复睡眠。"

    return ""


# ---------------------------------------------------------------------------
# Sandbox method: resume_sleep
# ---------------------------------------------------------------------------


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    "resume_sleep",
    "主动重新进入睡眠（仅在被提前叫醒后、计划起床前可用）",
)
async def resume_sleep_tool(*call_args: Any, **call_kwargs: Any) -> str:
    """Transition this chat from AWAKE_EARLY back to ASLEEP before planned wake.

    The host normally passes ctx positionally; pulling it from either args or
    kwargs keeps the tool resilient to sandbox call-shape changes.
    """
    _ctx: AgentCtx = call_args[0] if call_args else call_kwargs.get("ctx")
    chat_key = _ctx.chat_key
    store = _get_store()
    now_utc = datetime.now(ZoneInfo("UTC"))
    persona_name = await _get_persona_name(_ctx)

    result_text = ""

    async def _process(state: ChatSleepState) -> ChatSleepState:
        nonlocal result_text
        state, action = handle_resume_sleep(state, now_utc, persona_name)
        if isinstance(action, ActionSendResumeSleep):
            try:
                await _ctx.send_text(action.text, record=False)
            except Exception as exc:
                logger.error("Failed to send resume sleep message: %s", exc)
        result_text = "ok"
        return state

    try:
        await store.with_state(chat_key, _process)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    return result_text


# ---------------------------------------------------------------------------
# Sandbox method: get_sleep_report
# ---------------------------------------------------------------------------


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    "get_sleep_report",
    "查询本会话当前的睡眠状态与最近的睡眠质量记录（近7天）",
)
async def get_sleep_report_tool(*call_args: Any, **call_kwargs: Any) -> str:
    """Report the chat's current sleep status and recent quality history."""
    _ctx: AgentCtx = call_args[0] if call_args else call_kwargs.get("ctx")
    chat_key = _ctx.chat_key
    store = _get_store()
    now_utc = datetime.now(ZoneInfo("UTC"))

    state = store.get_cached(chat_key)
    if state is None:
        async with store._get_lock(chat_key):  # noqa: SLF001 - read-through load
            state = await store.load_or_create(chat_key)

    if state.status == SleepStatus.ASLEEP and state.cycle is not None:
        status_line = (
            f"当前状态：睡眠中（计划 {state.cycle.planned_wake_at.astimezone(ZoneInfo(state.cycle.timezone)).strftime('%H:%M')} 左右自然醒）"
        )
    elif state.status == SleepStatus.ASLEEP:
        status_line = "当前状态：睡眠中"
    elif state.status == SleepStatus.AWAKE_EARLY:
        status_line = "当前状态：被提前叫醒、还没睡回（空闲后会自动睡回）"
    else:
        status_line = "当前状态：清醒"

    if not state.quality_history:
        return f"{status_line}。最近还没有睡眠质量记录。"

    lines = [status_line, "最近 7 天睡眠质量："]
    for date_str in sorted(state.quality_history, reverse=True)[:7]:
        q = state.quality_history[date_str]
        tier_name, _, _ = quality_tier(q)
        lines.append(f"- {date_str}：{q}%（{tier_name}）")
    return "\n".join(lines)


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

            now_utc = datetime.now(ZoneInfo("UTC"))
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
            if not has_active_timer_lease(state) and not lease_ledger.has_active_for_chat(chat_key):
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
) -> None:
    """Compensate AWAKE -> ASLEEP for the active scheduled sleep cycle."""
    before = store.get_cached(chat_key)
    was_awake = before is not None and before.status == SleepStatus.AWAKE

    async def _sleep(s: ChatSleepState) -> ChatSleepState:
        return _compensate_sleep_if_due(s, now_utc, tz)

    await store.with_state(chat_key, _sleep)

    after = store.get_cached(chat_key)
    if (
        was_awake
        and after is not None
        and after.status == SleepStatus.ASLEEP
        and after.cycle is not None
    ):
        await _maybe_send_bedtime(chat_key, after.cycle.sleep_date)


async def _settle_wake(
    store: SleepStateStore,
    chat_key: str,
    now_utc: datetime,
) -> None:
    """Settle natural wake-up, record quality history and send the morning report."""
    should_notify = False
    settled_quality: int = 0
    settled_duration: float = 0.0
    sleep_date: str = ""
    quality_history: dict[str, int] = {}

    async def _wake(s: ChatSleepState) -> ChatSleepState:
        nonlocal should_notify, settled_quality, settled_duration, sleep_date, quality_history
        if s.cycle is None:
            return transition_to_awake(s, now_utc)
        persona_name = s.cycle.config_snapshot.fallback_persona_name
        settled_duration = compute_actual_sleep_seconds(s.cycle)
        settled_quality = compute_quality(s.cycle, settled_duration)
        sleep_date = s.cycle.sleep_date
        new_state, action = settle_natural_wake(
            s, now_utc, persona_name, settled_quality,
            always_notice=config.WAKE_NOTICE_ALWAYS,
        )
        if isinstance(action, ActionSendWakeNotice):
            should_notify = True

        history = dict(s.quality_history)
        if sleep_date:
            history[sleep_date] = settled_quality
            keep = max(1, min(90, config.QUALITY_HISTORY_DAYS))
            if len(history) > keep:
                for old in sorted(history)[:-keep]:
                    history.pop(old, None)
        quality_history = history
        return new_state.model_copy(update={"quality_history": history})

    await store.with_state(chat_key, _wake)

    if not should_notify:
        return

    try:
        ctx = await AgentCtx.create_by_chat_key(chat_key)
        persona_name = await _get_persona_name(ctx)

        from .engine import format_sleep_duration
        duration_str = format_sleep_duration(settled_duration)

        tier_name, emoji, comment = quality_tier(settled_quality)
        lines = [
            f"【{persona_name}已起床：昨日睡眠质量 {settled_quality}%（{tier_name}），睡眠时长 {duration_str}】",
            f"{emoji} {comment}",
        ]
        if config.DREAM_ENABLED and sleep_date:
            dream = pick_dream(f"dream:{chat_key}:{sleep_date}", settled_quality)
            if dream:
                lines.append(f"🌙 {dream}")
        note = compute_streak_note(quality_history, sleep_date) if sleep_date else None
        if note:
            lines.append(f"📈 {note}")
        text = "\n".join(lines)

        token = current_source.set(SourceType.INTERNAL_WAKE_NOTICE)
        try:
            await ctx.send_text(text, record=False)
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


# ---------------------------------------------------------------------------
# Runtime wrapping (capability probing)
# ---------------------------------------------------------------------------


def _install_wraps() -> bool:
    """Install runtime wraps via capability probing (spec §2.3)."""
    success = True

    try:
        from nekro_agent.services.message_service import message_service as ms
        if ms is None:
            logger.error("message_service singleton not found")
            return False

        if hasattr(ms, "schedule_agent_task") and callable(ms.schedule_agent_task):
            wrapper = make_schedule_agent_task_wrapper(_is_sleeping, _has_permission)
            if wrap_callable(ms, "schedule_agent_task", wrapper):
                _installed_wraps.append((ms, "schedule_agent_task"))
        else:
            logger.error("schedule_agent_task not found on message_service")
            success = False

        if hasattr(ms, "_run_chat_agent_task") and callable(ms._run_chat_agent_task):
            async def _on_agent_start(chat_key: str) -> None:
                pass

            async def _on_agent_end(chat_key: str) -> None:
                store = _get_store()
                state = store.get_cached(chat_key)
                if state and state.status == SleepStatus.AWAKE_EARLY:
                    src = current_source.get()
                    if src == SourceType.USER_DIRECT or src == SourceType.USER_WAKE_CONFIRM:
                        async def _refresh(s: ChatSleepState) -> ChatSleepState:
                            return refresh_idle_deadline(s, datetime.now(ZoneInfo("UTC")))
                        await store.with_state(chat_key, _refresh)

            wrapper = make_run_agent_task_wrapper(
                _is_sleeping, _on_agent_start, _on_agent_end
            )
            if wrap_callable(ms, "_run_chat_agent_task", wrapper):
                _installed_wraps.append((ms, "_run_chat_agent_task"))
        else:
            logger.warning("_run_chat_agent_task not found, layer-3 gate unavailable")

    except ImportError as exc:
        logger.error("Cannot import message_service: %s", exc)
        return False

    try:
        from nekro_agent.services.timer.timer_service import timer_service as ts
        if ts and hasattr(ts, "_execute_task") and callable(ts._execute_task):
            wrapper = make_timer_task_wrapper(
                SourceType.TIMER_ONESHOT,
                on_task_start=_on_timer_task_start,
                on_task_end=_on_timer_task_end,
                lease_ttl_seconds=float(
                    max(60, config.TIMER_AGENT_WAIT_TIMEOUT_SECONDS)
                ),
            )
            if wrap_callable(ts, "_execute_task", wrapper):
                _installed_wraps.append((ts, "_execute_task"))
    except ImportError:
        logger.info("TimerService not available, timer wrapping skipped")

    try:
        from nekro_agent.services.timer.recurring_timer_service import recurring_timer_service as rts
        if rts and hasattr(rts, "_fire_job") and callable(rts._fire_job):
            wrapper = make_timer_task_wrapper(
                SourceType.TIMER_RECURRING,
                on_task_start=_on_timer_task_start,
                on_task_end=_on_timer_task_end,
                lease_ttl_seconds=float(
                    max(60, config.TIMER_AGENT_WAIT_TIMEOUT_SECONDS)
                ),
            )
            if wrap_callable(rts, "_fire_job", wrapper):
                _installed_wraps.append((rts, "_fire_job"))
    except ImportError:
        logger.info("RecurringTimerService not available, recurring timer wrapping skipped")

    return success


def _uninstall_wraps() -> None:
    """Restore all wrapped callables."""
    for obj, attr in reversed(_installed_wraps):
        unwrap_callable(obj, attr)
    _installed_wraps.clear()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def _discover_legacy_chat_keys() -> set[str]:
    """Enumerate pre-index ``state.v1`` rows through the host's shared ORM model."""
    from nekro_agent.models.db_plugin_data import DBPluginData

    rows = await DBPluginData.filter(
        plugin_key=plugin.key,
        data_key=DATA_KEY,
        target_user_id="",
    ).values_list("target_chat_key", flat=True)
    return {str(chat_key) for chat_key in rows if chat_key}


async def _start_runtime() -> None:
    """Start plugin runtime components idempotently."""
    global _store, _maintenance_task

    _check_install_dir_name()

    if _store is None:
        _store = SleepStateStore(plugin.store)
        await _store.initialize(_discover_legacy_chat_keys)

    if not _install_wraps():
        logger.error("Some runtime wraps failed to install; plugin may not fully function")

    if _maintenance_task is None or _maintenance_task.done():
        _maintenance_task = asyncio.create_task(_maintenance_loop())

    logger.info("Auto-sleep plugin runtime started")


async def _stop_runtime() -> None:
    """Stop plugin runtime components idempotently without deleting persisted state."""
    global _store, _maintenance_task

    task = _maintenance_task
    _maintenance_task = None
    if task is not None:
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("Maintenance task stopped with an error: %s", exc)

    _uninstall_wraps()
    lease_ledger.clear()
    chat_key_locks.clear()
    _wake_inject_cache.clear()

    if _store is not None:
        _store.clear_all()
        _store = None

    logger.info("Auto-sleep plugin runtime stopped")


@plugin.mount_init_method()
async def init(*_args: Any, **_kwargs: Any) -> None:
    await _start_runtime()


@plugin.on_enabled()
async def handle_enabled(*_args: Any, **_kwargs: Any) -> None:
    await _start_runtime()


@plugin.on_disabled()
async def handle_disabled(*_args: Any, **_kwargs: Any) -> None:
    await _stop_runtime()


@plugin.mount_cleanup_method()
async def cleanup(*_args: Any, **_kwargs: Any) -> None:
    await _stop_runtime()
