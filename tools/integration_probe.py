"""Integration probe for nekro_auto_sleep against a real NekroAgent install.

Runs inside the host venv and talks to the real database, but sends nothing:
every outbound path is replaced with a capture before the plugin is touched, and
the only channel it drives is a synthetic one created and deleted by this script.

Usage:  uv run python integration_probe.py
"""

import asyncio
import importlib.util
import sys
import traceback
from datetime import datetime, timedelta
import pathlib
from pathlib import Path
from zoneinfo import ZoneInfo

PLUGIN_PATH = Path("data/nekro_agent/plugins/workdir/nekro_auto_sleep/__init__.py")
TEST_CHAT_KEY = "onebot_v11-group_900000001"
UTC = ZoneInfo("UTC")

RESULTS: list[tuple[str, bool, str]] = []
SENT: list[tuple[str, str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"\n       {detail}" if detail else ""))


def _install_send_guards() -> None:
    """Nothing may reach an adapter, no matter which path the plugin takes."""
    from nekro_agent.schemas.agent_ctx import AgentCtx

    async def _capture_ctx(self, content: str, *, record: bool = True, **kw):
        SENT.append((self.chat_key, content, record))

    AgentCtx.send_text = _capture_ctx  # type: ignore[assignment]

    from nekro_agent.api import message as message_api

    async def _capture_api(chat_key: str, msg: str, ctx=None, **kw):
        SENT.append((chat_key, msg, kw.get("record", True)))

    message_api.send_text = _capture_api  # type: ignore[assignment]

    from nekro_agent.services.chat import universal_chat_service

    async def _blocked(*args, **kwargs):
        raise AssertionError("outbound send reached the adapter layer")

    universal_chat_service.universal_chat_service.send_agent_message = _blocked  # type: ignore[assignment]


def _load_plugin():
    spec = importlib.util.spec_from_file_location("nekro_auto_sleep", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["nekro_auto_sleep"] = module
    spec.loader.exec_module(module)
    return module


async def _ensure_channel():
    from nekro_agent.models.db_chat_channel import DBChatChannel

    existing = await DBChatChannel.filter(chat_key=TEST_CHAT_KEY).first()
    if existing:
        return existing
    return await DBChatChannel.create(
        adapter_key="onebot_v11",
        instance_key="",
        channel_id="900000001",
        channel_name="auto-sleep probe",
        channel_type="group",
        chat_key=TEST_CHAT_KEY,
        is_active=True,
        data="{}",
    )


async def _cleanup(module):
    from nekro_agent.models.db_chat_channel import DBChatChannel
    from nekro_agent.models.db_plugin_data import DBPluginData

    await DBPluginData.filter(target_chat_key=TEST_CHAT_KEY).delete()
    await DBChatChannel.filter(chat_key=TEST_CHAT_KEY).delete()


def _load_env(path: str = ".env.dev") -> None:
    """The app gets these from `bot --env=dev`; a standalone script does not."""
    import os

    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip(chr(34)).strip(chr(39)))


async def main() -> int:
    _load_env()
    from nekro_agent.core.database import init_db

    _install_send_guards()
    await init_db()
    print("database ready\n")

    m = _load_plugin()
    check("plugin imports against the real host", True, f"key={m.plugin.key}")

    # --- declared surface ---------------------------------------------------
    check(
        "sleep_brief declared alongside allow_sleep",
        bool(getattr(m.plugin, "sleep_brief", "")),
        repr(getattr(m.plugin, "sleep_brief", ""))[:80],
    )
    command_names = sorted(c.name for c in getattr(m.plugin, "_commands", []))
    check(
        "/sleep command group registered",
        len(command_names) >= 6,
        ", ".join(command_names),
    )
    check(
        "config schema builds",
        m.plugin.get_config(m.SleepConfig) is not None,
        f"{len(m.SleepConfig.model_fields)} fields",
    )

    await _ensure_channel()

    cfg = m.config
    cfg.ENABLED = True
    cfg.TIMEZONE = "Asia/Shanghai"
    cfg.WAKE_NOTICE_POLICY = "always"
    cfg.HISTORY_MODE = "preserve"
    cfg.URGENT_KEYWORDS = ""

    # --- discovery query against the real database --------------------------
    from nekro_auto_sleep.persistence import ScheduleOverrideStore, SleepStateStore

    m._store = SleepStateStore(m.plugin.store)
    m._overrides = ScheduleOverrideStore(m.plugin.store)
    try:
        discovered = await m._discover_chat_keys()
        check(
            "boot discovery query runs on the real schema",
            True,
            f"{len(discovered)} chat_key(s) visible, test channel included: "
            f"{TEST_CHAT_KEY in discovered}",
        )
    except Exception as exc:
        check("boot discovery query runs on the real schema", False, repr(exc))
        discovered = set()

    # --- v1 state on disk must still load ------------------------------------
    import json

    from nekro_agent.models.db_plugin_data import DBPluginData
    from nekro_auto_sleep.models import ChatSleepState

    rows = await DBPluginData.filter(
        plugin_key=m.plugin.key, data_key="state.v1"
    ).limit(500)
    legacy = [json.loads(r.data_value) for r in rows]
    legacy = [d for d in legacy if d.get("schema_version", 1) < 2]
    if legacy:
        accepted = 0
        first_error = ""
        for payload in legacy:
            try:
                ChatSleepState.model_validate(payload)
                accepted += 1
            except Exception as exc:
                first_error = first_error or f"{type(exc).__name__}: {exc}"[:200]
        check(
            "real v1 state migrates into the v2 model",
            accepted == len(legacy),
            f"{accepted}/{len(legacy)} rows accepted" + (f" | {first_error}" if first_error else ""),
        )
    else:
        print("[SKIP] no v1 state rows on this install")

    # From here on only the synthetic channel is touched.
    async def _only_test_channel():
        return {TEST_CHAT_KEY}

    m._discover_chat_keys = _only_test_channel

    # --- lifecycle ----------------------------------------------------------
    try:
        await m.init()
        check("mount_init_method runs (wraps + boot reconcile)", True)
    except Exception as exc:
        check("mount_init_method runs (wraps + boot reconcile)", False, traceback.format_exc()[-400:])

    from nekro_agent.services.message_service import message_service
    from nekro_auto_sleep.runtime import is_wrapped

    check(
        "schedule_agent_task really wrapped on the live singleton",
        is_wrapped(message_service, "schedule_agent_task"),
    )

    # --- persona name -------------------------------------------------------
    from nekro_agent.schemas.agent_ctx import AgentCtx

    ctx = await AgentCtx.create_by_chat_key(TEST_CHAT_KEY)
    persona = await m._get_persona_name(ctx)
    check(
        "persona name resolves through the real ctx",
        persona != "" and persona is not None,
        f"persona={persona!r} (fallback is {cfg.FALLBACK_PERSONA_NAME!r})",
    )

    # --- schedule layering --------------------------------------------------
    from nekro_auto_sleep.models import ScheduleOverride

    await m._get_overrides().set_channel(
        TEST_CHAT_KEY, ScheduleOverride(timezone="Asia/Tokyo", sleep_time="01:00")
    )
    m.invalidate_schedule_cache(TEST_CHAT_KEY)
    schedule = await m._resolve_schedule_for(TEST_CHAT_KEY)
    check(
        "channel override survives a real store round-trip",
        schedule.timezone == "Asia/Tokyo" and schedule.sleep_time == "01:00",
        f"{schedule.sleep_time} {schedule.timezone} sources={schedule.sources}",
    )
    await m._get_overrides().set_channel(TEST_CHAT_KEY, ScheduleOverride())
    m.invalidate_schedule_cache(TEST_CHAT_KEY)

    # --- commands -----------------------------------------------------------
    from nekro_agent.services.command.schemas import CommandExecutionContext

    cmd_ctx = CommandExecutionContext(
        user_id="1", chat_key=TEST_CHAT_KEY, username="probe", adapter_key="onebot_v11",
        is_super_user=True, is_advanced_user=True,
    )
    commands = {c.name: c.execute_func for c in m.plugin._commands}

    resp = await commands["sleep.now"](cmd_ctx)
    state = m._get_store().get_cached(TEST_CHAT_KEY)
    check(
        "/sleep now puts the channel to bed",
        state is not None and state.status.value == "ASLEEP",
        f"{resp.message} | sleep_at={state.cycle.sleep_at if state and state.cycle else None}",
    )

    # --- the wake protocol, end to end --------------------------------------
    from nekro_agent.schemas.chat_message import ChatMessage

    def _msg(text: str) -> ChatMessage:
        msg = ChatMessage.create_empty(TEST_CHAT_KEY)
        msg.content_text = text
        msg.sender_id = 1
        msg.sender_name = "probe"
        msg.platform_userid = "1"
        msg.is_tome = True
        return msg

    print(f"       (hook is plugin.on_user_message_method = "
          f"{getattr(m.plugin.on_user_message_method, '__name__', '?')})")

    SENT.clear()
    sig1 = await m.plugin.on_user_message_method(ctx, _msg("醒醒"))
    check(
        "first call answers without the LLM and keeps the message in history",
        sig1.name == "BLOCK_TRIGGER" and len(SENT) == 1 and SENT[0][2] is True,
        f"signal={sig1.name} sent={SENT}",
    )

    sig2 = await m.plugin.on_user_message_method(ctx, _msg("要"))
    state = m._get_store().get_cached(TEST_CHAT_KEY)
    check(
        "second message force-triggers the LLM round",
        sig2.name == "FORCE_TRIGGER" and state.wake_decision_pending,
        f"signal={sig2.name} pending={state.wake_decision_pending}",
    )

    inject = await m.inject_sleep_status(ctx)
    check(
        "decision round gets its instructions",
        "resume_sleep" in inject and "不要输出任何内容" in inject,
        inject.splitlines()[1] if len(inject.splitlines()) > 1 else inject,
    )

    methods = await m.collect_available_methods(ctx)
    check(
        "resume_sleep exposed only while it is usable",
        len(methods) == 1,
        f"{[getattr(f, '__name__', f) for f in methods]}",
    )

    SENT.clear()
    await m.resume_sleep_tool(ctx)
    state = m._get_store().get_cached(TEST_CHAT_KEY)
    check(
        "declining the wake is silent and demotes the call",
        SENT == []
        and state.status.value == "ASLEEP"
        and not any(a.is_confirmed for a in state.cycle.wake_attempts)
        and len(state.cycle.sleep_segments) == 1,
        f"sent={SENT} attempts={[a.is_confirmed for a in state.cycle.wake_attempts]} "
        f"segments={len(state.cycle.sleep_segments)}",
    )

    # --- settlement ---------------------------------------------------------
    SENT.clear()
    resp = await commands["sleep.wake"](cmd_ctx)
    state = m._get_store().get_cached(TEST_CHAT_KEY)
    notice = SENT[0][1] if SENT else ""
    breakdown = state.cycle.quality_breakdown if state and state.cycle else None
    check(
        "wake settles, reports, and stores the breakdown",
        state.status.value == "AWAKE" and "已起床" in notice and breakdown is not None,
        f"notice={notice}",
    )
    if breakdown:
        print("       breakdown=" + ", ".join(f"{k}={v}" for k, v in breakdown.items()))

    status = await commands["sleep.status"](cmd_ctx)
    check("/sleep status renders", status.status.value == "success")
    print("\n--- /sleep status ---\n" + status.message + "\n---------------------\n")

    # --- restart recovery ---------------------------------------------------
    await commands["sleep.now"](cmd_ctx)
    state = m._get_store().get_cached(TEST_CHAT_KEY)
    planned = state.cycle.planned_wake_at
    m._get_store().clear_all()
    SENT.clear()
    await m._boot_reconcile(m._get_store(), planned + timedelta(minutes=5))
    state = m._get_store().get_cached(TEST_CHAT_KEY)
    check(
        "restart after the wake-up point settles and reports",
        state is not None and state.status.value == "AWAKE" and any("已起床" in t for _, t, _ in SENT),
        f"status={state.status.value if state else None} sent={[t for _, t, _ in SENT]}",
    )

    await m.cleanup()
    await _cleanup(m)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 60)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name in failed:
        print(f"  FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
