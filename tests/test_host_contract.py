"""Guard the assumptions this plugin makes about the NekroAgent host.

``tests/hoststub.py`` is hand-written, so a passing suite proves nothing unless
something checks the stub against the real thing. These tests parse a real
NekroAgent checkout with ``ast`` — no import, no DB, no event loop — and fail
when an assumption this release depends on stops holding.

Point the suite at a checkout with ``NEKRO_AGENT_SRC=/path/to/nekro-agent``.
Without it the whole module skips: the checks are only meaningful against a
real tree, and guessing a sibling directory would make a green run mean nothing.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from hoststub import MsgSignal

_SRC = os.environ.get("NEKRO_AGENT_SRC", "").strip()
HOST = Path(_SRC) if _SRC and (Path(_SRC) / "nekro_agent").is_dir() else None

pytestmark = pytest.mark.skipif(
    HOST is None,
    reason="NEKRO_AGENT_SRC is not set to a NekroAgent checkout",
)


def _parse(relpath: str) -> ast.Module:
    return ast.parse((HOST / relpath).read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _func(scope: ast.AST, name: str):
    for node in ast.walk(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _decorator_names(node: ast.AST) -> set[str]:
    names = set()
    for dec in node.decorator_list:  # type: ignore[attr-defined]
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _positional(node: ast.AST) -> list[str]:
    args = node.args  # type: ignore[attr-defined]
    return [a.arg for a in list(args.posonlyargs) + list(args.args)]


def _class_fields(cls: ast.ClassDef) -> set[str]:
    """Names assigned in the class body, in either ``x = ...`` or ``x: T = ...``."""
    found: set[str] = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    found.add(target.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            found.add(stmt.target.id)
    return found


# ---------------------------------------------------------------------------
# Signals and message flow
# ---------------------------------------------------------------------------


def test_msg_signal_values_match_the_stub():
    """The plugin returns these by name; the host branches on them by identity."""
    cls = _class(_parse("nekro_agent/schemas/signal.py"), "MsgSignal")
    real = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
            real[stmt.targets[0].id] = ast.literal_eval(stmt.value)

    assert real == {member.name: member.value for member in MsgSignal}


def test_block_all_discards_the_message_before_it_is_recorded():
    """The reason ordinary night messages must return BLOCK_TRIGGER.

    ``push_human_message`` returns on BLOCK_ALL *before* the message is
    recorded, so blocking everything overnight would leave the bot with no
    history to wake up to.
    """
    tree = _parse("nekro_agent/services/message_service.py")
    node = _func(tree, "push_human_message")

    block_lines = [
        cmp_node.lineno
        for cmp_node in ast.walk(node)
        if isinstance(cmp_node, ast.Compare)
        and isinstance(cmp_node.comparators[0], ast.Attribute)
        and cmp_node.comparators[0].attr == "BLOCK_ALL"
    ]

    def _is_recording_call(call: ast.Call) -> bool:
        if not isinstance(call.func, ast.Attribute):
            return False
        if call.func.attr.startswith("_persist"):
            return True
        return (
            call.func.attr == "create"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "DBChatMessage"
        )

    record_lines = [
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and _is_recording_call(call)
    ]

    assert block_lines, "no BLOCK_ALL check in push_human_message"
    assert record_lines, "no message-recording call found in push_human_message"
    assert min(block_lines) < min(record_lines)


def test_schedule_agent_task_is_public_and_async():
    """The host callable this plugin wraps to gate agent rounds."""
    cls = _class(_parse("nekro_agent/services/message_service.py"), "MessageService")
    node = _func(cls, "schedule_agent_task")

    assert isinstance(node, ast.AsyncFunctionDef)
    assert not node.name.startswith("_")
    assert "chat_key" in _positional(node)


def test_the_host_tracks_the_running_round_per_chat():
    """``_send_after_current_round`` waits on this to order 【已睡下】.

    Losing it is not fatal — the plugin falls back to a short pause — but the
    ordering guarantee weakens, so the test says so out loud.
    """
    cls = _class(_parse("nekro_agent/services/message_service.py"), "MessageService")
    assigned = {
        target.attr
        for node in ast.walk(cls)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute)
    }
    subscripted = {
        node.value.attr
        for node in ast.walk(cls)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
    }
    assert "running_tasks" in (assigned | subscripted)


# ---------------------------------------------------------------------------
# AgentCtx surface
# ---------------------------------------------------------------------------


def test_db_chat_channel_is_a_plain_property():
    """If this ever becomes awaitable, ``_get_persona_name`` has to change back."""
    cls = _class(_parse("nekro_agent/schemas/agent_ctx.py"), "AgentCtx")
    node = _func(cls, "db_chat_channel")

    assert isinstance(node, ast.FunctionDef), "db_chat_channel became async"
    assert "property" in _decorator_names(node)


def test_send_text_takes_a_keyword_only_record_flag():
    """``record=False`` is how the sleep notice stays out of the history."""
    cls = _class(_parse("nekro_agent/schemas/agent_ctx.py"), "AgentCtx")
    node = _func(cls, "send_text")

    assert isinstance(node, ast.AsyncFunctionDef)
    assert "record" in {a.arg for a in node.args.kwonlyargs}


# ---------------------------------------------------------------------------
# Plugin mount points
# ---------------------------------------------------------------------------


def test_plugin_exposes_every_mount_point_we_use():
    cls = _class(_parse("nekro_agent/services/plugin/base.py"), "NekroPlugin")
    defined = {
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    required = {
        "mount_config",
        "mount_init_method",
        "mount_cleanup_method",
        "mount_on_user_message",
        "mount_on_system_message",
        "mount_prompt_inject_method",
        "mount_sandbox_method",
        "mount_collect_methods",
        "mount_command",
        "mount_command_group",
    }
    assert required <= defined, required - defined


def test_collect_available_methods_is_async_and_skipped_when_unmounted():
    """``_mount_collect_methods`` exists because older hosts lack this hook."""
    cls = _class(_parse("nekro_agent/services/plugin/base.py"), "NekroPlugin")
    node = _func(cls, "collect_available_methods")

    assert isinstance(node, ast.AsyncFunctionDef)
    # The renderer falls back to ``self.sandbox_methods`` when no collect
    # function is mounted; the plugin relies on that for its own fallback.
    render = _func(cls, "render_sandbox_methods_prompt")
    dumped = ast.dump(render)
    assert "_collect_methods_func" in dumped


def test_the_prompt_names_sandbox_methods_by_function_name():
    """Why the tool function is named ``resume_sleep``, not ``resume_sleep_tool``.

    The prompt writes ``method.func.__name__``, so the function's own name is
    what the model sees and calls — the display name passed to
    ``mount_sandbox_method`` never reaches it.
    """
    cls = _class(_parse("nekro_agent/services/plugin/base.py"), "NekroPlugin")
    node = _func(cls, "render_sandbox_methods_prompt")

    assert "__name__" in ast.dump(node)


def test_mount_sandbox_method_accepts_allow_sleep_style_kwargs():
    """``allow_sleep=False`` / ``sleep_brief`` are passed at construction."""
    cls = _class(_parse("nekro_agent/services/plugin/base.py"), "NekroPlugin")
    init = _func(cls, "__init__")
    args = set(_positional(init)) | {a.arg for a in init.args.kwonlyargs}

    assert {"allow_sleep", "sleep_brief"} <= args
    assigned = {
        target.attr
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
    }
    assert {"allow_sleep", "sleep_brief"} <= assigned


def test_a_dormant_plugin_is_rendered_without_its_methods():
    """Why this plugin sets ``allow_sleep=False``.

    A sleeping plugin renders only its brief, withholding every method until
    the model calls ``extend_plugin_activation``. That is fatal for
    ``resume_sleep``: the one moment it matters is when somebody tells the bot
    to go back to sleep.
    """
    tree = _parse("nekro_agent/services/plugin/prompt_activation.py")
    node = _func(tree, "build_prompt_disclosure_view")

    sleeping_units = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and getattr(call.func, "id", getattr(call.func, "attr", ""))
        == "PluginPromptRenderUnit"
        and any(
            kw.arg == "state"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value == "sleeping"
            for kw in call.keywords
        )
    ]
    assert sleeping_units, "no sleeping render unit found; dormancy may have changed"
    for call in sleeping_units:
        assert "plugin_method_prompt" not in {kw.arg for kw in call.keywords}, (
            "a dormant plugin now discloses its methods; allow_sleep=False may no "
            "longer be necessary"
        )


def test_allow_sleep_false_is_honoured_over_any_strategy():
    tree = _parse("nekro_agent/services/plugin/prompt_activation.py")
    node = _func(tree, "is_sleep_effective")

    assert "plugin_strategy_is_protected" in ast.dump(node)


# ---------------------------------------------------------------------------
# Per-instance config and per-channel identity
# ---------------------------------------------------------------------------


def test_resolve_scoped_config_takes_an_explicit_instance_key():
    """Item 2 resolves instance config without relying on a contextvar.

    The maintenance loop is a background task and carries no inbound context,
    so the resolution has to be explicit or two bots in one group share hours.
    """
    tree = _parse("nekro_agent/services/plugin/scope.py")
    node = _func(tree, "resolve_scoped_config")
    args = _positional(node)

    assert args[:4] == ["plugin_key", "config_cls", "global_config", "instance_key"]


def test_instance_overrides_can_be_listed_by_field():
    """``_instance_schedule_fields`` reports provenance from this mapping."""
    tree = _parse("nekro_agent/services/plugin/scope.py")
    node = _func(tree, "load_instance_config_overrides")
    args = _positional(node)

    assert args[:2] == ["plugin_key", "instance_key"]


def test_adapter_instance_exposes_the_platform_account_id():
    """``_own_account_ids`` reads this to recognise this deployment's own bots.

    Without it, one instance's wake prompt (which contains the other's persona
    name, a wake keyword) is a valid call to the other.
    """
    cls = _class(_parse("nekro_agent/models/db_adapter_instance.py"), "DBAdapterInstance")
    assert "provider_account_id" in _class_fields(cls)


def test_chat_channel_exposes_instance_key_and_preset_id():
    """Per-channel config needs the instance; per-preset overrides need the preset."""
    cls = _class(_parse("nekro_agent/models/db_chat_channel.py"), "DBChatChannel")
    fields = _class_fields(cls)

    assert "instance_key" in fields
    assert "preset_id" in fields


def test_chat_message_fields_and_system_sender_convention():
    """``_recent_bedtime_greetings`` keys the whole lookup on these facts.

    A bedtime prompt is a ``DBChatMessage`` row with ``sender_id="-1"`` and
    ``sender_name="SYSTEM"``; the bot's own reply is also ``sender_id="-1"`` but
    carries the persona name. If either half of the convention moves, the
    collected "recent goodnights" silently become empty instead of wrong, so
    this pins both.
    """
    cls = _class(_parse("nekro_agent/models/db_chat_message.py"), "DBChatMessage")
    fields = _class_fields(cls)
    assert {
        "chat_key",
        "sender_id",
        "sender_name",
        "content_text",
        "send_timestamp",
    } <= fields

    tree = _parse("nekro_agent/services/message_service.py")
    node = _func(tree, "push_system_message")
    creates = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "create"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "DBChatMessage"
    ]
    assert creates, "push_system_message no longer records a DBChatMessage"
    assert any(
        any(
            kw.arg == "sender_name"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value == "SYSTEM"
            for kw in call.keywords
        )
        for call in creates
    ), "no DBChatMessage.create(..., sender_name=\"SYSTEM\") in push_system_message"


# ---------------------------------------------------------------------------
# Plugin storage
# ---------------------------------------------------------------------------


def test_plugin_store_has_the_three_calls_the_override_store_makes():
    """``ScheduleOverrideStore`` keeps its rows beside the night state."""
    cls = _class(_parse("nekro_agent/services/plugin/base.py"), "PluginStore")
    methods = {
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"get", "set", "delete"} <= methods

    for name in ("get", "set", "delete"):
        node = _func(cls, name)
        assert isinstance(node, ast.AsyncFunctionDef), name
        assert {"chat_key", "store_key"} <= set(_positional(node)), name

    store_prop = _func(
        _class(_parse("nekro_agent/services/plugin/base.py"), "NekroPlugin"), "store"
    )
    assert "property" in _decorator_names(store_prop)


def test_db_plugin_data_has_the_columns_we_key_on():
    cls = _class(_parse("nekro_agent/models/db_plugin_data.py"), "DBPluginData")
    assert {"plugin_key", "data_key", "target_chat_key"} <= _class_fields(cls)


# ---------------------------------------------------------------------------
# Command system
# ---------------------------------------------------------------------------


def test_a_command_line_is_split_at_the_first_space():
    """Why ``/sleep status`` needs its own top-level command.

    ``detect_command`` takes everything before the first space as the command
    name, so a group registered as ``sleep.status`` is only reachable dotted —
    the spaced form resolves the bare name ``sleep``.
    """
    cls = _class(_parse("nekro_agent/adapters/interface/base.py"), "BaseAdapter")
    node = _func(cls, "detect_command")

    splits = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "split"
    ]
    assert splits, "detect_command no longer splits the command line"


def test_command_system_surface():
    """``/sleep ...`` is registered against these; missing them disables the group."""
    perms = _class(_parse("nekro_agent/services/command/base.py"), "CommandPermission")
    members = {
        stmt.targets[0].id
        for stmt in perms.body
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
    }
    assert {"PUBLIC", "ADVANCED", "SUPER_USER"} <= members

    schemas = _parse("nekro_agent/services/command/schemas.py")
    ctx = _class(schemas, "CommandExecutionContext")
    assert "chat_key" in _class_fields(ctx)

    arg_init = _func(_class(schemas, "Arg"), "__init__")
    arg_kwargs = set(_positional(arg_init)) | {a.arg for a in arg_init.args.kwonlyargs}
    assert {"description", "default", "positional", "choices"} <= arg_kwargs

    ctl = _class(_parse("nekro_agent/services/command/ctl.py"), "CmdCtl")
    ctl_methods = {
        node.name
        for node in ctl.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"success", "failed"} <= ctl_methods


def test_commands_are_executed_with_keyword_arguments():
    """The dispatcher forwards ``context`` positionally; the host uses keywords.

    ``CommandHandler.execute`` calls ``execute_func(context, **parsed_kwargs)``,
    so every handler must accept its arguments by name as well.
    """
    tree = _parse("nekro_agent/services/command/base.py")
    node = _func(tree, "execute")

    assert node.args.kwarg is not None, "handlers no longer receive **kwargs"


# ---------------------------------------------------------------------------
# Timer hooks we wrap
# ---------------------------------------------------------------------------


def test_timer_internals_we_wrap_still_exist():
    """The lease gate is installed on these two; a rename would silently drop it."""
    one = _func(
        _class(_parse("nekro_agent/services/timer/timer_service.py"), "TimerService"),
        "_execute_task",
    )
    assert isinstance(one, ast.AsyncFunctionDef)

    rec = _parse("nekro_agent/services/timer/recurring_timer_service.py")
    two = _func(rec, "_fire_job")
    assert isinstance(two, ast.AsyncFunctionDef)
