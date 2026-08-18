"""Guard the assumptions this plugin makes about the NekroAgent host.

`tests/hoststub.py` is hand-written, so a passing suite proves nothing unless
something checks the stub against the real thing. These tests parse a real
NekroAgent checkout with `ast` (no import, no DB, no event loop) and fail when
an assumption the fixes depend on stops holding.

Point the suite at a checkout with `NEKRO_AGENT_SRC=/path/to/nekro-agent`;
it falls back to a sibling `NekroAgent_ByAkiyo` directory and skips when neither
is present. Run it against upstream too before publishing a release.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from tests.hoststub import MsgSignal


def _host_root() -> Path | None:
    env = os.environ.get("NEKRO_AGENT_SRC")
    candidates = [Path(env)] if env else []
    here = Path(__file__).resolve().parents[2]
    candidates += [here / "NekroAgent_ByAkiyo", here / "nekro-agent"]
    for path in candidates:
        if (path / "nekro_agent").is_dir():
            return path
    return None


HOST = _host_root()
pytestmark = pytest.mark.skipif(
    HOST is None,
    reason="no NekroAgent checkout found; set NEKRO_AGENT_SRC to enable",
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


def _decorator_names(node) -> set[str]:
    names = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def test_msg_signal_values_match_the_stub():
    """The plugin returns these by name; the host branches on them by identity."""
    cls = _class(_parse("nekro_agent/schemas/signal.py"), "MsgSignal")
    real = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
            real[stmt.targets[0].id] = ast.literal_eval(stmt.value)

    assert real == {member.name: member.value for member in MsgSignal}


def test_db_chat_channel_is_a_plain_property():
    """If this ever becomes awaitable, `_get_persona_name` has to change back.

    Awaiting the plain property is what silently pinned every persona name to
    the configured fallback, so the fix depends on it staying synchronous.
    """
    cls = _class(_parse("nekro_agent/schemas/agent_ctx.py"), "AgentCtx")
    node = _func(cls, "db_chat_channel")

    assert isinstance(node, ast.FunctionDef), "db_chat_channel became async"
    assert "property" in _decorator_names(node)


def test_send_text_takes_a_keyword_only_record_flag():
    cls = _class(_parse("nekro_agent/schemas/agent_ctx.py"), "AgentCtx")
    node = _func(cls, "send_text")

    assert isinstance(node, ast.AsyncFunctionDef)
    assert "record" in {a.arg for a in node.args.kwonlyargs}


def test_block_all_discards_the_message_before_it_is_recorded():
    """The reason ordinary night messages must return BLOCK_TRIGGER.

    `push_human_message` returns on BLOCK_ALL *before* `DBChatMessage.create`,
    so blocking everything overnight left the bot with no history to wake up to.
    If the host ever records first, BLOCK_ALL becomes safe again.
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
    # The fork writes DBChatMessage.create inline; upstream extracted the same
    # work into `_persist_human_message`. Accept either spelling.
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
    }
    assert required <= defined, required - defined


@pytest.mark.parametrize(
    ("relpath", "cls_name", "fields"),
    [
        (
            "nekro_agent/models/db_plugin_data.py",
            "DBPluginData",
            {"plugin_key", "data_key", "target_chat_key"},
        ),
        (
            "nekro_agent/models/db_chat_message.py",
            "DBChatMessage",
            {"chat_key", "send_timestamp"},
        ),
    ],
)
def test_boot_discovery_queries_have_the_columns_they_filter_on(relpath, cls_name, fields):
    """Boot reconciliation queries these models directly (PluginStore cannot list)."""
    cls = _class(_parse(relpath), cls_name)
    defined = {
        stmt.targets[0].id
        for stmt in cls.body
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
    }
    assert fields <= defined, fields - defined
