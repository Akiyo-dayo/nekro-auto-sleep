"""A deliberately small stand-in for the NekroAgent host APIs under test.

Only the pieces the plugin actually touches are modelled: the plugin API
surface (config/i18n/plugin/schemas/signal), the ORM models it reads, the plugin
scope helpers, the command system and the message service. Nothing here is a
general-purpose host emulator — it exists so the wiring layer (hooks, maintenance
loop, settlement, commands) can be exercised without a running agent.

`:func:`install` registers the tree in ``sys.modules`` and returns a
``restore()`` callable; tests must always call it, otherwise the mock tree leaks
into the older suites that install their own minimal host.
"""

from __future__ import annotations

import enum
import sys
import types
from typing import Any, Callable, Optional

from pydantic import BaseModel


class MsgSignal(enum.Enum):
    """Mirror of ``nekro_agent.schemas.signal.MsgSignal`` (values matter)."""

    FORCE_TRIGGER = -1
    CONTINUE = 0
    BLOCK_TRIGGER = 1
    BLOCK_ALL = 2


class ConfigBase(BaseModel):
    pass


class ExtraField(BaseModel):
    model_config = {"extra": "allow"}


def i18n_text(*, zh_CN: str, en_US: str) -> dict[str, str]:
    return {"zh-CN": zh_CN, "en-US": en_US}


class SandboxMethodType(str, enum.Enum):
    TOOL = "tool"
    AGENT = "agent"
    BEHAVIOR = "behavior"
    MULTIMODAL_AGENT = "multimodal_agent"


# ---------------------------------------------------------------------------
# Messages and contexts
# ---------------------------------------------------------------------------


class ChatMessage:
    def __init__(
        self,
        content: str = "",
        chat_key: str = "",
        platform_userid: str = "u1",
        sender_id: int = 1,
        is_tome: bool = False,
        channel_type: str = "group",
        ext_data: Optional[dict] = None,
    ) -> None:
        self.content = content
        self.content_text = content
        self.chat_key = chat_key
        self.platform_userid = platform_userid
        self.sender_id = sender_id
        self.is_tome = is_tome
        self.channel_type = channel_type
        self.ext_data = ext_data or {}


class FakePreset:
    def __init__(self, name: str, preset_id: Optional[int] = None) -> None:
        self.name = name
        self.id = preset_id


def instance_key_of(chat_key: str) -> str:
    """``onebot_v11-bot2-group_1`` -> ``bot2``; the single-instance form -> ``""``."""
    parts = chat_key.split("-")
    return parts[1] if len(parts) >= 3 else ""


class FakeChatChannel:
    def __init__(
        self,
        chat_key: str,
        preset_name: str = "小助手",
        preset_id: Optional[int] = 1,
        instance_key: Optional[str] = None,
    ) -> None:
        self.chat_key = chat_key
        self.preset_id = preset_id
        self.instance_key = (
            instance_key if instance_key is not None else instance_key_of(chat_key)
        )
        self._preset = FakePreset(preset_name, preset_id)

    async def get_preset(self) -> FakePreset:
        return self._preset

    @classmethod
    async def get_channel(cls, chat_key: str = "") -> "FakeChatChannel":
        return _channel_factory(chat_key)


_channel_factory: Callable[[str], FakeChatChannel] = FakeChatChannel


def set_channel_factory(factory: Optional[Callable[[str], FakeChatChannel]]) -> None:
    global _channel_factory
    _channel_factory = factory or FakeChatChannel


class AgentCtx:
    """``db_chat_channel`` is a plain property here, exactly like the real host."""

    def __init__(self, chat_key: str) -> None:
        self.from_chat_key = chat_key
        self.sent: list[tuple[str, bool]] = []
        self.system_pushes: list[tuple[str, bool]] = []

    @property
    def chat_key(self) -> str:
        return self.from_chat_key

    @property
    def db_chat_channel(self) -> Optional[FakeChatChannel]:
        return None

    async def send_text(self, content: str, *, record: bool = True) -> None:
        self.sent.append((content, record))

    async def push_system(self, message: str, trigger_agent: bool = False) -> None:
        self.system_pushes.append((message, trigger_agent))

    @classmethod
    async def create_by_chat_key(cls, chat_key: str) -> "AgentCtx":
        return _ctx_factory(chat_key)


_ctx_factory: Callable[[str], AgentCtx] = AgentCtx


def set_ctx_factory(factory: Optional[Callable[[str], AgentCtx]]) -> None:
    global _ctx_factory
    _ctx_factory = factory or AgentCtx


# ---------------------------------------------------------------------------
# Plugin base class (captures every mount so tests can call the wiring)
# ---------------------------------------------------------------------------


class _StubLogger:
    """Records what the plugin hands to the host loguru logger."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def _make(self, level: str):
        def _log(message: Any = "", *args: Any, **kwargs: Any) -> None:
            self.lines.append((level, str(message)))

        return _log

    def __getattr__(self, name: str):
        return self._make(name)

    def opt(self, *args: Any, **kwargs: Any) -> "_StubLogger":
        return self


class NekroPlugin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.logger = _StubLogger()
        self.store: Any = None
        self._config_cls: Any = None
        self._config_instance: Any = None
        self.on_user_message: Any = None
        self.on_system_message: Any = None
        self.prompt_injects: dict[str, Callable] = {}
        self.sandbox_methods: dict[str, Callable] = {}
        self.collect_methods: Any = None
        self.commands: dict[str, Callable] = {}
        self.command_groups: dict[str, "CommandGroup"] = {}
        self.init_method: Any = None
        self.cleanup_method: Any = None

    @property
    def module_name(self) -> str:
        return self.init_kwargs.get("module_name", "nekro_auto_sleep")

    @property
    def key(self) -> str:
        return f"{self.init_kwargs.get('author', 'anon')}.{self.module_name}"

    def mount_config(self) -> Callable:
        def decorator(cls):
            self._config_cls = cls
            return cls

        return decorator

    def get_config(self, cls=None):
        if self._config_instance is None:
            self._config_instance = (cls or self._config_cls)()
        return self._config_instance

    def get_global_config(self, cls=None):
        return self.get_config(cls)

    def mount_on_user_message(self) -> Callable:
        def decorator(func):
            self.on_user_message = func
            return func

        return decorator

    def mount_on_system_message(self) -> Callable:
        def decorator(func):
            self.on_system_message = func
            return func

        return decorator

    def mount_prompt_inject_method(self, name: str, description: str = "") -> Callable:
        def decorator(func):
            self.prompt_injects[name] = func
            return func

        return decorator

    def mount_sandbox_method(
        self, method_type, name: str, description: str = "", **kwargs: Any
    ) -> Callable:
        def decorator(func):
            self.sandbox_methods[name] = func
            return func

        return decorator

    def mount_collect_methods(self) -> Callable:
        def decorator(func):
            self.collect_methods = func
            return func

        return decorator

    def mount_command_group(self, name: str, description: str = "", **kwargs: Any):
        group = CommandGroup(self, name)
        self.command_groups[name] = group
        return group

    def mount_command(self, name: str, description: str = "", **kwargs: Any) -> Callable:
        def decorator(func):
            self.commands[name] = func
            return func

        return decorator

    def mount_init_method(self) -> Callable:
        def decorator(func):
            self.init_method = func
            return func

        return decorator

    def mount_cleanup_method(self) -> Callable:
        def decorator(func):
            self.cleanup_method = func
            return func

        return decorator

    def on_enabled(self) -> Callable:
        def decorator(func):
            self.enabled_callback = func
            return func

        return decorator

    def on_disabled(self) -> Callable:
        def decorator(func):
            self.disabled_callback = func
            return func

        return decorator


# ---------------------------------------------------------------------------
# ORM models the plugin reads
# ---------------------------------------------------------------------------


class _QueryStub:
    """Enough of the peewee query builder for the reads under test."""

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def order_by(self, *args: Any) -> "_QueryStub":
        return self

    def values(self, *fields: str):
        async def _values() -> list[dict[str, Any]]:
            return [
                {field: getattr(row, field, None) for field in fields} for row in self._rows
            ]

        return _values()

    async def first(self):
        return self._rows[0] if self._rows else None


class _ModelStub:
    rows: list[Any] = []

    @classmethod
    def filter(cls, **kwargs: Any) -> _QueryStub:
        return _QueryStub(list(cls.rows))

    @classmethod
    def all(cls) -> _QueryStub:
        return _QueryStub(list(cls.rows))


class _AdapterInstanceRow:
    def __init__(self, provider_account_id: str) -> None:
        self.provider_account_id = provider_account_id


class DBAdapterInstance(_ModelStub):
    rows: list[_AdapterInstanceRow] = []


def set_own_bot_accounts(accounts: Any) -> None:
    DBAdapterInstance.rows = [_AdapterInstanceRow(str(a)) for a in accounts]


class _ChatMessageRow:
    def __init__(self, send_timestamp: int) -> None:
        self.send_timestamp = send_timestamp


class DBChatMessage(_ModelStub):
    rows: list[_ChatMessageRow] = []


class DBPluginData(_ModelStub):
    rows: list[Any] = []


# ---------------------------------------------------------------------------
# Plugin scope helpers (fork-only; the plugin probes for them)
# ---------------------------------------------------------------------------

_INSTANCE_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {}


def set_instance_config_override(plugin_key: str, instance_key: str, values: dict) -> None:
    _INSTANCE_OVERRIDES[(plugin_key, instance_key)] = dict(values)


def clear_instance_config_overrides() -> None:
    _INSTANCE_OVERRIDES.clear()


def load_instance_config_overrides(plugin_key: str, instance_key: str) -> dict:
    return dict(_INSTANCE_OVERRIDES.get((plugin_key, instance_key), {}))


def resolve_scoped_config(plugin_key: str, config_cls, global_config, instance_key: str):
    overrides = load_instance_config_overrides(plugin_key, instance_key)
    if not overrides:
        return global_config
    return config_cls.model_validate({**global_config.model_dump(), **overrides})


# ---------------------------------------------------------------------------
# Command system
# ---------------------------------------------------------------------------


class CommandPermission(str, enum.Enum):
    PUBLIC = "public"
    USER = "user"
    ADVANCED = "advanced"
    SUPER_USER = "super_user"


class CommandResponse(BaseModel):
    status: str
    message: str = ""


class CmdCtl:
    @staticmethod
    def success(content: str = "", **kwargs: Any) -> CommandResponse:
        return CommandResponse(status="success", message=content)

    @staticmethod
    def failed(content: str = "", **kwargs: Any) -> CommandResponse:
        return CommandResponse(status="error", message=content)


_UNSET = object()


class _Missing:
    """Stands in for an argument the real parser would have filled in."""


def Arg(description: str = "", *, default: Any = _UNSET, **kwargs: Any) -> Any:
    """Collapse to the declared default, like the real parser does before calling."""
    return _Missing() if default is _UNSET else default


class CommandExecutionContext(BaseModel):
    user_id: str = "u1"
    chat_key: str = ""
    username: str = "tester"
    adapter_key: str = "onebot_v11"
    is_super_user: bool = True
    is_advanced_user: bool = True


class CommandGroup:
    def __init__(self, plugin: NekroPlugin, name: str) -> None:
        self._plugin = plugin
        self._name = name

    def command(self, name: str, description: str = "", **kwargs: Any) -> Callable:
        def decorator(func):
            self._plugin.commands[f"{self._name}.{name}"] = func
            return func

        return decorator


# ---------------------------------------------------------------------------
# Message service
# ---------------------------------------------------------------------------


class FakeMessageService:
    """Enough for ``_install_wraps`` to find, wrap and unwrap the gate."""

    def __init__(self) -> None:
        self.scheduled: list[str] = []
        self.running_tasks: dict[str, Any] = {}

    async def schedule_agent_task(self, chat_key: str = "", **kwargs: Any) -> str:
        self.scheduled.append(chat_key)
        return "scheduled"

    async def _run_chat_agent_task(self, chat_key: str = "", **kwargs: Any) -> str:
        return "ran"


# ---------------------------------------------------------------------------
# Installation / restoration
# ---------------------------------------------------------------------------


def _module(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def install(*, with_commands: bool = True, with_scope: bool = True) -> Callable[[], None]:
    """Register the stub tree; returns a callable that undoes it."""
    saved = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == "nekro_agent" or name.startswith("nekro_agent.")
    }
    for name in saved:
        del sys.modules[name]

    pkg = _module("nekro_agent")
    pkg.__path__ = []  # type: ignore[attr-defined]
    api = _module("nekro_agent.api")
    api.__path__ = []  # type: ignore[attr-defined]
    _module("nekro_agent.api.i18n", i18n_text=i18n_text)
    plugin_mod = _module(
        "nekro_agent.api.plugin",
        ConfigBase=ConfigBase,
        ExtraField=ExtraField,
        NekroPlugin=NekroPlugin,
        SandboxMethodType=SandboxMethodType,
    )
    if with_commands:
        plugin_mod.Arg = Arg
        plugin_mod.CmdCtl = CmdCtl
        plugin_mod.CommandExecutionContext = CommandExecutionContext
        plugin_mod.CommandPermission = CommandPermission
        plugin_mod.CommandResponse = CommandResponse
    _module("nekro_agent.api.schemas", AgentCtx=AgentCtx)
    _module("nekro_agent.api.signal", MsgSignal=MsgSignal)

    schemas = _module("nekro_agent.schemas")
    schemas.__path__ = []  # type: ignore[attr-defined]
    _module("nekro_agent.schemas.chat_message", ChatMessage=ChatMessage)
    _module("nekro_agent.schemas.agent_ctx", AgentCtx=AgentCtx)

    models = _module("nekro_agent.models")
    models.__path__ = []  # type: ignore[attr-defined]
    _module("nekro_agent.models.db_chat_channel", DBChatChannel=FakeChatChannel)
    _module("nekro_agent.models.db_adapter_instance", DBAdapterInstance=DBAdapterInstance)
    _module("nekro_agent.models.db_chat_message", DBChatMessage=DBChatMessage)
    _module("nekro_agent.models.db_plugin_data", DBPluginData=DBPluginData)

    services = _module("nekro_agent.services")
    services.__path__ = []  # type: ignore[attr-defined]
    plugin_pkg = _module("nekro_agent.services.plugin")
    plugin_pkg.__path__ = []  # type: ignore[attr-defined]
    if with_scope:
        _module(
            "nekro_agent.services.plugin.scope",
            resolve_scoped_config=resolve_scoped_config,
            load_instance_config_overrides=load_instance_config_overrides,
        )

    if with_commands:
        command_pkg = _module("nekro_agent.services.command")
        command_pkg.__path__ = []  # type: ignore[attr-defined]
        _module(
            "nekro_agent.services.command.base",
            CommandPermission=CommandPermission,
            CommandResponse=CommandResponse,
        )
        _module("nekro_agent.services.command.ctl", CmdCtl=CmdCtl)
        _module(
            "nekro_agent.services.command.schemas",
            Arg=Arg,
            CommandExecutionContext=CommandExecutionContext,
            CommandResponse=CommandResponse,
        )

    message_service = FakeMessageService()
    _module(
        "nekro_agent.services.message_service",
        message_service=message_service,
        MessageService=FakeMessageService,
    )

    pkg.api = api  # type: ignore[attr-defined]
    pkg.schemas = schemas  # type: ignore[attr-defined]
    pkg.models = models  # type: ignore[attr-defined]
    pkg.services = services  # type: ignore[attr-defined]

    def restore() -> None:
        for name in [
            name
            for name in list(sys.modules)
            if name == "nekro_agent" or name.startswith("nekro_agent.")
        ]:
            del sys.modules[name]
        sys.modules.update(saved)

    return restore


def message_service() -> FakeMessageService:
    return sys.modules["nekro_agent.services.message_service"].message_service
