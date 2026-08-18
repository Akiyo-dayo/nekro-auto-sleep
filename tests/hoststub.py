"""Minimal but faithful stand-ins for the NekroAgent host APIs.

`nekro_auto_sleep/__init__.py` imports the host at module level, so the wiring
layer cannot be tested at all unless the host is importable. Installing a stub
into `sys.modules` before the first import makes the whole plugin — hooks,
maintenance loop, settlement — reachable from pytest.

The stub must stay honest about the parts under test. `tests/test_host_contract.py`
parses the real NekroAgent sources with `ast` and fails if this file drifts.
Most important invariant: `AgentCtx.db_chat_channel` is a plain property, **not**
awaitable — awaiting it is exactly the bug that pinned every persona name to the
configured fallback.
"""

from __future__ import annotations

import enum
import sys
import types
from typing import Any, Callable, Optional

from pydantic import BaseModel


# --- nekro_agent.schemas.signal ------------------------------------------------


class MsgSignal(enum.Enum):
    """Mirror of nekro_agent/schemas/signal.py (the values are load-bearing)."""

    FORCE_TRIGGER = -1
    CONTINUE = 0
    BLOCK_TRIGGER = 1
    BLOCK_ALL = 2


# --- nekro_agent.core.core_utils ----------------------------------------------


class ConfigBase(BaseModel):
    pass


class ExtraField(BaseModel):
    model_config = {"extra": "allow"}


def i18n_text(*, zh_CN: str, en_US: str) -> dict[str, str]:
    return {"zh-CN": zh_CN, "en-US": en_US}


# --- nekro_agent.services.plugin.schema ---------------------------------------


class SandboxMethodType(str, enum.Enum):
    TOOL = "tool"
    AGENT = "agent"
    BEHAVIOR = "behavior"
    MULTIMODAL_AGENT = "multimodal_agent"


# --- nekro_agent.schemas.chat_message -----------------------------------------


class ChatMessage:
    """Duck-typed stand-in; the plugin only ever probes attributes via hasattr."""

    def __init__(
        self,
        content: str = "",
        chat_key: str = "",
        platform_userid: str = "u1",
        sender_id: int = 1,
        is_tome: bool = False,
        channel_type: str = "group",
        extra_data: Optional[dict] = None,
    ) -> None:
        self.content = content
        self.content_text = content
        self.chat_key = chat_key
        self.platform_userid = platform_userid
        self.sender_id = sender_id
        self.is_tome = is_tome
        self.channel_type = channel_type
        self.extra_data = extra_data or {}


# --- nekro_agent.schemas.agent_ctx --------------------------------------------


class FakePreset:
    def __init__(self, name: str) -> None:
        self.name = name


def instance_key_of(chat_key: str) -> str:
    """Pull the instance segment out of a fork-style chat_key.

    `onebot_v11-inst1-group_123` -> "inst1"; the single-instance form
    `onebot_v11-group_123` -> "". Two accounts sitting in the same group differ
    only here, which is what keeps their nights apart.
    """
    parts = chat_key.split("-")
    return parts[1] if len(parts) >= 3 else ""


class FakeChatChannel:
    def __init__(
        self,
        chat_key: str,
        preset_name: str = "小助手",
        is_active: bool = True,
        preset_id: Optional[int] = 1,
        instance_key: Optional[str] = None,
    ) -> None:
        self.chat_key = chat_key
        self.is_active = is_active
        self.preset_id = preset_id
        self.instance_key = (
            instance_key if instance_key is not None else instance_key_of(chat_key)
        )
        self._preset = FakePreset(preset_name)

    async def get_preset(self) -> FakePreset:
        return self._preset

    @classmethod
    async def get_channel(cls, chat_key: str = "") -> "FakeChatChannel":
        return _channel_factory(chat_key)


def _default_channel_factory(chat_key: str) -> FakeChatChannel:
    return FakeChatChannel(chat_key)


_channel_factory: Callable[[str], FakeChatChannel] = _default_channel_factory


def set_channel_factory(factory: Callable[[str], FakeChatChannel]) -> None:
    """Control what `DBChatChannel.get_channel` hands back."""
    global _channel_factory
    _channel_factory = factory


def reset_channel_factory() -> None:
    global _channel_factory
    _channel_factory = _default_channel_factory


class AgentCtx:
    """Stand-in for nekro_agent.schemas.agent_ctx.AgentCtx.

    `db_chat_channel` is a plain property here because it is a plain property in
    the real host (both fork and upstream). Do not turn it into a coroutine to
    make a test pass — test_host_contract.py asserts the real one is sync.
    """

    def __init__(self, chat_key: str, db_chat_channel: Optional[FakeChatChannel] = None) -> None:
        self.from_chat_key = chat_key
        self._db_chat_channel = db_chat_channel
        self.sent: list[tuple[str, bool]] = []
        self.system_pushes: list[tuple[str, bool]] = []

    @property
    def chat_key(self) -> str:
        return self.from_chat_key

    @property
    def db_chat_channel(self) -> Optional[FakeChatChannel]:
        return self._db_chat_channel

    async def send_text(self, content: str, *, record: bool = True) -> None:
        self.sent.append((content, record))

    async def push_system(self, message: str, trigger_agent: bool = False) -> None:
        self.system_pushes.append((message, trigger_agent))

    @classmethod
    async def create_by_chat_key(cls, chat_key: str) -> "AgentCtx":
        return _ctx_factory(chat_key)


def _default_ctx_factory(chat_key: str) -> AgentCtx:
    return AgentCtx(chat_key, FakeChatChannel(chat_key))


_ctx_factory: Callable[[str], AgentCtx] = _default_ctx_factory


def set_ctx_factory(factory: Callable[[str], AgentCtx]) -> None:
    """Let a test control what `AgentCtx.create_by_chat_key` hands back."""
    global _ctx_factory
    _ctx_factory = factory


def reset_ctx_factory() -> None:
    global _ctx_factory
    _ctx_factory = _default_ctx_factory


# --- nekro_agent.api.plugin.NekroPlugin ---------------------------------------


class _StubLogger:
    def __getattr__(self, _name: str) -> Callable[..., None]:
        return lambda *a, **k: None


class NekroPlugin:
    """Captures every mount so tests can invoke the wiring directly."""

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
        self.init_method: Any = None
        self.cleanup_method: Any = None

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

    @property
    def key(self) -> str:
        return f"{self.init_kwargs.get('author', 'anon')}.{self.init_kwargs.get('module_name', 'plugin')}"

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

    def mount_sandbox_method(self, method_type, name: str, description: str = "", **kw) -> Callable:
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
        return CommandGroup(self, name)

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


# --- nekro_agent.models.db_chat_message -----------------------------------------

_LAST_MESSAGE_AT: dict[str, int] = {}


def set_last_message_at(chat_key: str, timestamp: Optional[int]) -> None:
    """Test helper: pretend this channel last spoke at `timestamp`."""
    if timestamp is None:
        _LAST_MESSAGE_AT.pop(chat_key, None)
    else:
        _LAST_MESSAGE_AT[chat_key] = int(timestamp)


def clear_last_message_times() -> None:
    _LAST_MESSAGE_AT.clear()


class _ChatMessageRow:
    def __init__(self, send_timestamp: int) -> None:
        self.send_timestamp = send_timestamp


class _ChatMessageQuery:
    def __init__(self, chat_key: str) -> None:
        self._chat_key = chat_key

    def order_by(self, *args: Any) -> "_ChatMessageQuery":
        return self

    async def first(self):
        stamp = _LAST_MESSAGE_AT.get(self._chat_key)
        return _ChatMessageRow(stamp) if stamp else None


class DBChatMessage:
    @classmethod
    def filter(cls, **kwargs: Any) -> "_ChatMessageQuery":
        return _ChatMessageQuery(kwargs.get("chat_key", ""))


# --- nekro_agent.models.db_adapter_instance -------------------------------------

_OWN_ACCOUNTS: list[str] = []


def set_own_bot_accounts(accounts) -> None:
    """Test helper: declare which platform accounts are our own bots."""
    _OWN_ACCOUNTS[:] = [str(a) for a in accounts]


class _AdapterInstanceQuery:
    async def values(self, *fields: str):
        return [{"provider_account_id": a} for a in _OWN_ACCOUNTS]


class DBAdapterInstance:
    @classmethod
    def all(cls) -> "_AdapterInstanceQuery":
        return _AdapterInstanceQuery()


# --- nekro_agent.services.plugin.scope (fork only) ------------------------------

_INSTANCE_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {}


def set_instance_config_override(plugin_key: str, instance_key: str, values: dict) -> None:
    """Test helper: pretend an operator saved per-instance plugin config."""
    _INSTANCE_OVERRIDES[(plugin_key, instance_key)] = dict(values)


def clear_instance_config_overrides() -> None:
    _INSTANCE_OVERRIDES.clear()


def load_instance_config_overrides(plugin_key: str, instance_key: str) -> dict:
    return dict(_INSTANCE_OVERRIDES.get((plugin_key, instance_key), {}))


def resolve_scoped_config(plugin_key: str, config_cls, global_config, instance_key: str):
    """Mirror of the fork: merge the instance override over the global config."""
    overrides = load_instance_config_overrides(plugin_key, instance_key)
    if not overrides:
        return global_config
    return config_cls.model_validate({**global_config.model_dump(), **overrides})


# --- nekro_agent.services.command.* --------------------------------------------


class CommandPermission(str, enum.Enum):
    PUBLIC = "public"
    USER = "user"
    ADVANCED = "advanced"
    SUPER_USER = "super_user"


class CommandExecutionContext(BaseModel):
    user_id: str = "u1"
    chat_key: str = ""
    username: str = "tester"
    adapter_key: str = "onebot_v11"
    is_super_user: bool = True
    is_advanced_user: bool = True


class CommandResponse(BaseModel):
    status: str
    message: str = ""


class CmdCtl:
    @staticmethod
    def success(content: str = "", **kw: Any) -> CommandResponse:
        return CommandResponse(status="success", message=content)

    @staticmethod
    def failed(content: str = "", **kw: Any) -> CommandResponse:
        return CommandResponse(status="error", message=content)


class _Missing:
    """Stands in for a positional argument the real parser would have filled."""


_UNSET = object()


def Arg(description: str = "", *, default: Any = _UNSET, **kwargs: Any) -> Any:
    """Collapse to the plain default, the way the real parser resolves it.

    The real `Arg` is a descriptor the command framework replaces before the
    handler runs; returning the default here lets tests call the handler as an
    ordinary function and still exercise the declared defaults.
    """
    return _Missing() if default is _UNSET else default


class CommandGroup:
    def __init__(self, plugin: "NekroPlugin", name: str) -> None:
        self._plugin = plugin
        self._name = name

    def command(self, name: str, description: str = "", **kwargs: Any) -> Callable:
        def decorator(func):
            self._plugin.commands[f"{self._name}.{name}"] = func
            return func

        return decorator


# --- nekro_agent.services.message_service --------------------------------------


class FakeMessageService:
    """Enough of the singleton for `_install_wraps` to find and wrap."""

    def __init__(self) -> None:
        self.scheduled: list[str] = []
        # The host keeps the in-flight agent round per chat here; the plugin
        # waits on it so 【已睡下】 lands after the model has spoken.
        self.running_tasks: dict[str, Any] = {}

    async def schedule_agent_task(self, chat_key: str = "", **kwargs: Any) -> str:
        self.scheduled.append(chat_key)
        return "scheduled"


# --- installation --------------------------------------------------------------


def _module(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def install_host_stub() -> None:
    """Register the fake `nekro_agent` package tree in sys.modules (idempotent)."""
    if "nekro_agent" in sys.modules and getattr(sys.modules["nekro_agent"], "_is_stub", False):
        return

    pkg = _module("nekro_agent")
    pkg._is_stub = True  # noqa: SLF001
    pkg.__path__ = []  # type: ignore[attr-defined]

    api = _module("nekro_agent.api")
    api.__path__ = []  # type: ignore[attr-defined]
    _module("nekro_agent.api.i18n", i18n_text=i18n_text)
    _module(
        "nekro_agent.api.plugin",
        ConfigBase=ConfigBase,
        ExtraField=ExtraField,
        NekroPlugin=NekroPlugin,
        SandboxMethodType=SandboxMethodType,
    )
    _module("nekro_agent.api.schemas", AgentCtx=AgentCtx)
    _module("nekro_agent.api.signal", MsgSignal=MsgSignal)

    schemas = _module("nekro_agent.schemas")
    schemas.__path__ = []  # type: ignore[attr-defined]
    _module("nekro_agent.schemas.chat_message", ChatMessage=ChatMessage)
    _module("nekro_agent.schemas.signal", MsgSignal=MsgSignal)
    _module("nekro_agent.schemas.agent_ctx", AgentCtx=AgentCtx)

    models = _module("nekro_agent.models")
    models.__path__ = []  # type: ignore[attr-defined]
    _module("nekro_agent.models.db_chat_channel", DBChatChannel=FakeChatChannel)
    _module("nekro_agent.models.db_adapter_instance", DBAdapterInstance=DBAdapterInstance)
    _module("nekro_agent.models.db_chat_message", DBChatMessage=DBChatMessage)

    services = _module("nekro_agent.services")
    services.__path__ = []  # type: ignore[attr-defined]
    plugin_pkg = _module("nekro_agent.services.plugin")
    plugin_pkg.__path__ = []  # type: ignore[attr-defined]
    _module(
        "nekro_agent.services.plugin.scope",
        resolve_scoped_config=resolve_scoped_config,
        load_instance_config_overrides=load_instance_config_overrides,
    )

    command_pkg = _module("nekro_agent.services.command")
    command_pkg.__path__ = []  # type: ignore[attr-defined]
    _module("nekro_agent.services.command.base", CommandPermission=CommandPermission)
    _module("nekro_agent.services.command.ctl", CmdCtl=CmdCtl)
    _module(
        "nekro_agent.services.command.schemas",
        Arg=Arg,
        CommandExecutionContext=CommandExecutionContext,
        CommandResponse=CommandResponse,
    )
    _module(
        "nekro_agent.services.message_service",
        message_service=FakeMessageService(),
        MessageService=FakeMessageService,
    )

    pkg.api = api  # type: ignore[attr-defined]
    pkg.schemas = schemas  # type: ignore[attr-defined]
    pkg.services = services  # type: ignore[attr-defined]
    pkg.models = models  # type: ignore[attr-defined]
