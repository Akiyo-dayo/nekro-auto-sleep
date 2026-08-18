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


class FakeChatChannel:
    def __init__(self, chat_key: str, preset_name: str = "小助手", is_active: bool = True) -> None:
        self.chat_key = chat_key
        self.is_active = is_active
        self._preset = FakePreset(preset_name)

    async def get_preset(self) -> FakePreset:
        return self._preset


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

    @property
    def chat_key(self) -> str:
        return self.from_chat_key

    @property
    def db_chat_channel(self) -> Optional[FakeChatChannel]:
        return self._db_chat_channel

    async def send_text(self, content: str, *, record: bool = True) -> None:
        self.sent.append((content, record))

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

    pkg.api = api  # type: ignore[attr-defined]
    pkg.schemas = schemas  # type: ignore[attr-defined]
