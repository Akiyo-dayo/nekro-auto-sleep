"""The plugin's stdlib logging must reach the host log, exactly once.

The domain layer logs through ``logging.getLogger("nekro_auto_sleep").*``; the
host logs through loguru. Without a bridge every INFO the plugin emitted at
runtime went nowhere, which is how "did boot reconciliation run?" became
unanswerable from a real install.
"""

from __future__ import annotations

import logging

import pytest


class _Recorder:
    """A loguru-shaped logger that records (level, message)."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def _make(self, level: str):
        def _log(message, *args, **kwargs):
            self.lines.append((level, str(message)))

        return _log

    def __getattr__(self, name):
        return self._make(name)

    def opt(self, *args, **kwargs):
        return self


class _Backend:
    async def get(self, chat_key="", user_key="", store_key=""):
        return None

    async def set(self, chat_key="", user_key="", store_key="", value=""):
        return 0

    async def delete(self, chat_key="", user_key="", store_key=""):
        return 0


@pytest.fixture
def wired(plugin_mod):
    plugin_mod._remove_log_bridge()
    recorder = _Recorder()
    plugin_mod.plugin.logger = recorder
    yield plugin_mod, recorder
    plugin_mod._remove_log_bridge()


class TestHostLogBridge:
    def test_installing_twice_forwards_once(self, wired):
        plugin_mod, recorder = wired

        plugin_mod._install_log_bridge()
        plugin_mod._install_log_bridge()

        logging.getLogger("nekro_auto_sleep").info("hello %s", "world")

        assert recorder.lines == [("info", "hello world")]

    def test_child_loggers_are_forwarded(self, wired):
        plugin_mod, recorder = wired
        plugin_mod._install_log_bridge()

        logging.getLogger("nekro_auto_sleep.persistence").warning("row %s bad", 7)

        assert recorder.lines == [("warning", "row 7 bad")]

    def test_removing_stops_forwarding(self, wired):
        plugin_mod, recorder = wired
        plugin_mod._install_log_bridge()
        logging.getLogger("nekro_auto_sleep").info("before")
        plugin_mod._remove_log_bridge()
        plugin_mod._remove_log_bridge()  # idempotent

        logging.getLogger("nekro_auto_sleep").info("after")

        assert recorder.lines == [("info", "before")]

    async def test_runtime_start_installs_and_stop_removes(self, wired, monkeypatch):
        plugin_mod, recorder = wired

        async def _noop(*args, **kwargs):
            return None

        monkeypatch.setattr(plugin_mod, "_install_wraps", lambda: True)
        monkeypatch.setattr(plugin_mod, "_maintenance_loop", _noop)
        monkeypatch.setattr(plugin_mod, "_check_install_dir_name", lambda: None)
        plugin_mod._store = plugin_mod.SleepStateStore(_Backend())
        monkeypatch.setattr(plugin_mod, "_discover_legacy_chat_keys", _noop)

        await plugin_mod._start_runtime()
        await plugin_mod._start_runtime()  # on_enabled may fire twice

        recorder.lines.clear()
        logging.getLogger("nekro_auto_sleep").info("tick")
        assert recorder.lines == [("info", "tick")]

        await plugin_mod._stop_runtime()
        recorder.lines.clear()
        logging.getLogger("nekro_auto_sleep").info("tick2")
        assert recorder.lines == []

    async def test_start_up_logs_reach_the_host(self, wired, monkeypatch):
        """The bridge is up before the store prewarms and the install dir is checked.

        1.3.0 installed it after ``initialize``: every "v2 row migrated" INFO
        from prewarming was emitted with no handler attached, so a real install
        could not tell whether the migration ran.
        """
        plugin_mod, recorder = wired

        async def _noop(*args, **kwargs):
            return None

        class _PrewarmingStore:
            def __init__(self, backend) -> None:
                pass

            async def initialize(self, discover_legacy_chat_keys=None) -> None:
                logging.getLogger("nekro_auto_sleep.persistence").info("prewarmed")

            def clear_all(self) -> None:
                pass

        def _check_dir() -> None:
            logging.getLogger("nekro_auto_sleep").error("install dir checked")

        monkeypatch.setattr(plugin_mod, "_install_wraps", lambda: True)
        monkeypatch.setattr(plugin_mod, "_maintenance_loop", _noop)
        monkeypatch.setattr(plugin_mod, "_check_install_dir_name", _check_dir)
        monkeypatch.setattr(plugin_mod, "SleepStateStore", _PrewarmingStore)
        monkeypatch.setattr(plugin_mod, "_discover_legacy_chat_keys", _noop)
        monkeypatch.setattr(plugin_mod, "_store", None)

        await plugin_mod._start_runtime()
        try:
            assert ("error", "install dir checked") in recorder.lines
            assert ("info", "prewarmed") in recorder.lines
        finally:
            await plugin_mod._stop_runtime()

    def test_existing_log_text_is_unchanged(self, wired):
        """The bridge forwards, it does not rewrite: the message text is verbatim."""
        plugin_mod, recorder = wired
        plugin_mod._install_log_bridge()

        logging.getLogger("nekro_auto_sleep").error(
            "自动睡眠插件安装目录名 `%s` 与 module_name `%s` 不一致！", "x", "y"
        )

        assert recorder.lines == [
            ("error", "自动睡眠插件安装目录名 `x` 与 module_name `y` 不一致！")
        ]
