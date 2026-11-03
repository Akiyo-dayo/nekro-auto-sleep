"""Operator commands: /sleep status|now|wake|skip|set|unset.

The host splits a command line at the first space only
(``adapters/interface/base.py:447``), so ``/sleep status`` arrives as the
command ``sleep`` with ``status`` as its first argument. The group subcommands
(``/sleep.status``) and the top-level dispatcher must therefore both work.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from conftest import FakeStoreBackend
from hoststub import AgentCtx, CommandExecutionContext
from nekro_auto_sleep.engine import transition_to_sleep
from nekro_auto_sleep.models import (
    ChatSleepState,
    ResolvedSchedule,
    ScheduleOverride,
    SleepStatus,
)
from nekro_auto_sleep.persistence import ScheduleOverrideStore, SleepStateStore
from nekro_auto_sleep.schedule import resolve_schedule

UTC = ZoneInfo("UTC")
TZ = ZoneInfo("Asia/Shanghai")
CHAT_KEY = "onebot_v11-group_888"
NIGHT = datetime(2026, 8, 17, 15, 0, tzinfo=UTC)  # 23:00 Asia/Shanghai


@pytest.fixture
def wired(plugin_mod):
    store = SleepStateStore(FakeStoreBackend())
    plugin_mod._store = store
    plugin_mod._overrides = ScheduleOverrideStore(store._backend)
    plugin_mod._is_runtime_active = True
    plugin_mod.invalidate_settings_cache()
    yield plugin_mod, store
    plugin_mod._cancel_deferred_sends()
    plugin_mod._overrides = None


def _ctx(chat_key: str = CHAT_KEY, **kwargs) -> CommandExecutionContext:
    return CommandExecutionContext(chat_key=chat_key, **kwargs)


async def _prime(plugin_mod, chat_key: str = CHAT_KEY) -> None:
    async def _touch(state: ChatSleepState) -> ChatSleepState:
        return state

    await plugin_mod._store.with_state(chat_key, _touch)


class TestCommandRegistration:
    def test_all_subcommands_and_the_dispatcher_are_registered(self, wired):
        plugin_mod, _store = wired
        names = set(plugin_mod.plugin.commands)
        assert {
            "sleep",
            "sleep.status",
            "sleep.now",
            "sleep.wake",
            "sleep.skip",
            "sleep.set",
            "sleep.unset",
        } <= names
        assert "sleep" in plugin_mod.plugin.command_groups

    def test_every_command_requires_super_user(self, wired):
        """These commands move a bot's night; none may be open to ordinary users."""
        plugin_mod, _store = wired
        # The decorators are the only place permission is declared, so check the
        # source text: a stub cannot observe decorator arguments.
        import pathlib

        source = pathlib.Path(plugin_mod.__file__).read_text(encoding="utf-8")
        marker = "Operator commands"
        commands_source = source[source.index(marker) :]
        assert commands_source.count("permission=CommandPermission.SUPER_USER") >= 7


class TestResolveSchedule:
    def test_no_overrides_keeps_the_base_schedule(self):
        resolved = resolve_schedule(
            timezone="Asia/Shanghai",
            sleep_time="23:00",
            wake_time_start="06:45",
            wake_time_end="08:30",
        )
        assert resolved.sleep_time == "23:00"
        assert set(resolved.sources.values()) == {"global"}

    def test_channel_beats_preset_beats_base(self):
        resolved = resolve_schedule(
            timezone="Asia/Shanghai",
            sleep_time="23:00",
            wake_time_start="06:45",
            wake_time_end="08:30",
            preset_override=ScheduleOverride(sleep_time="01:00", wake_time_start="09:00"),
            channel_override=ScheduleOverride(sleep_time="02:00"),
        )
        assert resolved.sleep_time == "02:00"
        assert resolved.sources["sleep_time"] == "channel"
        assert resolved.wake_time_start == "09:00"
        assert resolved.sources["wake_time_start"] == "preset"
        assert resolved.wake_time_end == "08:30"
        assert resolved.sources["wake_time_end"] == "global"

    def test_an_override_only_changes_the_fields_it_sets(self):
        resolved = resolve_schedule(
            timezone="Asia/Shanghai",
            sleep_time="23:00",
            wake_time_start="06:45",
            wake_time_end="08:30",
            channel_override=ScheduleOverride(timezone="Asia/Tokyo"),
        )
        assert resolved.timezone == "Asia/Tokyo"
        assert resolved.sleep_time == "23:00"
        assert resolved.sources["timezone"] == "channel"


class TestOverrideStore:
    async def test_channel_override_round_trips(self, fake_backend):
        overrides = ScheduleOverrideStore(fake_backend)
        await overrides.set_channel(CHAT_KEY, ScheduleOverride(sleep_time="01:30"))
        loaded = await overrides.get_channel(CHAT_KEY)
        assert loaded is not None
        assert loaded.sleep_time == "01:30"

    async def test_preset_override_is_keyed_by_preset(self, fake_backend):
        overrides = ScheduleOverrideStore(fake_backend)
        await overrides.set_preset(7, ScheduleOverride(wake_time_end="10:00"))
        assert (await overrides.get_preset(7)).wake_time_end == "10:00"
        assert await overrides.get_preset(8) is None
        assert await overrides.get_preset(None) is None

    async def test_clearing_deletes_the_row(self, fake_backend):
        overrides = ScheduleOverrideStore(fake_backend)
        await overrides.set_channel(CHAT_KEY, ScheduleOverride(sleep_time="01:30"))
        await overrides.set_channel(CHAT_KEY, ScheduleOverride())
        assert await overrides.get_channel(CHAT_KEY) is None

    async def test_corrupt_override_is_ignored_not_fatal(self, fake_backend):
        overrides = ScheduleOverrideStore(fake_backend)
        await fake_backend.set(
            chat_key=CHAT_KEY, store_key="schedule_override.v1", value="{not json"
        )
        assert await overrides.get_channel(CHAT_KEY) is None

    async def test_store_keys_are_stable(self, fake_backend):
        """The keys are a storage contract: changing them orphans saved overrides."""
        overrides = ScheduleOverrideStore(fake_backend)
        await overrides.set_channel(CHAT_KEY, ScheduleOverride(sleep_time="01:30"))
        await overrides.set_preset(3, ScheduleOverride(sleep_time="02:30"))
        keys = set(fake_backend._data)
        assert f"{CHAT_KEY}||schedule_override.v1" in keys
        assert "||preset_schedule.v1.3" in keys


class TestSettingsIntegration:
    async def test_channel_override_reaches_the_resolved_schedule(self, wired):
        plugin_mod, _store = wired
        await plugin_mod._overrides.set_channel(
            CHAT_KEY, ScheduleOverride(sleep_time="01:00")
        )
        plugin_mod.invalidate_settings_cache()
        schedule = await plugin_mod._resolve_schedule_for(CHAT_KEY)
        assert schedule.sleep_time == "01:00"
        assert schedule.sources["sleep_time"] == "channel"

    async def test_snapshot_uses_the_overridden_schedule(self, wired):
        plugin_mod, store = wired
        await _prime(plugin_mod)
        await plugin_mod._overrides.set_channel(
            CHAT_KEY, ScheduleOverride(sleep_time="01:00", timezone="Asia/Tokyo")
        )
        plugin_mod.invalidate_settings_cache()
        settings = await plugin_mod._settings_for(CHAT_KEY)
        snap = plugin_mod._make_config_snapshot(settings.config, settings.schedule)
        assert snap.sleep_time == "01:00"
        assert snap.timezone == "Asia/Tokyo"

    async def test_make_config_snapshot_still_works_without_arguments(self, wired):
        """Several existing tests call it that way; it must stay valid."""
        plugin_mod, _store = wired
        snap = plugin_mod._make_config_snapshot()
        assert snap.timezone == plugin_mod.config.TIMEZONE

    async def test_override_moves_the_bedtime(self, wired):
        plugin_mod, store = wired
        await _prime(plugin_mod)
        await plugin_mod._overrides.set_channel(
            CHAT_KEY, ScheduleOverride(sleep_time="01:00")
        )
        plugin_mod.invalidate_settings_cache()

        # 23:30 local: past the configured 23:00, before the overridden 01:00.
        await plugin_mod._maintain_chat(store, CHAT_KEY, NIGHT + timedelta(minutes=30))
        assert store.get_cached(CHAT_KEY).status == SleepStatus.AWAKE

        await plugin_mod._maintain_chat(store, CHAT_KEY, NIGHT + timedelta(hours=2))
        assert store.get_cached(CHAT_KEY).status == SleepStatus.ASLEEP


class TestStatus:
    async def test_status_reports_the_schedule_and_provenance(self, wired):
        plugin_mod, _store = wired
        await _prime(plugin_mod)
        await plugin_mod._overrides.set_channel(
            CHAT_KEY, ScheduleOverride(sleep_time="01:00")
        )
        plugin_mod.invalidate_settings_cache()

        response = await plugin_mod.sleep_status_command(_ctx())
        assert response.status == "success"
        assert "01:00" in response.message
        assert "本频道" in response.message

    async def test_status_before_any_record(self, wired):
        plugin_mod, _store = wired
        response = await plugin_mod.sleep_status_command(_ctx())
        assert response.status == "success"
        assert "尚未装载" in response.message

    async def test_status_reports_the_current_night(self, wired):
        plugin_mod, store = wired
        await _prime(plugin_mod)
        snap = plugin_mod._make_config_snapshot()
        await store.save(transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), NIGHT, snap))

        response = await plugin_mod.sleep_status_command(_ctx())
        assert "睡着" in response.message
        assert "已睡" in response.message

    async def test_status_reports_skip_and_quality(self, wired):
        plugin_mod, store = wired
        await _prime(plugin_mod)
        state = ChatSleepState(
            chat_key=CHAT_KEY,
            skip_sleep_date="2026-08-17",
            quality_history={"2026-08-16": 92},
        )
        await store.save(state)

        response = await plugin_mod.sleep_status_command(_ctx())
        assert "不睡" in response.message
        assert "92%" in response.message


class TestNow:
    async def test_now_puts_an_awake_chat_to_sleep(self, wired):
        plugin_mod, store = wired
        await _prime(plugin_mod)
        response = await plugin_mod.sleep_now_command(_ctx())
        assert response.status == "success"
        assert store.get_cached(CHAT_KEY).status == SleepStatus.ASLEEP

    async def test_now_starts_the_cycle_at_the_current_instant(self, wired):
        plugin_mod, store = wired
        await _prime(plugin_mod)
        before = datetime.now(UTC)
        await plugin_mod.sleep_now_command(_ctx())
        cycle = store.get_cached(CHAT_KEY).cycle
        assert cycle is not None
        # Started now, not at the configured 23:00.
        assert cycle.sleep_at >= before - timedelta(seconds=5)
        assert cycle.sleep_at <= datetime.now(UTC) + timedelta(seconds=5)

    async def test_now_keeps_the_planned_wake_time(self, wired):
        plugin_mod, store = wired
        await _prime(plugin_mod)
        await plugin_mod.sleep_now_command(_ctx())
        cycle = store.get_cached(CHAT_KEY).cycle
        assert cycle.planned_wake_at > cycle.sleep_at

    async def test_now_refuses_when_already_asleep(self, wired):
        plugin_mod, store = wired
        await _prime(plugin_mod)
        snap = plugin_mod._make_config_snapshot()
        await store.save(transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), NIGHT, snap))

        response = await plugin_mod.sleep_now_command(_ctx())
        assert response.status == "error"
        assert "不能直接入睡" in response.message


class TestWake:
    async def test_wake_settles_the_night(self, wired):
        plugin_mod, store = wired
        await _prime(plugin_mod)
        snap = plugin_mod._make_config_snapshot()
        await store.save(transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), NIGHT, snap))

        response = await plugin_mod.sleep_wake_command(_ctx())
        assert response.status == "success"
        assert store.get_cached(CHAT_KEY).status == SleepStatus.AWAKE

    async def test_wake_refuses_when_awake(self, wired):
        plugin_mod, _store = wired
        await _prime(plugin_mod)
        response = await plugin_mod.sleep_wake_command(_ctx())
        assert response.status == "error"
        assert "本来就是醒着的" in response.message


class TestSkip:
    @pytest.fixture(autouse=True)
    def _frozen_now(self, wired, monkeypatch):
        """Pin "now" to the start of the night so the marker lands on 2026-08-17."""
        plugin_mod, _store = wired
        monkeypatch.setattr(plugin_mod, "_utcnow", lambda: NIGHT)

    async def test_skip_marks_tonight(self, wired):
        plugin_mod, store = wired
        await _prime(plugin_mod)
        response = await plugin_mod.sleep_skip_command(_ctx())
        assert response.status == "success"
        assert store.get_cached(CHAT_KEY).skip_sleep_date == "2026-08-17"

    async def test_skip_prevents_the_transition(self, wired):
        plugin_mod, store = wired
        await _prime(plugin_mod)
        await plugin_mod.sleep_skip_command(_ctx())

        await plugin_mod._maintain_chat(store, CHAT_KEY, NIGHT + timedelta(minutes=30))
        assert store.get_cached(CHAT_KEY).status == SleepStatus.AWAKE

    async def test_skip_does_not_block_a_message_path_transition(self, wired):
        """A message at 23:30 must not slip past the skip marker."""
        plugin_mod, store = wired
        await _prime(plugin_mod)
        await plugin_mod.sleep_skip_command(_ctx())

        snap = plugin_mod._make_config_snapshot()
        state = store.get_cached(CHAT_KEY)
        compensated = plugin_mod._compensate_sleep_if_due(
            state, NIGHT + timedelta(minutes=30), TZ, plugin_mod.config, None
        )
        assert compensated.status == SleepStatus.AWAKE
        assert snap.sleep_time

    async def test_skip_expires_with_the_next_night(self, wired):
        plugin_mod, store = wired
        await _prime(plugin_mod)
        state = store.get_cached(CHAT_KEY)
        await store.save(state.model_copy(update={"skip_sleep_date": "2026-08-16"}))

        await plugin_mod._maintain_chat(store, CHAT_KEY, NIGHT + timedelta(minutes=30))
        assert store.get_cached(CHAT_KEY).status == SleepStatus.ASLEEP


class TestSetUnset:
    async def test_set_writes_a_channel_override(self, wired):
        plugin_mod, _store = wired
        response = await plugin_mod.sleep_set_command(_ctx(), "bed", "01:30")
        assert response.status == "success"
        override = await plugin_mod._overrides.get_channel(CHAT_KEY)
        assert override.sleep_time == "01:30"

    async def test_set_rejects_a_bad_time(self, wired):
        plugin_mod, _store = wired
        response = await plugin_mod.sleep_set_command(_ctx(), "bed", "25:99")
        assert response.status == "error"
        assert "HH:MM" in response.message

    async def test_set_rejects_a_bad_timezone(self, wired):
        plugin_mod, _store = wired
        response = await plugin_mod.sleep_set_command(_ctx(), "tz", "Mars/Olympus")
        assert response.status == "error"
        assert "时区" in response.message

    async def test_set_accepts_a_valid_timezone(self, wired):
        plugin_mod, _store = wired
        response = await plugin_mod.sleep_set_command(_ctx(), "tz", "Asia/Tokyo")
        assert response.status == "success"
        assert (await plugin_mod._overrides.get_channel(CHAT_KEY)).timezone == "Asia/Tokyo"

    async def test_set_keeps_earlier_fields(self, wired):
        plugin_mod, _store = wired
        await plugin_mod.sleep_set_command(_ctx(), "bed", "01:30")
        await plugin_mod.sleep_set_command(_ctx(), "wake-end", "10:00")
        override = await plugin_mod._overrides.get_channel(CHAT_KEY)
        assert override.sleep_time == "01:30"
        assert override.wake_time_end == "10:00"

    async def test_set_invalidates_the_settings_cache(self, wired):
        plugin_mod, _store = wired
        await plugin_mod._resolve_schedule_for(CHAT_KEY)  # warm the cache
        await plugin_mod.sleep_set_command(_ctx(), "bed", "01:30")
        schedule = await plugin_mod._resolve_schedule_for(CHAT_KEY)
        assert schedule.sleep_time == "01:30"

    async def test_unset_clears_the_channel_override(self, wired):
        plugin_mod, _store = wired
        await plugin_mod.sleep_set_command(_ctx(), "bed", "01:30")
        response = await plugin_mod.sleep_unset_command(_ctx())
        assert response.status == "success"
        assert await plugin_mod._overrides.get_channel(CHAT_KEY) is None

    async def test_set_preset_scope_uses_the_channel_preset(self, wired):
        plugin_mod, _store = wired
        response = await plugin_mod.sleep_set_command(
            _ctx(), "bed", "02:00", "preset"
        )
        assert response.status == "success"
        assert (await plugin_mod._overrides.get_preset(1)).sleep_time == "02:00"

    async def test_unset_preset_scope(self, wired):
        plugin_mod, _store = wired
        await plugin_mod.sleep_set_command(_ctx(), "bed", "02:00", "preset")
        await plugin_mod.sleep_unset_command(_ctx(), "preset")
        assert await plugin_mod._overrides.get_preset(1) is None

    async def test_unknown_field_is_reported(self, wired):
        plugin_mod, _store = wired
        response = await plugin_mod.sleep_set_command(_ctx(), "nope", "x")
        assert response.status == "error"
        assert "未知字段" in response.message


class TestDispatcher:
    async def test_spaced_status_is_forwarded(self, wired):
        plugin_mod, _store = wired
        await _prime(plugin_mod)
        response = await plugin_mod.sleep_dispatch_command(_ctx(), "status")
        assert response.status == "success"
        assert "睡眠状态" in response.message

    async def test_bare_sleep_defaults_to_status(self, wired):
        plugin_mod, _store = wired
        await _prime(plugin_mod)
        response = await plugin_mod.sleep_dispatch_command(_ctx(), "")
        assert response.status == "success"

    async def test_spaced_set_passes_its_arguments(self, wired):
        plugin_mod, _store = wired
        response = await plugin_mod.sleep_dispatch_command(_ctx(), "set", "bed", "01:30")
        assert response.status == "success"
        assert (await plugin_mod._overrides.get_channel(CHAT_KEY)).sleep_time == "01:30"

    async def test_spaced_set_with_scope(self, wired):
        plugin_mod, _store = wired
        response = await plugin_mod.sleep_dispatch_command(
            _ctx(), "set", "bed", "01:30", "preset"
        )
        assert response.status == "success"
        assert (await plugin_mod._overrides.get_preset(1)).sleep_time == "01:30"

    async def test_unknown_subcommand_is_reported(self, wired):
        plugin_mod, _store = wired
        response = await plugin_mod.sleep_dispatch_command(_ctx(), "nope")
        assert response.status == "error"
        assert "未知子命令" in response.message

    async def test_spaced_now_and_skip_work(self, wired):
        plugin_mod, store = wired
        await _prime(plugin_mod)
        assert (await plugin_mod.sleep_dispatch_command(_ctx(), "now")).status == "success"
        assert store.get_cached(CHAT_KEY).status == SleepStatus.ASLEEP

    async def test_bad_arguments_are_reported_not_raised(self, wired):
        plugin_mod, _store = wired
        response = await plugin_mod.sleep_dispatch_command(_ctx(), "wake", "extra")
        assert response.status == "error"

    async def test_leading_dot_is_tolerated(self, wired):
        """/sleep.status parses as the name "sleep.status"; ".status" also lands here."""
        plugin_mod, _store = wired
        await _prime(plugin_mod)
        response = await plugin_mod.sleep_dispatch_command(_ctx(), ".status")
        assert response.status == "success"


class TestRenderStatus:
    def _schedule(self, sources: dict | None = None) -> ResolvedSchedule:
        return ResolvedSchedule(
            timezone="Asia/Shanghai",
            sleep_time="01:00",
            wake_time_start="09:00",
            wake_time_end="10:00",
            sources=sources if sources is not None else {"sleep_time": "channel"},
        )

    def test_marks_who_overrode_what(self, wired):
        plugin_mod, _store = wired
        text = plugin_mod._render_status(
            None, self._schedule({"sleep_time": "channel"}), "小助手", datetime.now(UTC)
        )
        assert "本频道" in text
        assert "01:00" in text

    def test_no_override_omits_the_origin_clause(self, wired):
        plugin_mod, _store = wired
        text = plugin_mod._render_status(
            None, self._schedule({}), "小助手", datetime.now(UTC)
        )
        assert "覆盖" not in text


class TestCommandGuards:
    async def test_status_survives_a_missing_store_entry(self, wired):
        plugin_mod, _store = wired
        response = await plugin_mod.sleep_status_command(_ctx("unknown-chat"))
        assert response.status == "success"

    @pytest.mark.parametrize(
        "command",
        ["sleep_status_command", "sleep_now_command", "sleep_wake_command", "sleep_skip_command"],
    )
    async def test_unreadable_row_is_answered_and_left_untouched(self, wired, command):
        plugin_mod, store = wired
        raw = "{not json"
        await store._backend.set(chat_key=CHAT_KEY, store_key="state.v1", value=raw)

        response = await getattr(plugin_mod, command)(_ctx())

        assert response.status != "success"
        assert "无法读取" in response.message
        assert await store._backend.get(chat_key=CHAT_KEY, store_key="state.v1") == raw
