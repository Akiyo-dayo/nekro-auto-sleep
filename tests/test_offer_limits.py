"""A chat that keeps calling at 3am gets one reply, not one per message.

The limits live in the cycle's config snapshot so a mid-night config change
cannot retroactively allow offers already spent, and they reset with the cycle
rather than with the calendar day or the process lifetime.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from nekro_auto_sleep.engine import (
    ActionSendFixed,
    ActionStayAsleep,
    handle_message_while_asleep,
    offer_is_suppressed,
    transition_to_awake,
    transition_to_awake_early,
    transition_to_sleep,
)
from nekro_auto_sleep.models import ChatSleepState, SleepStatus

UTC = ZoneInfo("UTC")
TZ = ZoneInfo("Asia/Shanghai")
CHAT_KEY = "onebot_v11-group_123456789"
NIGHT = datetime(2026, 8, 17, 15, 0, tzinfo=UTC)  # 23:00 local


def _snapshot(plugin_mod, **overrides):
    cfg = plugin_mod.config.model_copy(update=overrides)
    return plugin_mod._make_config_snapshot(cfg)


def _sleeping(snapshot) -> ChatSleepState:
    return transition_to_sleep(ChatSleepState(chat_key=CHAT_KEY), NIGHT, snapshot)


def _call(state, when, user="u1"):
    return handle_message_while_asleep(
        state, when, user, "小明", valid_call=True, is_confirm=False
    )


class TestConfigDefaults:
    def test_defaults_match_the_spec(self, plugin_mod):
        assert plugin_mod.config.MAX_OFFERS_PER_NIGHT == 3
        assert plugin_mod.config.OFFER_COOLDOWN_MINUTES == 20

    def test_snapshot_carries_the_limits(self, plugin_mod):
        snapshot = _snapshot(plugin_mod, MAX_OFFERS_PER_NIGHT=5, OFFER_COOLDOWN_MINUTES=45)
        assert snapshot.max_offers_per_night == 5
        assert snapshot.offer_cooldown_minutes == 45

    def test_fields_have_bilingual_titles(self, plugin_mod):
        fields = plugin_mod.SleepConfig.model_fields
        for name in ("MAX_OFFERS_PER_NIGHT", "OFFER_COOLDOWN_MINUTES"):
            extra = fields[name].json_schema_extra or {}
            assert extra.get("i18n_title", {}).get("zh-CN")
            assert extra.get("i18n_title", {}).get("en-US")
            assert extra.get("i18n_description", {}).get("zh-CN")
            assert extra.get("i18n_description", {}).get("en-US")


class TestNightlyCap:
    def test_three_offers_then_silence(self, plugin_mod):
        snapshot = _snapshot(plugin_mod)
        state = _sleeping(snapshot)

        for i in range(3):
            state, action = _call(state, NIGHT + timedelta(minutes=i * 30))
            assert isinstance(action, ActionSendFixed)
            assert state.offers_sent_tonight == i + 1

        state, action = _call(state, NIGHT + timedelta(hours=3))
        assert isinstance(action, ActionStayAsleep)
        assert state.offers_sent_tonight == 3
        assert offer_is_suppressed(state, NIGHT + timedelta(hours=3), snapshot) == "nightly limit"

    def test_both_zero_restores_the_unlimited_1_2_5_behaviour(self, plugin_mod):
        snapshot = _snapshot(plugin_mod, MAX_OFFERS_PER_NIGHT=0, OFFER_COOLDOWN_MINUTES=0)
        state = _sleeping(snapshot)

        # Each call lands after the previous 180 s confirm window has expired.
        for i in range(6):
            state, action = _call(state, NIGHT + timedelta(minutes=1 + i * 4))
            assert isinstance(action, ActionSendFixed), f"call {i + 1} was not asked"

        assert state.offers_sent_tonight == 6
        assert offer_is_suppressed(state, NIGHT + timedelta(minutes=30), snapshot) is None

    def test_cap_is_per_cycle_not_per_day(self, plugin_mod):
        snapshot = _snapshot(plugin_mod, MAX_OFFERS_PER_NIGHT=1)
        state = _sleeping(snapshot)
        state, _ = _call(state, NIGHT + timedelta(minutes=1))
        assert state.offers_sent_tonight == 1

        # Next night: a new cycle, a fresh allowance.
        next_night = NIGHT + timedelta(days=1)
        state = transition_to_awake(state, next_night)
        state = transition_to_sleep(state, next_night, snapshot)

        assert state.offers_sent_tonight == 0
        assert state.last_offer_at is None
        state, action = _call(state, next_night + timedelta(minutes=1))
        assert isinstance(action, ActionSendFixed)


class TestCooldown:
    def test_second_call_inside_cooldown_stays_silent(self, plugin_mod):
        snapshot = _snapshot(plugin_mod)
        state = _sleeping(snapshot)

        state, _ = _call(state, NIGHT)
        assert state.offers_sent_tonight == 1

        # Offer expired (180 s window) but only 5 minutes passed.
        state, action = _call(state, NIGHT + timedelta(minutes=5))

        assert isinstance(action, ActionStayAsleep)
        assert state.offers_sent_tonight == 1
        assert offer_is_suppressed(state, NIGHT + timedelta(minutes=5), snapshot) == "cooldown"

    def test_asks_again_once_cooldown_elapsed(self, plugin_mod):
        snapshot = _snapshot(plugin_mod)
        state = _sleeping(snapshot)
        state, _ = _call(state, NIGHT)

        state, action = _call(state, NIGHT + timedelta(minutes=21))

        assert isinstance(action, ActionSendFixed)
        assert state.offers_sent_tonight == 2
        assert state.last_offer_at == NIGHT + timedelta(minutes=21)

    def test_cooldown_zero_allows_back_to_back(self, plugin_mod):
        snapshot = _snapshot(plugin_mod, OFFER_COOLDOWN_MINUTES=0, MAX_OFFERS_PER_NIGHT=10)
        state = _sleeping(snapshot)
        state, _ = _call(state, NIGHT)

        state, action = _call(state, NIGHT + timedelta(seconds=200))

        assert isinstance(action, ActionSendFixed)

    def test_confirmed_wake_clears_the_offer_but_keeps_the_counter(self, plugin_mod):
        snapshot = _snapshot(plugin_mod)
        state = _sleeping(snapshot)
        state, _ = _call(state, NIGHT)
        state, action = handle_message_while_asleep(
            state, NIGHT + timedelta(seconds=30), "u1", "小明",
            valid_call=True, is_confirm=True,
        )
        assert state.status == SleepStatus.AWAKE_EARLY
        assert state.pending_wake_offers == {}
        assert state.offers_sent_tonight == 1


class TestResetOnNewCycle:
    def test_transition_to_sleep_resets_counters(self, plugin_mod):
        snapshot = _snapshot(plugin_mod)
        state = ChatSleepState(
            chat_key=CHAT_KEY,
            status=SleepStatus.AWAKE,
            offers_sent_tonight=3,
            last_offer_at=NIGHT - timedelta(hours=5),
        )

        state = transition_to_sleep(state, NIGHT, snapshot)

        assert state.offers_sent_tonight == 0
        assert state.last_offer_at is None

    def test_early_wake_and_resume_do_not_reset(self, plugin_mod):
        """Going back to sleep inside one night must not refill the allowance."""
        snapshot = _snapshot(plugin_mod, MAX_OFFERS_PER_NIGHT=2)
        state = _sleeping(snapshot)
        state, _ = _call(state, NIGHT)
        state, _ = _call(state, NIGHT + timedelta(minutes=25))
        assert state.offers_sent_tonight == 2

        woken = transition_to_awake_early(state, NIGHT + timedelta(minutes=26), "u1", 10)
        assert woken.offers_sent_tonight == 2

    def test_suppression_reason_prefers_nightly_limit(self, plugin_mod):
        snapshot = _snapshot(plugin_mod, MAX_OFFERS_PER_NIGHT=1)
        state = ChatSleepState(
            chat_key=CHAT_KEY,
            offers_sent_tonight=1,
            last_offer_at=NIGHT,
        )
        assert offer_is_suppressed(state, NIGHT + timedelta(seconds=1), snapshot) == "nightly limit"
