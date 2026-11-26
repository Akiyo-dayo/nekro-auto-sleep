"""Bedtime greetings must stop sounding like one fixed template.

The model never saw its own previous goodnights, so a prompt that merely says
"do not reuse any fixed phrasing" could not stop the same four-beat skeleton
from reappearing in every group on every night. These tests cover the text
helpers, the history collection, and the prompt actually pushed at bedtime.
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest
from zoneinfo import ZoneInfo

import hoststub

UTC = ZoneInfo("UTC")
CHAT_A = "onebot_v11-group_100"
CHAT_B = "onebot_v11-group_200"
CHAT_C = "onebot_v11-group_300"
SLEEP_DATE = "2026-08-17"
ACTIVE_SEEN = datetime(2026, 8, 17, 7, 0, tzinfo=UTC)  # 15:00 +8, same local day

HEAD = "【系统提示：时间已到你的就寝时间，你准备入睡休息。】"


def _prompt_row(chat_key: str, timestamp: int) -> dict:
    return dict(
        chat_key=chat_key,
        sender_id="-1",
        sender_name="SYSTEM",
        content_text=HEAD,
        send_timestamp=timestamp,
    )


def _bot_row(chat_key: str, text: str, timestamp: int) -> dict:
    return dict(
        chat_key=chat_key,
        sender_id="-1",
        sender_name="小助手",
        content_text=text,
        send_timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


class TestCleanGreetingText:
    def test_strips_host_markers(self, plugin_mod):
        raw = "晚安 [@id:123@] 明天见 [Image: /x/y.png]"

        assert plugin_mod._clean_greeting_text(raw) == "晚安 明天见"

    def test_collapses_whitespace_including_newlines(self, plugin_mod):
        assert plugin_mod._clean_greeting_text("  睡\n\n 了\t啦  ") == "睡 了 啦"

    def test_truncates_with_ellipsis(self, plugin_mod):
        cleaned = plugin_mod._clean_greeting_text("一二三四五六", max_len=3)

        assert cleaned == "一二三…"


class TestSharedPhrases:
    def test_three_channel_example_keeps_only_the_longest_fragment(self, plugin_mod):
        entries = [
            ("c1", "不知不觉已经这个时间了 我先去睡啦"),
            ("c2", "不知不觉已经这个时间了 前辈晚安"),
            ("c3", "今天就聊到这 不知不觉已经这个时间了"),
        ]

        phrases = plugin_mod._shared_phrases(entries)

        assert phrases[0] == "不知不觉已经这个时间了"
        # No short fragment of the winner survives alongside it.
        for phrase in phrases[1:]:
            assert phrase not in phrases[0]
        assert "我先去睡啦" not in phrases  # only ever seen in one channel

    def test_phrase_repeated_in_one_channel_only_does_not_qualify(self, plugin_mod):
        entries = [
            ("c1", "早点休息别熬夜"),
            ("c1", "早点休息别熬夜"),
            ("c1", "早点休息别熬夜"),
        ]

        assert plugin_mod._shared_phrases(entries) == []

    def test_no_result_is_a_substring_of_an_equally_shared_longer_one(self, plugin_mod):
        entries = [
            ("c1", "快点睡觉吧明天见"),
            ("c2", "快点睡觉吧明天见"),
        ]

        phrases = plugin_mod._shared_phrases(entries)

        assert phrases == ["快点睡觉吧明天见"]

    def test_one_member_per_phrase_family(self, plugin_mod):
        # "不知不觉" is shared by four channels, its long form by only two:
        # both survive the containment pass, but only one may be reported.
        entries = [
            ("c1", "不知不觉已经这个时间了 先撤了"),
            ("c2", "不知不觉已经这个时间了 明天聊"),
            ("c3", "不知不觉就这么晚了"),
            ("c4", "不知不觉天都黑透了"),
            ("c1", "早点休息"),
            ("c3", "早点休息"),
        ]

        phrases = plugin_mod._shared_phrases(entries)

        assert phrases[0] == "不知不觉"
        assert "早点休息" in phrases
        for i, a in enumerate(phrases):
            for b in phrases[i + 1 :]:
                assert a not in b and b not in a

    def test_limit_is_honoured(self, plugin_mod):
        entries = [
            ("c1", "今天真是辛苦大家了 早点休息别熬夜 明天还要上班"),
            ("c2", "今天真是辛苦大家了 早点休息别熬夜 明天还要上班"),
        ]

        assert len(plugin_mod._shared_phrases(entries, limit=1)) == 1

    def test_empty_input(self, plugin_mod):
        assert plugin_mod._shared_phrases([]) == []


class TestBuildBedtimePrompt:
    def test_no_reference_section_when_everything_is_empty(self, plugin_mod):
        prompt = plugin_mod._build_bedtime_prompt([], [], [])

        assert prompt.startswith(plugin_mod.BEDTIME_PROMPT_HEAD)
        assert "【参考" not in prompt

    def test_own_examples_only_add_the_own_line(self, plugin_mod):
        prompt = plugin_mod._build_bedtime_prompt(["昨晚的说法"], [], [])

        assert "你在这个频道最近几晚说过的晚安：「昨晚的说法」" in prompt
        assert "多个频道反复用过的说法" not in prompt
        assert "晚安负面约束" not in prompt

    def test_shared_only_adds_the_shared_line(self, plugin_mod):
        prompt = plugin_mod._build_bedtime_prompt([], ["不知不觉已经这个时间了"], [])

        assert "你最近在多个频道反复用过的说法：「不知不觉已经这个时间了」" in prompt
        assert "你在这个频道最近几晚" not in prompt
        assert "晚安负面约束" not in prompt

    def test_constraints_only_adds_the_constraint_line(self, plugin_mod):
        prompt = plugin_mod._build_bedtime_prompt([], [], ["不要感叹时间", "不要劝早睡"])

        assert "晚安负面约束（必须遵守）：不要感叹时间；不要劝早睡" in prompt
        assert "你在这个频道最近几晚" not in prompt
        assert "多个频道反复用过的说法" not in prompt

    def test_old_skeleton_instruction_is_gone(self, plugin_mod):
        prompt = plugin_mod._build_bedtime_prompt(["昨晚"], ["套话"], ["约束"])

        assert "道一声晚安并告知你准备睡了" not in prompt
        assert prompt.startswith(plugin_mod.BEDTIME_PROMPT_HEAD)


# ---------------------------------------------------------------------------
# History collection
# ---------------------------------------------------------------------------


class TestRecentBedtimeGreetings:
    async def test_collects_only_actual_replies_newest_first(self, plugin_mod):
        now = 1_800_000_000
        hoststub.set_chat_messages(
            [
                _prompt_row(CHAT_A, now - 3600),
                _bot_row(CHAT_A, "晚安，我先睡了 [@id:123@]", now - 3590),
                # Outside the reply window: not this prompt's goodnight.
                _bot_row(CHAT_A, "窗口外的晚安", now - 3600 + 700),
                _prompt_row(CHAT_B, now - 7200),
                _bot_row(CHAT_B, "  [Image: /x.png] 睡啦  ", now - 7195),
                # Prompt whose reply never came.
                _prompt_row(CHAT_C, now - 5000),
                # A different system message must never be treated as a prompt.
                dict(
                    chat_key=CHAT_A,
                    sender_id="-1",
                    sender_name="SYSTEM",
                    content_text="【系统提示：随便别的什么】",
                    send_timestamp=now - 2000,
                ),
                # Older than the seven-day lookback.
                _prompt_row(CHAT_B, now - 8 * 86400),
                _bot_row(CHAT_B, "太旧了", now - 8 * 86400 + 5),
            ]
        )

        entries = await plugin_mod._recent_bedtime_greetings(now)

        assert entries == [
            (CHAT_A, "晚安，我先睡了"),
            (CHAT_B, "睡啦"),
        ]

    async def test_cache_hit_skips_the_database(self, plugin_mod):
        now = 1_800_000_000
        hoststub.set_chat_messages(
            [
                _prompt_row(CHAT_A, now - 100),
                _bot_row(CHAT_A, "晚安", now - 90),
            ]
        )

        await plugin_mod._recent_bedtime_greetings(now)
        first = hoststub.DBChatMessage.filter_calls
        assert first > 0

        await plugin_mod._recent_bedtime_greetings(now)
        assert hoststub.DBChatMessage.filter_calls == first

        plugin_mod.invalidate_bedtime_ref_cache()
        await plugin_mod._recent_bedtime_greetings(now)
        assert hoststub.DBChatMessage.filter_calls > first

    async def test_query_failure_returns_empty_and_is_not_cached(self, plugin_mod):
        hoststub.set_chat_messages([])
        hoststub.fail_chat_message_queries(True)

        assert await plugin_mod._recent_bedtime_greetings(1_800_000_000) == []

        hoststub.fail_chat_message_queries(False)
        hoststub.set_chat_messages(
            [_prompt_row(CHAT_A, 1_799_999_000), _bot_row(CHAT_A, "晚安", 1_799_999_100)]
        )
        assert await plugin_mod._recent_bedtime_greetings(1_800_000_000) == [
            (CHAT_A, "晚安")
        ]


# ---------------------------------------------------------------------------
# End to end through ``_maybe_send_bedtime``
# ---------------------------------------------------------------------------


@pytest.fixture
def pushed(plugin_mod, monkeypatch):
    """Record every prompt pushed through the stub ``AgentCtx``."""
    calls: list[tuple[str, str, bool]] = []

    def factory(chat_key: str) -> hoststub.AgentCtx:
        ctx = hoststub.AgentCtx(chat_key)

        async def push_system(prompt: str, trigger_agent: bool = False) -> None:
            calls.append((chat_key, prompt, trigger_agent))
            await ctx.__class__.push_system(ctx, prompt, trigger_agent)

        ctx.push_system = push_system  # type: ignore[assignment]
        return ctx

    hoststub.set_ctx_factory(factory)
    monkeypatch.setattr(plugin_mod, "_deterministic_hit", lambda *_: True)
    monkeypatch.setattr(plugin_mod.config, "LLM_GREETINGS_ENABLED", True)
    plugin_mod.invalidate_bedtime_ref_cache()
    return calls


def _seed_three_channels() -> None:
    now = int(time.time())
    hoststub.set_chat_messages(
        [
            _prompt_row(CHAT_A, now - 3000),
            _bot_row(CHAT_A, "不知不觉已经这个时间了 我先去睡啦", now - 2990),
            _prompt_row(CHAT_B, now - 2900),
            _bot_row(CHAT_B, "不知不觉已经这个时间了 前辈晚安", now - 2890),
            _prompt_row(CHAT_C, now - 2800),
            _bot_row(CHAT_C, "今天就聊到这 不知不觉已经这个时间了", now - 2790),
        ]
    )


class TestMaybeSendBedtime:
    async def test_prompt_contains_own_history_and_cross_channel_phrases(
        self, plugin_mod, pushed
    ):
        _seed_three_channels()

        await plugin_mod._maybe_send_bedtime(CHAT_A, SLEEP_DATE, ACTIVE_SEEN)

        assert len(pushed) == 1
        chat_key, prompt, trigger = pushed[0]
        assert chat_key == CHAT_A
        assert trigger is True
        assert "【参考" in prompt
        assert "「不知不觉已经这个时间了 我先去睡啦」" in prompt
        assert "「不知不觉已经这个时间了」" in prompt
        assert plugin_mod.config.BEDTIME_CONSTRAINTS in prompt

    async def test_own_history_never_mixes_in_another_channel_sentence(
        self, plugin_mod, pushed
    ):
        _seed_three_channels()

        await plugin_mod._maybe_send_bedtime(CHAT_A, SLEEP_DATE, ACTIVE_SEEN)

        prompt = pushed[0][1]
        own_line = next(
            line for line in prompt.splitlines() if line.startswith("- 你在这个频道")
        )
        assert "我先去睡啦" in own_line
        assert "前辈晚安" not in own_line
        # The shared fragment is reported, but never as another channel's sentence.
        assert "前辈晚安" not in prompt
        assert "今天就聊到这" not in prompt

    async def test_empty_constraints_skip_the_database_entirely(
        self, plugin_mod, pushed, monkeypatch
    ):
        _seed_three_channels()
        monkeypatch.setattr(plugin_mod.config, "BEDTIME_CONSTRAINTS", "")
        plugin_mod.invalidate_bedtime_ref_cache()

        await plugin_mod._maybe_send_bedtime(CHAT_A, SLEEP_DATE, ACTIVE_SEEN)

        assert hoststub.DBChatMessage.filter_calls == 0
        prompt = pushed[0][1]
        assert "【参考" not in prompt

    async def test_database_failure_still_sends_a_greeting(
        self, plugin_mod, pushed
    ):
        _seed_three_channels()
        hoststub.fail_chat_message_queries(True)

        await plugin_mod._maybe_send_bedtime(CHAT_A, SLEEP_DATE, ACTIVE_SEEN)

        assert len(pushed) == 1
        prompt = pushed[0][1]
        assert prompt.startswith(plugin_mod.BEDTIME_PROMPT_HEAD)
        # The constraint still applies; only the history references are missing.
        assert "你在这个频道最近几晚" not in prompt
        assert "多个频道反复用过的说法" not in prompt
        assert "晚安负面约束（必须遵守）" in prompt
