"""Confirmation keywords are judged on the whole answer, not by substring.

1.2.5 skipped single-character keywords in "long" messages, so 「要」 as the entire
answer was fine but 「要啊」 or 「好的」 were not, while 「我要出门了待会儿再回来」
was correctly ignored. The rule was a length heuristic standing in for a
linguistic one. Here a single character counts when it *is* the answer (or a
repetition of it), and longer keywords still match inside a sentence.
"""

from __future__ import annotations

import pytest

from hoststub import ChatMessage


@pytest.fixture
def mod(plugin_mod):
    return plugin_mod


def _intent(mod, text: str, *, persona: str = "小助手", **kwargs):
    return mod._check_wake_intent(
        ChatMessage(content=text, chat_key="onebot_v11-group_1"), persona, **kwargs
    )


class TestWholeAnswer:
    @pytest.mark.parametrize(
        "text",
        ["要", "要啊", "是", "是的", "叫醒", "叫醒他吧", "起床", "确认"],
    )
    def test_bare_answers_confirm(self, mod, text):
        is_confirm, is_cancel = _intent(mod, text)
        assert is_confirm is True
        assert is_cancel is False

    @pytest.mark.parametrize("text", ["好的", "好", "嗯嗯", "嗯"])
    def test_trailing_particles_are_trimmed(self, mod, text):
        """「好的」 is the answer 「好」 with a particle, not a different answer."""
        is_confirm, _ = _intent(mod, text, confirm_keywords=["好", "嗯"])
        assert is_confirm is True

    @pytest.mark.parametrize("text", ["要！", "要。", " 要 ", "要~", "ＹＥＳ"])
    def test_punctuation_and_width_do_not_matter(self, mod, text):
        is_confirm, _ = _intent(mod, text, confirm_keywords=["要", "yes"])
        assert is_confirm is True

    @pytest.mark.parametrize("text", ["要要要", "好好好"])
    def test_repeated_single_character_still_confirms(self, mod, text):
        is_confirm, _ = _intent(mod, text, confirm_keywords=["要", "好"])
        assert is_confirm is True


class TestMidSentence:
    @pytest.mark.parametrize(
        "text",
        [
            "我要出门了待会儿再回来",
            "我要睡了",
            "这个要点一下",
            "对不起吵到你了",
            "好困啊不想动",
            "是昨天那个事",
        ],
    )
    def test_single_character_inside_a_sentence_is_not_consent(self, mod, text):
        is_confirm, is_cancel = _intent(mod, text)
        assert is_confirm is False
        assert is_cancel is False

    @pytest.mark.parametrize("text", ["那就叫醒吧", "帮我叫醒一下", "你可以起床了"])
    def test_longer_keyword_inside_a_sentence_confirms(self, mod, text):
        is_confirm, _ = _intent(mod, text)
        assert is_confirm is True

    def test_ascii_keyword_needs_word_boundaries(self, mod):
        assert _intent(mod, "please wake up now", confirm_keywords=["wake up"])[0] is True
        # "wakeup" contains "wake" but is not the word.
        assert _intent(mod, "wakeupcall", confirm_keywords=["wake up"])[0] is False


class TestNegatives:
    @pytest.mark.parametrize("text", ["不要", "算了吧，你继续睡", "不用了", "别叫了"])
    def test_negative_wins(self, mod, text):
        is_confirm, is_cancel = _intent(mod, text)
        assert is_confirm is False
        assert is_cancel is True

    def test_negative_beats_a_confirm_keyword(self, mod):
        is_confirm, is_cancel = _intent(mod, "不要叫醒")
        assert is_confirm is False
        assert is_cancel is True


class TestMention:
    def test_mention_still_confirms_a_long_message(self, mod):
        is_confirm, _ = _intent(mod, "小助手 帮我查个东西")
        assert is_confirm is True

    def test_private_chat_confirms(self, mod):
        msg = ChatMessage(content="随便说点什么", chat_key="onebot_v11-private_1")
        msg.channel_type = "private"
        assert mod._check_wake_intent(msg, "小助手")[0] is True


class TestConfiguredKeywords:
    def test_custom_single_character_keyword(self, mod):
        assert _intent(mod, "行", confirm_keywords=["行"])[0] is True
        assert _intent(mod, "行吧我看看", confirm_keywords=["行"])[0] is False

    def test_empty_keyword_list_never_confirms(self, mod):
        assert _intent(mod, "要", confirm_keywords=[])[0] is False

    def test_call_keywords_are_unchanged(self, mod):
        """Item 6 touches only the answer path; calls stay substring matches."""
        msg = ChatMessage(content="喂 醒醒", chat_key="onebot_v11-group_1")
        assert mod._is_valid_call(msg, "小助手") is True
        assert mod._is_valid_call(ChatMessage(content="今天天气不错", chat_key="k"), "小助手") is False
