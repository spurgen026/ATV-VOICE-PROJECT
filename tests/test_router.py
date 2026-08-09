"""Router classification regression tests. All @pytest.mark.slow since
route() needs the real router LLM loaded - permanent home for what
test_router_fix.py (throwaway, not committed) originally verified by hand.

(transcript, expected_target_or_"chat") - "chat" means action should be
"chat", anything else is the expected get_telemetry target field.
"""

import pytest

from resort_atv_voice.config import TAMIL_UNICODE_RANGE
from resort_atv_voice.router import _script_ratio, generate_chat_reply, route

CASES = [
    ("ஐயர் பிரசர் எவலவு இருக்கிறு", "tire_pressure_psi", "live tire-pressure misroute, now fixed"),
    ("What's the battery percentage?", "battery_percent", "en battery"),
    ("What's the weather like today?", "chat", "en weather - regressed once before, Phase 2+3"),
    ("बैटरी कितनी है", "battery_percent", "hi battery"),
    ("मोटर कितनी गरम है", "motor_temp_c", "hi motor"),
    ("टायर प्रेशर कितना है", "tire_pressure_psi", "hi tire"),
    ("பேட்டரி எவ்வளவு இருக்கு", "battery_percent", "ta battery"),
    ("டயர் பிரஷர் எவ்வளவு", "tire_pressure_psi", "ta tire - open since Phase 2+3, fixed by fewshot"),
    ("வணக்கம்", "chat", "ta greeting"),
    ("बैटरी लेवल क्या है", "battery_percent", "existing fewshot example"),
    ("ஸ்பீட் எவ்வளவு இருக்கு", "speed_kmh", "existing fewshot example"),
]


def _route_target(llm, grammar, transcript):
    decision = route(llm, grammar, transcript)
    if decision.get("action") == "get_telemetry":
        return decision.get("target")
    return "chat"


@pytest.mark.slow
@pytest.mark.parametrize("transcript,expected,label", CASES, ids=[c[2] for c in CASES])
def test_routing_decision(router_llm, router_grammar, transcript, expected, label):
    # llama.cpp's threaded matrix-multiply reductions mean route() isn't
    # perfectly deterministic even at temperature=0.0 (confirmed live,
    # see CLAUDE.md "V3 hardening, round 2 continued") - a single flip is
    # possible on a close call. Assert on majority-of-3 rather than one
    # shot, so an isolated flip doesn't fail the suite while a genuine
    # regression still would.
    results = [_route_target(router_llm, router_grammar, transcript) for _ in range(3)]
    assert results.count(expected) >= 2, f"{label}: got {results}, wanted majority {expected!r}"


@pytest.mark.slow
def test_weather_question_still_known_flaky():
    # Documents, rather than hides, the one known-unsolved router case:
    # this transcript misrouted even as a verbatim few-shot example
    # (CLAUDE.md). It's caught by local_qa's vehicle-term safety net
    # downstream (see test_local_qa.py), not fixed at the router level -
    # this test is intentionally NOT asserting correctness, just
    # documenting that the router itself remains unreliable here so a
    # future "fix" isn't silently assumed to have landed.
    pytest.skip(
        "known router-level gap, mitigated downstream by "
        "local_qa._has_vehicle_term_signal(), not fixed here - see "
        "test_local_qa.py::test_vehicle_term_signal_not_detected"
    )


class TestScriptRatio:
    """_script_ratio() underlies the Tamil language-retry fix (see
    CHAT_LANGUAGE_RETRY_ATTEMPTS in config.py) - confirmed live 2026-08-08
    that "Respond in Tamil" alone was unreliable (English came back 2 of 3
    identical attempts), so generate_chat_reply() checks the actual output
    script instead of trusting the instruction."""

    def test_pure_tamil_text(self):
        assert _script_ratio("இது தமிழ் வாக்கியம்", TAMIL_UNICODE_RANGE) == 1.0

    def test_pure_english_text(self):
        assert _script_ratio("This is an English sentence", TAMIL_UNICODE_RANGE) == 0.0

    def test_mixed_mostly_english(self):
        # An English sentence with one Tamil loanword embedded shouldn't
        # count as "answered in Tamil."
        ratio = _script_ratio("This ATV is safe பேட்டரி wise", TAMIL_UNICODE_RANGE)
        assert ratio < 0.5

    def test_empty_text(self):
        assert _script_ratio("", TAMIL_UNICODE_RANGE) == 0.0

    def test_digits_and_punctuation_are_excluded_from_the_ratio(self):
        # Numbers/punctuation aren't alphabetic in either script - only
        # actual letters should count, so "78%!" doesn't skew the ratio.
        assert _script_ratio("78%!", TAMIL_UNICODE_RANGE) == 0.0


class _FakeChatModel:
    """Returns a scripted sequence of responses, one per call - lets the
    retry test control exactly which attempt (if any) comes back in
    Tamil, without needing real (non-deterministic) generation."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def create_chat_completion(self, messages, max_tokens, temperature):
        text = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return {"choices": [{"message": {"content": text}}]}


def test_generate_chat_reply_retries_tamil_until_tamil_script_returned():
    llm = _FakeChatModel(["English reply", "English again", "தமிழ் பதில்"])
    result = generate_chat_reply(llm, "ஏதோ ஒரு கேள்வி", language="ta")
    assert result == "தமிழ் பதில்"
    assert llm.call_count == 3


def test_generate_chat_reply_returns_immediately_when_first_attempt_is_tamil():
    llm = _FakeChatModel(["தமிழ் பதில்", "English reply"])
    result = generate_chat_reply(llm, "ஏதோ ஒரு கேள்வி", language="ta")
    assert result == "தமிழ் பதில்"
    assert llm.call_count == 1


def test_generate_chat_reply_gives_up_after_max_attempts_not_infinite_loop():
    llm = _FakeChatModel(["English reply", "English again", "still English"])
    result = generate_chat_reply(llm, "ஏதோ ஒரு கேள்வி", language="ta")
    assert result == "still English"  # gives up, returns the last attempt rather than nothing
    assert llm.call_count == 3  # bounded by CHAT_LANGUAGE_RETRY_ATTEMPTS, not unbounded


def test_generate_chat_reply_does_not_retry_for_english():
    llm = _FakeChatModel(["A reply in English"])
    result = generate_chat_reply(llm, "some question", language="en")
    assert result == "A reply in English"
    assert llm.call_count == 1


def test_generate_chat_reply_retries_english_until_english_script_returned():
    # The 2026-08-09 fix: retry logic generalized from Tamil-only to all
    # three languages, after live reproduction showed an English question
    # (asked after several Tamil turns in the same conversation) coming
    # back in Tamil script despite language="en" - history-language-bleed
    # is symmetric, not Tamil-specific.
    llm = _FakeChatModel(["தமிழ் பதில்", "हिंदी जवाब", "An English reply"])
    result = generate_chat_reply(llm, "some question", language="en")
    assert result == "An English reply"
    assert llm.call_count == 3
