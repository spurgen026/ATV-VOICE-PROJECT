import pytest

from resort_atv_voice.can_telemetry import TelemetryCache
from resort_atv_voice.local_qa import (
    _detect_requested_fields,
    _fuzzy_contains,
    _has_vehicle_term_signal,
    answer_query,
)

# Every real transcript captured during this project's live testing that
# was used to tune the fuzzy vehicle-term safety net (see CLAUDE.md "V3
# hardening round 2 continued" and "Router safety net for the last known
# dangerous misroute") - kept here as permanent regression coverage
# instead of the throwaway scripts these were originally verified with.
TRUE_CASES = [
    ("ஐயர் பிரசர் எவலவு இருக்கிறு", "live tire-pressure misroute"),
    ("What's the battery percentage?", "en battery"),
    ("बैटरी कितनी है", "hi battery"),
    ("मोटर कितनी गरम है", "hi motor"),
    ("टायर प्रेशर कितना है", "hi tire"),
    ("பேட்டரி எவ்வளவு இருக்கு", "ta battery"),
    ("டயர் பிரஷர் எவ்வளவு", "ta tire"),
    ("बैटरी लेवल क्या है", "existing router fewshot example"),
    ("ஸ்பீட் எவ்வளவு இருக்கு", "existing router fewshot example"),
    ("நான் எவளோ ச்பீட்டில போயிட்டிருக்கேன்.", "live speed question, correctly routed"),
    ("what speed i am going in", "en speed"),
    ("பாட்டிரியவளவு இருக்கு", "live: STT merged 'battery'+'how much' with no space"),
]

FALSE_CASES = [
    ("வெளியா வதர் எப்படி இருக்கு?", "live weather misroute - the target case"),
    ("What's the weather like today?", "en weather"),
    ("வணக்கம்", "ta greeting"),
    ("boundary.", "live STT mis-transcription of battery"),
    ("Siri, bye.", "en farewell"),
    ("Bye.", "en farewell"),
]


@pytest.mark.parametrize("transcript,label", TRUE_CASES, ids=[c[1] for c in TRUE_CASES])
def test_vehicle_term_signal_detected(transcript, label):
    assert _has_vehicle_term_signal(transcript) is True


@pytest.mark.parametrize("transcript,label", FALSE_CASES, ids=[c[1] for c in FALSE_CASES])
def test_vehicle_term_signal_not_detected(transcript, label):
    assert _has_vehicle_term_signal(transcript) is False


class TestFuzzyContains:
    """The windowed substring match found live 2026-08-08: STT sometimes
    merges adjacent Tamil words with no space, and a whole-token-only
    comparison tanks on the length mismatch even though the keyword is
    clearly present inside. This specific bug made the vehicle-term
    safety net override a *correct* router decision into a wrong one -
    worth its own direct tests, not just coverage via the higher-level
    functions that use it."""

    def test_exact_match(self):
        assert _fuzzy_contains("battery", "battery", 0.8) is True

    def test_keyword_embedded_in_a_longer_merged_token(self):
        # 'பாட்டிரியவளவு' = 'பாட்டிரிய' (battery-ish) + 'வளவு' (how much),
        # no space - the exact live-captured failure.
        assert _fuzzy_contains("பாட்டிரியவளவு", "பாட்டரி", 0.8) is True

    def test_unrelated_short_word_does_not_match(self):
        # Must not become so lenient that any short word matches anything -
        # this is a safety net, false positives here are the dangerous
        # direction (see VEHICLE_TERM_FUZZY_THRESHOLD's reasoning).
        assert _fuzzy_contains("இருக்கு", "பேட்டரி", 0.8) is False

    def test_keyword_longer_than_token_does_not_match(self):
        assert _fuzzy_contains("கா", "பேட்டரி", 0.8) is False


class TestDetectRequestedFields:
    """Compound-question fact-dropping bug, found live 2026-08-08:
    'battery, tire pressure, and speed?' answered only the first field.
    Same bug class already fixed once before in this project's now-unused
    Gemini-era fallback (data_store.try_local_answer()), which never
    carried over to the current router+CAN-cache architecture - fixed
    properly this time via _detect_requested_fields()."""

    def test_single_field_question(self):
        assert _detect_requested_fields("What's the battery percentage?") == ["battery_percent"]

    def test_no_field_mentioned(self):
        assert _detect_requested_fields("What's the weather like today?") == []

    def test_the_exact_live_compound_question_that_dropped_two_facts(self):
        fields = _detect_requested_fields(
            "Can you tell me about the battery, the tire pressure and the speed I am going on?"
        )
        assert set(fields) == {"battery_percent", "tire_pressure_psi", "speed_kmh"}

    def test_the_exact_live_followup_that_dropped_a_fact(self):
        fields = _detect_requested_fields("What about the tire pressure and the speed?")
        assert set(fields) == {"tire_pressure_psi", "speed_kmh"}

    def test_field_order_is_stable_regardless_of_mention_order(self):
        # Order follows TELEMETRY_FIELD_KEYWORDS (battery -> tire -> motor
        # -> speed), not the order the rider said them in - documented
        # behavior, not a bug, so a spoken answer always has consistent
        # ordering across turns.
        fields = _detect_requested_fields("speed and then battery please")
        assert fields == ["battery_percent", "speed_kmh"]


def test_resort_knowledge_trigger_catches_activity_questions():
    from resort_atv_voice.config import RESORT_KNOWLEDGE_TRIGGERS

    assert any(t in "what activities are there".lower() for t in RESORT_KNOWLEDGE_TRIGGERS)


def test_resort_knowledge_trigger_does_not_catch_telemetry_questions():
    from resort_atv_voice.config import RESORT_KNOWLEDGE_TRIGGERS

    assert not any(t in "what is my battery percent".lower() for t in RESORT_KNOWLEDGE_TRIGGERS)


@pytest.mark.slow
class TestAnswerQueryEndToEnd:
    """Real integration tests through the full answer_query() path - small
    talk -> router -> vehicle-term safety net -> CAN cache / chat
    generation. Slow: loads the router and chat LLMs."""

    @pytest.fixture
    def cache(self):
        c = TelemetryCache()
        c.update("battery_percent", 78)
        c.update("motor_temp_c", 42)
        c.update("speed_kmh", 0)
        c.update("tire_pressure_psi", 32)
        return c

    def test_telemetry_question_answers_from_cache_not_the_model(
        self, router_llm, router_grammar, chat_llm, cache
    ):
        response = answer_query(
            router_llm, router_grammar, cache, "What's the battery percentage?", chat_llm=chat_llm
        )
        assert "78" in response

    def test_weather_question_never_states_a_fake_vehicle_reading(
        self, router_llm, router_grammar, chat_llm, cache
    ):
        # The exact case the safety net exists for - regardless of what
        # the chat model's answer actually says, it must never claim a
        # specific telemetry number for an unrelated question.
        response = answer_query(
            router_llm,
            router_grammar,
            cache,
            "வெளியா வதர் எப்படி இருக்கு?",
            chat_llm=chat_llm,
            language="ta",
        )
        assert "பூஜ்யம்" not in response  # the (fabricated) speed reading this used to produce

    def test_thanks_is_answered_by_small_talk_not_the_router(
        self, router_llm, router_grammar, chat_llm, cache
    ):
        response = answer_query(
            router_llm, router_grammar, cache, "thank you", chat_llm=chat_llm
        )
        assert response == "You're welcome!"

    def test_missing_cache_value_gives_the_unavailable_response_not_a_crash(
        self, router_llm, router_grammar, chat_llm
    ):
        empty_cache = TelemetryCache()
        response = answer_query(
            router_llm, router_grammar, empty_cache, "What's the battery percentage?", chat_llm=chat_llm
        )
        assert "don't have" in response.lower()

    def test_compound_question_answers_every_field_not_just_the_first(
        self, router_llm, router_grammar, chat_llm, cache
    ):
        # The exact live-captured question that used to answer only
        # "The battery is at 78 percent," silently dropping tire
        # pressure and speed.
        response = answer_query(
            router_llm,
            router_grammar,
            cache,
            "Can you tell me about the battery, the tire pressure and the speed I am going on?",
            chat_llm=chat_llm,
        )
        assert "78" in response
        assert "32" in response
        assert "0" in response
