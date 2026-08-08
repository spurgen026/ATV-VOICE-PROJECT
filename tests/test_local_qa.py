import pytest

from resort_atv_voice.can_telemetry import TelemetryCache
from resort_atv_voice.local_qa import _has_vehicle_term_signal, answer_query

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
