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
    # 2026-08-10: colloquial vocabulary added to TELEMETRY_FIELD_KEYWORDS -
    # charge/air/hot alternatives to the more literal battery/pressure/
    # temperature terms, each verified false-positive-safe before adding.
    ("சார்ஜ் எவ்வளவு இருக்கு", "ta battery via 'charge' colloquialism"),
    ("चार्ज कितना है", "hi battery via 'charge' colloquialism"),
    ("டயர்ல காற்று எவ்வளவு இருக்கு", "ta tire via 'air' instead of 'pressure'"),
    ("மோட்டர் சூடா இருக்கா", "ta motor via 'hot' colloquialism"),
    ("मोटर गर्म है क्या", "hi motor via alternate 'गर्म' spelling"),
]

FALSE_CASES = [
    ("வெளியா வதர் எப்படி இருக்கு?", "live weather misroute - the target case"),
    ("What's the weather like today?", "en weather"),
    ("வணக்கம்", "ta greeting"),
    ("boundary.", "live STT mis-transcription of battery"),
    ("Siri, bye.", "en farewell"),
    ("Bye.", "en farewell"),
    # 2026-08-10: locks in the false-positive fix found while adding the
    # 'air' colloquialism above - காத்து (spoken "air") was dropped from
    # TELEMETRY_FIELD_KEYWORDS specifically because it fuzzy-matched
    # காத்திரு ("wait," a genuinely common word) - this confirms the fix
    # holds, not just that the risky keyword was removed.
    ("இன்னும் கொஞ்சம் காத்திருங்க", "ta 'please wait a bit' - must not match dropped 'air' keyword"),
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


class TestGeminiDispatch:
    """Hybrid split (2026-08-10): anything that isn't small talk or a
    telemetry match reaches Gemini via cloud_chat.answer_via_gemini() -
    see local_qa.answer_query()'s docstring. Fast test: mocks route() and
    answer_via_gemini() directly so this verifies the dispatch wiring
    itself without needing a real model or network call - a regression
    here (e.g. the fallback silently calling something else, or not being
    reached at all) would be easy to miss otherwise."""

    def test_chat_action_dispatches_to_gemini(self, monkeypatch):
        import resort_atv_voice.local_qa as local_qa_module

        monkeypatch.setattr(local_qa_module, "route", lambda llm, grammar, q: {"action": "chat"})
        captured = {}

        def fake_answer_via_gemini(question, language):
            captured["call"] = (question, language)
            return "reply"

        monkeypatch.setattr(local_qa_module, "answer_via_gemini", fake_answer_via_gemini)

        response = answer_query(
            router_llm=object(),
            grammar=object(),
            cache=TelemetryCache(),
            question="some random question",
            language="en",
        )
        assert response == "reply"
        assert captured["call"] == ("some random question", "en")

    def test_language_is_passed_through_to_gemini(self, monkeypatch):
        import resort_atv_voice.local_qa as local_qa_module

        monkeypatch.setattr(local_qa_module, "route", lambda llm, grammar, q: {"action": "chat"})
        captured = {}

        def fake_answer_via_gemini(question, language):
            captured["language"] = language
            return "reply"

        monkeypatch.setattr(local_qa_module, "answer_via_gemini", fake_answer_via_gemini)

        answer_query(
            router_llm=object(),
            grammar=object(),
            cache=TelemetryCache(),
            question="ஏதோ ஒரு கேள்வி",
            language="ta",
        )
        assert captured["language"] == "ta"


@pytest.mark.slow
class TestAnswerQueryEndToEnd:
    """Real integration tests through the full answer_query() path - small
    talk -> router -> vehicle-term safety net -> CAN cache / Gemini
    dispatch. Slow: loads the real router LLM. The Gemini call itself is
    mocked (see test_cloud_chat.py for that logic tested directly) so
    this suite stays network-free and fast even with --run-slow."""

    @pytest.fixture
    def cache(self):
        c = TelemetryCache()
        c.update("battery_percent", 78)
        c.update("motor_temp_c", 42)
        c.update("speed_kmh", 0)
        c.update("tire_pressure_psi", 32)
        return c

    @pytest.fixture(autouse=True)
    def _no_real_gemini_calls(self, monkeypatch):
        # Autouse so it's impossible to forget on a new test in this class
        # and accidentally make a real network call during --run-slow.
        import resort_atv_voice.local_qa as local_qa_module

        monkeypatch.setattr(
            local_qa_module, "answer_via_gemini", lambda question, language: "MOCKED_GEMINI_REPLY"
        )

    def test_telemetry_question_answers_from_cache_not_the_model(self, router_llm, router_grammar, cache):
        response = answer_query(router_llm, router_grammar, cache, "What's the battery percentage?")
        assert "78" in response

    def test_weather_question_dispatches_to_gemini_never_states_a_fake_vehicle_reading(
        self, router_llm, router_grammar, cache
    ):
        # The exact case the vehicle-term safety net exists for - this
        # used to be answered by an unguarded local chat model that
        # fabricated a speed reading. Now it should reach the (mocked)
        # Gemini path instead of ever touching the CAN cache.
        response = answer_query(
            router_llm, router_grammar, cache, "வெளியா வதர் எப்படி இருக்கு?", language="ta"
        )
        assert response == "MOCKED_GEMINI_REPLY"

    def test_thanks_is_answered_by_small_talk_not_the_router(self, router_llm, router_grammar, cache):
        response = answer_query(router_llm, router_grammar, cache, "thank you")
        assert response == "You're welcome!"

    def test_missing_cache_value_gives_the_unavailable_response_not_a_crash(self, router_llm, router_grammar):
        empty_cache = TelemetryCache()
        response = answer_query(router_llm, router_grammar, empty_cache, "What's the battery percentage?")
        assert "don't have" in response.lower()

    def test_compound_question_answers_every_field_not_just_the_first(self, router_llm, router_grammar, cache):
        # The exact live-captured question that used to answer only
        # "The battery is at 78 percent," silently dropping tire
        # pressure and speed.
        response = answer_query(
            router_llm,
            router_grammar,
            cache,
            "Can you tell me about the battery, the tire pressure and the speed I am going on?",
        )
        assert "78" in response
        assert "32" in response
        assert "0" in response

    @pytest.mark.parametrize(
        "question,expected_substring",
        [
            ("காட்டிரியை உலோ இருக்கு", "எழுபத்தி எட்டு"),  # battery 78 - router misrouted this to speed/motor live
            ("டாயர் பிரச்சர் எவ்வளோ இருக்கு", "முப்பத்தி இரண்டு"),  # tire 32 - router misrouted this to battery live
            ("எவ்வளோ வேகத்தில நம்மும் போயிட்டிருக்கும்.", "பூஜ்யம்"),  # speed 0 - was hallucinated as 80-90 by chat
        ],
        ids=["ta battery misroute (was speed/motor)", "ta tire misroute (was battery)", "ta speed (was hallucinated)"],
    )
    def test_keyword_match_overrides_a_flaky_router_pick(
        self, router_llm, router_grammar, cache, question, expected_substring
    ):
        # Live-captured 2026-08-08: the router itself picked the wrong
        # field for the first two, and the vehicle-term safety net
        # blocked a *correct* router decision for the third (native
        # Tamil "வேகத்தில" wasn't in the keyword list, only the loanword
        # "ஸ்பீட்" was) - letting it fall through to chat, which invented
        # a fake speed reading. All three are now answered directly from
        # a literal keyword match, bypassing the router's field pick
        # entirely rather than trusting it.
        response = answer_query(router_llm, router_grammar, cache, question, language="ta")
        assert expected_substring in response
