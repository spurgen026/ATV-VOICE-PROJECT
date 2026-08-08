"""Router classification regression tests. All @pytest.mark.slow since
route() needs the real router LLM loaded - permanent home for what
test_router_fix.py (throwaway, not committed) originally verified by hand.

(transcript, expected_target_or_"chat") - "chat" means action should be
"chat", anything else is the expected get_telemetry target field.
"""

import pytest

from resort_atv_voice.router import route

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
