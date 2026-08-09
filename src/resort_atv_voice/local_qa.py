import difflib

from llama_cpp import Llama, LlamaGrammar

from .can_telemetry import TelemetryCache, describe
from .config import (
    CHAT_UNAVAILABLE_RESPONSE,
    DEFAULT_LANGUAGE,
    LANGUAGE_NAMES,
    TELEMETRY_FIELD_KEYWORDS,
    TELEMETRY_UNAVAILABLE_RESPONSE,
    VEHICLE_TERM_FUZZY_THRESHOLD,
    VEHICLE_TERM_KEYWORDS,
)
from .router import route
from .small_talk import try_small_talk_answer


def _fuzzy_contains(token: str, keyword: str, threshold: float) -> bool:
    """True if `keyword` fuzzy-matches `token` as a whole, OR as a
    substring embedded inside it.

    Found live 2026-08-08: Tamil STT output sometimes merges adjacent
    words with no space ('பாட்டிரியவளவு' = 'பாட்டிரிய' [battery] +
    'வளவு' [how much], no gap) - comparing the whole merged token
    against a short keyword tanks the SequenceMatcher ratio on length
    alone, even though the keyword is clearly present. That specific
    case made the vehicle-term safety net override a router decision
    that was actually correct, turning a right answer into a wrong one -
    the safety net built to prevent dangerous misroutes became the cause
    of a different failure. Sliding a keyword-length window across the
    token catches the embedded case without weakening the whole-token
    check for the common case (single, unmerged tokens)."""
    if difflib.SequenceMatcher(None, token, keyword).ratio() >= threshold:
        return True
    window = len(keyword)
    if window >= len(token):
        return False
    return any(
        difflib.SequenceMatcher(None, token[start : start + window], keyword).ratio() >= threshold
        for start in range(len(token) - window + 1)
    )


def _has_vehicle_term_signal(text: str) -> bool:
    """Safety net for the router's get_telemetry decisions - see
    VEHICLE_TERM_KEYWORDS in config.py for the reasoning and threshold
    tuning. Fuzzy (not exact) match, since STT garbling means the real
    word rarely comes through with exact spelling."""
    tokens = text.split()
    return any(
        _fuzzy_contains(token, keyword, VEHICLE_TERM_FUZZY_THRESHOLD)
        for token in tokens
        for keyword in VEHICLE_TERM_KEYWORDS
    )


def _detect_requested_fields(text: str) -> list[str]:
    """Which telemetry fields does this question's text actually mention?
    Compound questions ("battery, tire pressure, and speed?") need more
    than the router's single get_telemetry target - live testing 2026-08-08
    found a 3-field question silently answered with only the first field,
    the same bug class already fixed once before in this project's now-
    unused Gemini-era fallback (data_store.try_local_answer()), which
    never carried over to the current router+CAN-cache architecture.
    Same fuzzy-matching approach as _has_vehicle_term_signal(), just
    grouped per field instead of a single yes/no signal."""
    tokens = text.split()
    matched = []
    for field, keywords in TELEMETRY_FIELD_KEYWORDS.items():
        if any(
            _fuzzy_contains(token, keyword, VEHICLE_TERM_FUZZY_THRESHOLD)
            for token in tokens
            for keyword in keywords
        ):
            matched.append(field)
    return matched


def answer_query(
    router_llm: Llama,
    grammar: LlamaGrammar,
    cache: TelemetryCache,
    question: str,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """V3's fully-local answer path: small talk -> grammar-constrained
    router -> either a CAN cache lookup (get_telemetry) or a fixed
    honest "I don't know" response. Replaces V2's Gemini-backed
    answer_query() in the live app - no cloud call anywhere in this
    function.

    Free-form chat generation was retired from this path 2026-08-09,
    after repeated fabrication/wrong-language failures kept recurring no
    matter how many individual cases got patched (fabricated resort
    amenities, a fabricated speed reading, a fabricated battery
    percentage alongside an unrelated movie reference, a fabricated
    driving-safety tangent instead of a battery reading, an English
    question answered in Tamil due to history-language-bleed). Rather
    than keep patching generation failures one at a time, anything that
    isn't a clear small-talk or telemetry match now gets
    CHAT_UNAVAILABLE_RESPONSE instead of a generated one - see that
    constant in config.py for the full list of failures that led here.
    The `history` parameter (conversational memory across follow-ups)
    was dropped along with chat generation, the only thing that used it -
    main.py no longer tracks it either.

    small_talk.py now matches Hindi/Tamil phrases too, not just English -
    keyword-only per detected language, hand-written and not verified by a
    native speaker, so unexpected phrasing can still miss and fall through
    to the router path below (which still answers correctly, just
    without the canned fast-path response).
    """
    if language not in LANGUAGE_NAMES:
        language = DEFAULT_LANGUAGE

    small_talk = try_small_talk_answer(question, language)
    if small_talk:
        return small_talk

    # Literal keyword matches are checked BEFORE the router, and answered
    # directly from the CAN cache without ever asking the router to pick
    # a field - found live 2026-08-08 that trusting the router's own
    # target even when a keyword clearly named the field let two real
    # misroutes through (a battery question routed to speed/motor_temp, a
    # tire question routed to battery) despite the correct field being
    # unambiguous from the text itself. The router is still needed below
    # for cases with no literal keyword at all (e.g. "how fast am I
    # going," no literal "speed" word) - this only short-circuits the
    # clear-cut cases, which also skips a model call for them.
    requested_fields = _detect_requested_fields(question)
    if requested_fields:
        if len(requested_fields) == 1:
            target = requested_fields[0]
            value = cache.get(target)
            if value is None:
                return TELEMETRY_UNAVAILABLE_RESPONSE.get(
                    language, TELEMETRY_UNAVAILABLE_RESPONSE[DEFAULT_LANGUAGE]
                )
            return describe(target, value, language)

        # 2+ fields explicitly mentioned - a compound question. Answer
        # every field the text actually asked about, not just one -
        # live testing 2026-08-08 found a 3-field question silently
        # answered with only the first, the same bug class already fixed
        # once before in this project's now-unused Gemini-era fallback
        # (data_store.try_local_answer()), which never carried over to
        # the current router+CAN-cache architecture.
        answers = [
            describe(field, cache.get(field), language)
            for field in requested_fields
            if cache.get(field) is not None
        ]
        if not answers:
            return TELEMETRY_UNAVAILABLE_RESPONSE.get(
                language, TELEMETRY_UNAVAILABLE_RESPONSE[DEFAULT_LANGUAGE]
            )
        return " ".join(answers)

    decision = route(router_llm, grammar, question)
    if decision.get("action") == "get_telemetry" and _has_vehicle_term_signal(question):
        # No literal keyword matched a specific field above, but the
        # router still thinks this is telemetry AND some fuzzy
        # vehicle-term signal exists somewhere in the text - trust the
        # router's semantic understanding for cases the keyword scan
        # alone can't resolve. NOTE: "how fast am I going" is NOT such a
        # case, despite an earlier (wrong) claim in this project's history
        # that it was - re-verified 2026-08-09 and it still fails
        # _has_vehicle_term_signal() (no "fast" keyword; adding one caused
        # real false positives, see VEHICLE_TERM_KEYWORDS in config.py),
        # so it still falls through to CHAT_UNAVAILABLE_RESPONSE below.
        # Open gap, not fixed here - a real example of this branch's
        # actual reach being narrower than once assumed.
        # Named differently from the `target` above (not just style) -
        # decision is a plain dict, so decision.get("target") is honestly
        # Optional even though the GBNF grammar constrains get_telemetry
        # decisions to always include one in practice.
        router_target = decision.get("target")
        if not router_target:
            return TELEMETRY_UNAVAILABLE_RESPONSE.get(
                language, TELEMETRY_UNAVAILABLE_RESPONSE[DEFAULT_LANGUAGE]
            )
        value = cache.get(router_target)
        if value is None:
            return TELEMETRY_UNAVAILABLE_RESPONSE.get(
                language, TELEMETRY_UNAVAILABLE_RESPONSE[DEFAULT_LANGUAGE]
            )
        return describe(router_target, value, language)

    # Not a clear small-talk or telemetry match - see this function's
    # docstring and CHAT_UNAVAILABLE_RESPONSE in config.py for why this
    # is a fixed honest answer now, not a generated one.
    return CHAT_UNAVAILABLE_RESPONSE.get(language, CHAT_UNAVAILABLE_RESPONSE[DEFAULT_LANGUAGE])
