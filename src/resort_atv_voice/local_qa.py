import difflib

from llama_cpp import Llama, LlamaGrammar

from .can_telemetry import TelemetryCache, describe
from .cloud_chat import answer_via_gemini
from .config import (
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
    """Hybrid answer path (2026-08-10): small talk -> grammar-constrained
    router -> either a CAN cache lookup (get_telemetry, fully local and
    deterministic) or a Gemini chat reply (anything else). Telemetry and
    small talk never touch the network, regardless of connectivity - only
    the final fallback (weather, general knowledge, casual chat) makes a
    cloud call, via cloud_chat.answer_via_gemini(). See
    CLOUD_CHAT_SYSTEM_PROMPT in config.py for the reasoning: two prior
    local-generation attempts (a dedicated 3B chat model, then a
    Tamil-specialized 7B model) both kept fabricating things despite
    explicit prompt instructions not to, the same unreliable-uncertainty
    pattern this codebase has hit repeatedly at small local model sizes -
    Gemini replaces that generation entirely rather than being tuned
    further, while the actual vehicle-safety-critical path (telemetry)
    stays exactly as local and deterministic as it always was.

    small_talk.py now matches Hindi/Tamil phrases too, not just English -
    keyword-only per detected language, hand-written and not verified by a
    native speaker, so unexpected phrasing can still miss and fall through
    to the router/Gemini path below (which still answers correctly, just
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
        # so it still falls through to unguarded chat below. Open gap, not
        # fixed here - a real example of this branch's actual reach being
        # narrower than once assumed.
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

    # Not a clear small-talk or telemetry match - the one place this
    # function ever touches the network. See this function's docstring
    # and CLOUD_CHAT_SYSTEM_PROMPT in config.py for why Gemini handles
    # this instead of a local model, and cloud_chat.py for the offline/
    # failure fallback.
    return answer_via_gemini(question, language)
