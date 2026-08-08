import difflib
from typing import List, Optional, Tuple

from llama_cpp import Llama, LlamaGrammar

from .can_telemetry import TelemetryCache, describe
from .config import (
    DEFAULT_LANGUAGE,
    LANGUAGE_NAMES,
    RESORT_KNOWLEDGE_TRIGGERS,
    RESORT_KNOWLEDGE_UNAVAILABLE_RESPONSE,
    TELEMETRY_UNAVAILABLE_RESPONSE,
    VEHICLE_TERM_FUZZY_THRESHOLD,
    VEHICLE_TERM_KEYWORDS,
)
from .router import generate_chat_reply, route
from .small_talk import try_small_talk_answer

History = List[Tuple[str, str]]


def _has_vehicle_term_signal(text: str) -> bool:
    """Safety net for the router's get_telemetry decisions - see
    VEHICLE_TERM_KEYWORDS in config.py for the reasoning and threshold
    tuning. Fuzzy (not exact) match, since STT garbling means the real
    word rarely comes through with exact spelling."""
    tokens = text.split()
    return any(
        difflib.SequenceMatcher(None, token, keyword).ratio() >= VEHICLE_TERM_FUZZY_THRESHOLD
        for token in tokens
        for keyword in VEHICLE_TERM_KEYWORDS
    )


def answer_query(
    router_llm: Llama,
    grammar: LlamaGrammar,
    cache: TelemetryCache,
    question: str,
    chat_llm: Optional[Llama] = None,
    history: Optional[History] = None,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """V3's fully-local answer path: small talk -> grammar-constrained
    router -> either a CAN cache lookup (get_telemetry) or a free-form
    local chat reply (chat). Replaces V2's Gemini-backed answer_query() in
    the live app - no cloud call anywhere in this function.

    router_llm and chat_llm are deliberately different model instances
    (V3 hardening round 2) - the router is a strict classification task
    the small 1.5B model already handles well, chat is open-ended
    generation where a bigger model (3B) gives noticeably less rambling,
    more concise answers. chat_llm defaults to router_llm so this still
    works if a caller only has one model loaded (e.g. quick scripts/tests).

    small_talk.py now matches Hindi/Tamil phrases too, not just English -
    keyword-only per detected language, hand-written and not verified by a
    native speaker, so unexpected phrasing can still miss and fall through
    to the router/chat path below (which still answers correctly, just
    without the canned fast-path response).
    """
    if language not in LANGUAGE_NAMES:
        language = DEFAULT_LANGUAGE
    if chat_llm is None:
        chat_llm = router_llm

    small_talk = try_small_talk_answer(question, language)
    if small_talk:
        return small_talk

    decision = route(router_llm, grammar, question)
    if decision.get("action") == "get_telemetry" and _has_vehicle_term_signal(question):
        value = cache.get(decision.get("target"))
        if value is None:
            return TELEMETRY_UNAVAILABLE_RESPONSE.get(
                language, TELEMETRY_UNAVAILABLE_RESPONSE[DEFAULT_LANGUAGE]
            )
        return describe(decision["target"], value, language)

    # Deterministic gate, checked before the free-form model ever runs -
    # see RESORT_KNOWLEDGE_TRIGGERS in config.py for why prompting alone
    # wasn't trusted to stop fabricated amenities.
    lowered = question.lower()
    if any(trigger in lowered for trigger in RESORT_KNOWLEDGE_TRIGGERS):
        return RESORT_KNOWLEDGE_UNAVAILABLE_RESPONSE.get(
            language, RESORT_KNOWLEDGE_UNAVAILABLE_RESPONSE[DEFAULT_LANGUAGE]
        )

    return generate_chat_reply(chat_llm, question, history, language)
