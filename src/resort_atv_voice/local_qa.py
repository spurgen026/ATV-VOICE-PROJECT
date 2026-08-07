from typing import List, Optional, Tuple

from llama_cpp import Llama, LlamaGrammar

from .can_telemetry import TelemetryCache, describe
from .config import DEFAULT_LANGUAGE, LANGUAGE_NAMES, TELEMETRY_UNAVAILABLE_RESPONSE
from .router import generate_chat_reply, route
from .small_talk import try_small_talk_answer

History = List[Tuple[str, str]]


def answer_query(
    llm: Llama,
    grammar: LlamaGrammar,
    cache: TelemetryCache,
    question: str,
    history: Optional[History] = None,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """V3's fully-local answer path: small talk -> grammar-constrained
    router -> either a CAN cache lookup (get_telemetry) or a free-form
    local chat reply (chat). Replaces V2's Gemini-backed answer_query() in
    the live app - no cloud call anywhere in this function.

    small_talk.py's keyword matching is English-only (a known, tracked
    gap - see CLAUDE.md) - a Hindi/Tamil greeting falls through to the
    router/chat path below instead, which still answers correctly, just
    without the canned fast-path response.
    """
    if language not in LANGUAGE_NAMES:
        language = DEFAULT_LANGUAGE

    small_talk = try_small_talk_answer(question)
    if small_talk:
        return small_talk

    decision = route(llm, grammar, question)
    if decision.get("action") == "get_telemetry":
        value = cache.get(decision.get("target"))
        if value is None:
            return TELEMETRY_UNAVAILABLE_RESPONSE.get(
                language, TELEMETRY_UNAVAILABLE_RESPONSE[DEFAULT_LANGUAGE]
            )
        return describe(decision["target"], value, language)

    return generate_chat_reply(llm, question, history, language)
