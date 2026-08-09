import json
import logging

from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LlamaGrammar

from .config import (
    CHAT_LANGUAGE_RETRY_ATTEMPTS,
    CHAT_MODEL_CONTEXT_SIZE,
    CHAT_MODEL_FILENAME,
    CHAT_MODEL_REPO,
    DEFAULT_LANGUAGE,
    HINDI_UNICODE_RANGE,
    LANGUAGE_NAMES,
    LOCAL_CHAT_SYSTEM_PROMPT,
    MAX_HISTORY_TURNS,
    QWEN_MODEL_FILENAME,
    QWEN_MODEL_REPO,
    ROUTER_CONTEXT_SIZE,
    ROUTER_FEWSHOT_EXAMPLES,
    ROUTER_GRAMMAR_PATH,
    ROUTER_SYSTEM_PROMPT,
    SCRIPT_MIN_RATIO,
    TAMIL_CHAT_MODEL_CONTEXT_SIZE,
    TAMIL_CHAT_MODEL_FILENAME,
    TAMIL_CHAT_MODEL_REPO,
    TAMIL_UNICODE_RANGE,
)

History = list[tuple[str, str]]

logger = logging.getLogger(__name__)


def load_router_model() -> Llama:
    # Not Llama.from_pretrained(repo_id=..., filename=...): found the hard
    # way that it always calls the HF Hub API to list/resolve the file even
    # when it's fully cached locally, which breaks true offline startup.
    # hf_hub_download() resolves straight from cache with zero network
    # calls once the file exists, so resolve the path ourselves instead.
    model_path = hf_hub_download(repo_id=QWEN_MODEL_REPO, filename=QWEN_MODEL_FILENAME)
    return Llama(model_path=model_path, n_ctx=ROUTER_CONTEXT_SIZE, verbose=False)


def load_chat_model() -> Llama:
    """Separate, larger model dedicated to generate_chat_reply() - see
    CHAT_MODEL_REPO in config.py for why this is a different model from
    the router's, not just a bigger version of the same one."""
    model_path = hf_hub_download(repo_id=CHAT_MODEL_REPO, filename=CHAT_MODEL_FILENAME)
    return Llama(model_path=model_path, n_ctx=CHAT_MODEL_CONTEXT_SIZE, verbose=False)


def load_tamil_chat_model() -> Llama:
    """Third chat model, used only for Tamil - see TAMIL_CHAT_MODEL_REPO
    in config.py for why Qwen's Tamil generation needed a genuinely
    Tamil-specialized model rather than a bigger general one."""
    model_path = hf_hub_download(repo_id=TAMIL_CHAT_MODEL_REPO, filename=TAMIL_CHAT_MODEL_FILENAME)
    return Llama(model_path=model_path, n_ctx=TAMIL_CHAT_MODEL_CONTEXT_SIZE, verbose=False)


def load_grammar() -> LlamaGrammar:
    return LlamaGrammar.from_file(str(ROUTER_GRAMMAR_PATH))


def route(llm: Llama, grammar: LlamaGrammar, query: str) -> dict:
    """Classifies a query as a telemetry lookup or general chat. The GBNF
    grammar constrains the model to one of two exact JSON shapes, so the
    only failure mode is picking the wrong one/wrong field - it can never
    hallucinate a number or malformed output."""
    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
    for example_query, example_decision in ROUTER_FEWSHOT_EXAMPLES:
        messages.append({"role": "user", "content": example_query})
        messages.append({"role": "assistant", "content": json.dumps(example_decision)})
    messages.append({"role": "user", "content": query})

    # The ignores below suppress create_chat_completion()'s stub typing
    # `messages`/the return value against llama-cpp-python's full
    # streaming/non-streaming API surface (a Union keyed on the `stream`
    # kwarg's value, which mypy can't narrow statically). We never pass
    # stream=True here, so at runtime this is always the plain response
    # dict with string content - confirmed by this whole project's
    # extensive live testing, not just assumed.
    result = llm.create_chat_completion(
        messages=messages,  # type: ignore[arg-type]
        grammar=grammar,
        max_tokens=40,
        temperature=0.0,
    )
    raw = result["choices"][0]["message"]["content"]  # type: ignore[index]
    return json.loads(raw)  # type: ignore[arg-type]


def _script_ratio(text: str, unicode_range: tuple[int, int]) -> float:
    """Fraction of `text`'s alphabetic characters that fall within
    `unicode_range`. Used to verify a reply actually came back in the
    requested script, instead of trusting the "Respond in {language_name}"
    prompt instruction - see generate_chat_reply()'s retry loop."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    in_range = sum(1 for c in letters if unicode_range[0] <= ord(c) <= unicode_range[1])
    return in_range / len(letters)


def _response_matches_language(text: str, language: str) -> bool:
    """True if `text` actually came back in the script `language` implies.
    Tamil/Hindi: most of the text should be in that script. English: most
    of the text should NOT be Tamil or Hindi script - found live
    2026-08-09 that an English question, asked after several Tamil turns
    in the same conversation, got answered in Tamil script despite
    language='en' being correctly detected - multi-turn history in a
    different language biases generation regardless of the target
    language, a problem that turned out symmetric across en/hi/ta, not
    Tamil-specific (see CHAT_LANGUAGE_RETRY_ATTEMPTS in config.py)."""
    if language == "ta":
        return _script_ratio(text, TAMIL_UNICODE_RANGE) >= SCRIPT_MIN_RATIO
    if language == "hi":
        return _script_ratio(text, HINDI_UNICODE_RANGE) >= SCRIPT_MIN_RATIO
    return (
        _script_ratio(text, TAMIL_UNICODE_RANGE) < SCRIPT_MIN_RATIO
        and _script_ratio(text, HINDI_UNICODE_RANGE) < SCRIPT_MIN_RATIO
    )


def generate_chat_reply(
    llm: Llama, query: str, history: History | None = None, language: str = DEFAULT_LANGUAGE
) -> str:
    """Free-form (no grammar) local reply. Not grounded in any document
    corpus (that's V2's parked Gemini/RAG path, not this). As of
    2026-08-09 this is parked, not called from the live app - see
    local_qa.answer_query(), which now returns a fixed honest "no
    information" response instead of generating anything for the cases
    that used to reach this function, after repeated fabrication/wrong-
    language failures. Kept (not deleted) in case a safer approach to
    free-form chat is revisited later, same "parked, not deleted"
    philosophy as the V2 Gemini/RAG code.

    Retries if the reply doesn't actually come back in the requested
    language's script - confirmed live 2026-08-08 for Tamil (asked the
    same question 3 times, got English back twice) and, after that fix
    was scoped Tamil-only, confirmed live 2026-08-09 that the same
    problem happens in the other direction too (an English question,
    asked after several Tamil turns in the same conversation, answered
    in Tamil) - multi-turn history in a different language biases
    generation regardless of the target language, symmetric across
    en/hi/ta. See CHAT_LANGUAGE_RETRY_ATTEMPTS in config.py."""
    language_name = LANGUAGE_NAMES.get(language, LANGUAGE_NAMES[DEFAULT_LANGUAGE])
    system_prompt = LOCAL_CHAT_SYSTEM_PROMPT.format(language_name=language_name)
    messages = [{"role": "system", "content": system_prompt}]
    for q, a in (history or [])[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})

    text = ""
    for attempt in range(CHAT_LANGUAGE_RETRY_ATTEMPTS):
        # Same stream=True-vs-False typing gap as route() above.
        result = llm.create_chat_completion(
            messages=messages, max_tokens=150, temperature=0.7  # type: ignore[arg-type]
        )
        content = result["choices"][0]["message"]["content"]  # type: ignore[index]
        text = (content or "").strip()  # type: ignore[union-attr]
        if _response_matches_language(text, language):
            return text
        logger.warning(
            "generate_chat_reply: attempt %d didn't come back in the expected language, retrying",
            attempt + 1,
        )
    return text
