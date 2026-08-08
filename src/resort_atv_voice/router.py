import json
from typing import List, Optional, Tuple

from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LlamaGrammar

from .config import (
    CHAT_LANGUAGE_RETRY_ATTEMPTS,
    CHAT_MODEL_CONTEXT_SIZE,
    CHAT_MODEL_FILENAME,
    CHAT_MODEL_REPO,
    DEFAULT_LANGUAGE,
    LANGUAGE_NAMES,
    LOCAL_CHAT_SYSTEM_PROMPT,
    MAX_HISTORY_TURNS,
    QWEN_MODEL_FILENAME,
    QWEN_MODEL_REPO,
    ROUTER_CONTEXT_SIZE,
    ROUTER_FEWSHOT_EXAMPLES,
    ROUTER_GRAMMAR_PATH,
    ROUTER_SYSTEM_PROMPT,
    TAMIL_CHAT_MODEL_CONTEXT_SIZE,
    TAMIL_CHAT_MODEL_FILENAME,
    TAMIL_CHAT_MODEL_REPO,
    TAMIL_SCRIPT_MIN_RATIO,
    TAMIL_UNICODE_RANGE,
)

History = List[Tuple[str, str]]


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

    result = llm.create_chat_completion(
        messages=messages,
        grammar=grammar,
        max_tokens=40,
        temperature=0.0,
    )
    raw = result["choices"][0]["message"]["content"]
    return json.loads(raw)


def _script_ratio(text: str, unicode_range: Tuple[int, int]) -> float:
    """Fraction of `text`'s alphabetic characters that fall within
    `unicode_range`. Used to verify a reply actually came back in the
    requested script, instead of trusting the "Respond in {language_name}"
    prompt instruction - see generate_chat_reply()'s retry loop."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    in_range = sum(1 for c in letters if unicode_range[0] <= ord(c) <= unicode_range[1])
    return in_range / len(letters)


def generate_chat_reply(
    llm: Llama, query: str, history: Optional[History] = None, language: str = DEFAULT_LANGUAGE
) -> str:
    """Free-form (no grammar) local reply for anything the router classifies
    as chat and small_talk.py doesn't already handle - the same Qwen model
    the router uses, so this needs no cloud call. Not grounded in any
    document corpus (that's V2's parked Gemini/RAG path, not this).

    For Tamil, retries if the reply doesn't actually come back in Tamil
    script - confirmed live 2026-08-08 that the "Respond in {language_name}"
    instruction alone is unreliable (2 of 3 identical attempts answered in
    English), the same unreliable-instruction-following pattern already
    seen elsewhere in this codebase. See CHAT_LANGUAGE_RETRY_ATTEMPTS in
    config.py."""
    language_name = LANGUAGE_NAMES.get(language, LANGUAGE_NAMES[DEFAULT_LANGUAGE])
    system_prompt = LOCAL_CHAT_SYSTEM_PROMPT.format(language_name=language_name)
    messages = [{"role": "system", "content": system_prompt}]
    for q, a in (history or [])[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})

    attempts = CHAT_LANGUAGE_RETRY_ATTEMPTS if language == "ta" else 1
    text = ""
    for attempt in range(attempts):
        result = llm.create_chat_completion(messages=messages, max_tokens=150, temperature=0.7)
        text = result["choices"][0]["message"]["content"].strip()
        if language != "ta" or _script_ratio(text, TAMIL_UNICODE_RANGE) >= TAMIL_SCRIPT_MIN_RATIO:
            return text
        print(f"generate_chat_reply: attempt {attempt + 1} didn't come back in Tamil script, retrying")
    return text
