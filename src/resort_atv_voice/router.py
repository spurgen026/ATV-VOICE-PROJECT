import json
from typing import List, Optional, Tuple

from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LlamaGrammar

from .config import (
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


def generate_chat_reply(
    llm: Llama, query: str, history: Optional[History] = None, language: str = DEFAULT_LANGUAGE
) -> str:
    """Free-form (no grammar) local reply for anything the router classifies
    as chat and small_talk.py doesn't already handle - the same Qwen model
    the router uses, so this needs no cloud call. Not grounded in any
    document corpus (that's V2's parked Gemini/RAG path, not this)."""
    language_name = LANGUAGE_NAMES.get(language, LANGUAGE_NAMES[DEFAULT_LANGUAGE])
    system_prompt = LOCAL_CHAT_SYSTEM_PROMPT.format(language_name=language_name)
    messages = [{"role": "system", "content": system_prompt}]
    for q, a in (history or [])[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})

    result = llm.create_chat_completion(messages=messages, max_tokens=150, temperature=0.7)
    return result["choices"][0]["message"]["content"].strip()
