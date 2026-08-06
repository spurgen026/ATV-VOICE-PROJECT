import json
from typing import List, Optional, Tuple

import faiss
import httpx
import numpy as np
from google.genai import errors, types

from .config import (
    FAISS_INDEX_PATH,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL,
    INDEX_METADATA_PATH,
    MAX_HISTORY_TURNS,
    RAG_SYSTEM_PROMPT,
    TOP_K_CHUNKS,
)
from .data_store import try_local_answer
from .gemini_client import client
from .small_talk import try_small_talk_answer

NOT_FOUND_RESPONSE = "I couldn't find that information in the provided documents."
UNAVAILABLE_RESPONSE = "Sorry, I'm having trouble reaching the assistant right now."
GEMINI_ERRORS = (errors.APIError, httpx.HTTPError)

History = List[Tuple[str, str]]


def load_index():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {FAISS_INDEX_PATH}. "
            "Run `python -m resort_atv_voice.index_documents` first."
        )
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(INDEX_METADATA_PATH) as f:
        metadata = json.load(f)
    return index, metadata


def _embed_query(question: str) -> np.ndarray:
    resp = client.models.embed_content(model=GEMINI_EMBEDDING_MODEL, contents=question)
    return np.array([resp.embeddings[0].values], dtype="float32")


def _search(index, metadata: list[dict], query_vector: np.ndarray) -> list[str]:
    k = min(TOP_K_CHUNKS, index.ntotal)
    if k == 0:
        return []
    _, neighbor_indices = index.search(query_vector, k)
    return [metadata[i]["text"] for i in neighbor_indices[0] if i != -1]


def _format_history(history: Optional[History]) -> str:
    if not history:
        return ""
    turns = history[-MAX_HISTORY_TURNS:]
    lines = ["Earlier in this conversation:"]
    for q, a in turns:
        lines.append(f"Rider asked: {q}\nYou answered: {a}")
    return "\n\n".join(lines) + "\n\n"


def answer_query(
    index, metadata: list[dict], question: str, history: Optional[History] = None
) -> str:
    small_talk = try_small_talk_answer(question)
    if small_talk:
        return small_talk

    try:
        chunks = _search(index, metadata, _embed_query(question))
        if not chunks:
            return NOT_FOUND_RESPONSE

        context = "\n\n".join(chunks)
        history_text = _format_history(history)
        prompt = f"{history_text}Context:\n{context}\n\nQuestion: {question}"
        resp = client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=RAG_SYSTEM_PROMPT),
        )
        return resp.text.strip()
    except GEMINI_ERRORS as exc:
        print(f"Gemini call failed: {exc}")
        local_answer = try_local_answer(question)
        if local_answer:
            print(f"Answered locally instead: {local_answer!r}")
            return local_answer
        return UNAVAILABLE_RESPONSE
