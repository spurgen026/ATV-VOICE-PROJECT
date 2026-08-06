import json

import faiss
import httpx
import numpy as np
from google.genai import errors, types

from .config import (
    FAISS_INDEX_PATH,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL,
    INDEX_METADATA_PATH,
    RAG_SYSTEM_PROMPT,
    TOP_K_CHUNKS,
)
from .data_store import try_local_answer
from .gemini_client import client

NOT_FOUND_RESPONSE = "I couldn't find that information in the provided documents."
UNAVAILABLE_RESPONSE = "Sorry, I'm having trouble reaching the assistant right now."
GEMINI_ERRORS = (errors.APIError, httpx.HTTPError)


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


def answer_query(index, metadata: list[dict], question: str) -> str:
    try:
        chunks = _search(index, metadata, _embed_query(question))
        if not chunks:
            return NOT_FOUND_RESPONSE

        context = "\n\n".join(chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
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
