import json
import logging

import faiss
import numpy as np

from .config import (
    FAISS_INDEX_PATH,
    GEMINI_EMBEDDING_MODEL,
    INDEX_DIR,
    INDEX_METADATA_PATH,
)
from .data_store import write_vehicle_status_document
from .documents import chunk_text, load_documents
from .gemini_client import client

EMBED_BATCH_SIZE = 90

logger = logging.getLogger(__name__)


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = client.models.embed_content(model=GEMINI_EMBEDDING_MODEL, contents=batch)
        if not resp.embeddings:
            # Offline, one-shot indexing script - fail loudly and clearly
            # rather than silently skip a batch, unlike rag.py's live
            # answer_query() which degrades gracefully instead.
            raise RuntimeError(f"embed_content returned no embeddings for batch starting at index {i}")
        for e in resp.embeddings:
            if e.values is None:
                raise RuntimeError(f"embed_content returned an embedding with no values (batch index {i})")
            vectors.append(e.values)
    return vectors


def build_index() -> None:
    write_vehicle_status_document()

    chunks = []
    metadata = []
    for source, text in load_documents():
        for chunk in chunk_text(text):
            chunks.append(chunk)
            metadata.append({"source": source, "text": chunk})

    if not chunks:
        logger.warning("No documents found in documents/ - nothing to index.")
        return

    sources = {m["source"] for m in metadata}
    logger.info("Embedding %d chunks from %d document(s)...", len(chunks), len(sources))
    vectors = embed_texts(chunks)

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype="float32"))

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(INDEX_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Indexed %d chunks -> %s", len(chunks), FAISS_INDEX_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    build_index()
