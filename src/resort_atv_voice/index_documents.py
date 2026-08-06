import json

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


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = client.models.embed_content(model=GEMINI_EMBEDDING_MODEL, contents=batch)
        vectors.extend(e.values for e in resp.embeddings)
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
        print("No documents found in documents/ - nothing to index.")
        return

    sources = {m["source"] for m in metadata}
    print(f"Embedding {len(chunks)} chunks from {len(sources)} document(s)...")
    vectors = embed_texts(chunks)

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype="float32"))

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(INDEX_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Indexed {len(chunks)} chunks -> {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    build_index()
