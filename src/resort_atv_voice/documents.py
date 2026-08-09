from collections.abc import Iterator
from pathlib import Path

import docx
import pypdf

from .config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, DOCUMENTS_DIR

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


_READERS = {".txt": _read_txt, ".pdf": _read_pdf, ".docx": _read_docx}


def load_documents() -> Iterator[tuple[str, str]]:
    """Yields (source_filename, full_text) for every supported file in DOCUMENTS_DIR."""
    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if path.suffix.lower() in _READERS:
            yield path.name, _READERS[path.suffix.lower()](path)


def chunk_text(text: str) -> list[str]:
    """Splits text into overlapping fixed-size character chunks."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE_CHARS].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
