"""Hybrid cloud/local split (2026-08-10) - the one function in the live
app that ever makes a network call. Everything else (wake word, STT,
telemetry routing, small talk, TTS) stays fully local regardless of
network state; only local_qa.answer_query()'s final fallback (anything
that isn't small talk or a telemetry match) reaches this module. See
CLOUD_CHAT_SYSTEM_PROMPT in config.py for why Gemini replaces the local
chat_llm/tamil_chat_llm models here rather than just tuning them further.
"""

import logging

import httpx
from google.genai import errors, types

from .config import (
    CLOUD_CHAT_SYSTEM_PROMPT,
    CLOUD_CHAT_UNAVAILABLE_RESPONSE,
    DEFAULT_LANGUAGE,
    GEMINI_CHAT_MODEL,
    LANGUAGE_NAMES,
)
from .gemini_client import client

logger = logging.getLogger(__name__)


class _EmptyGeminiResponse(Exception):
    """Raised when Gemini returns a response with no usable content (e.g.
    blocked by a safety filter) - the SDK types this field Optional, a
    real possibility, not assumed away. Same pattern as rag.py's version
    of this class - kept separate rather than imported from that parked,
    unused module, so this live-path module doesn't depend on it."""


GEMINI_ERRORS = (errors.APIError, httpx.HTTPError, _EmptyGeminiResponse)


def answer_via_gemini(question: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Answers anything that isn't small talk or a telemetry match, via
    Gemini. Falls back to a fixed, honest response on any failure (no
    internet, API error, blocked content) rather than the local chat
    models - see CLOUD_CHAT_UNAVAILABLE_RESPONSE in config.py for why."""
    language_name = LANGUAGE_NAMES.get(language, LANGUAGE_NAMES[DEFAULT_LANGUAGE])
    system_prompt = CLOUD_CHAT_SYSTEM_PROMPT.format(language_name=language_name)
    try:
        resp = client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=question,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        if resp.text is None:
            raise _EmptyGeminiResponse("generate_content returned no text (possibly blocked)")
        return resp.text.strip()
    except GEMINI_ERRORS:
        logger.exception("Gemini chat call failed, falling back to fixed response")
        return CLOUD_CHAT_UNAVAILABLE_RESPONSE.get(language, CLOUD_CHAT_UNAVAILABLE_RESPONSE[DEFAULT_LANGUAGE])
