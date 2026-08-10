"""answer_via_gemini() is the one function in the live app that ever
touches the network (see local_qa.answer_query()'s docstring). These
tests mock the Gemini client directly - no real network calls, no API
key needed - covering the success path, the "empty response" case (the
SDK types this Optional, a real possibility per rag.py's own mypy-driven
fix), a genuine API failure, and that the offline/failure fallback is
picked per-language rather than always English.
"""

import httpx
import pytest

from resort_atv_voice import cloud_chat
from resort_atv_voice.config import CLOUD_CHAT_UNAVAILABLE_RESPONSE


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, response=None, exception=None):
        self._response = response
        self._exception = exception
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self._exception is not None:
            raise self._exception
        return self._response


class _FakeClient:
    def __init__(self, response=None, exception=None):
        self.models = _FakeModels(response, exception)


def test_successful_call_returns_the_response_text(monkeypatch):
    fake_client = _FakeClient(response=_FakeResponse("The weather looks clear today."))
    monkeypatch.setattr(cloud_chat, "client", fake_client)

    result = cloud_chat.answer_via_gemini("What's the weather like?", language="en")

    assert result == "The weather looks clear today."
    assert fake_client.models.calls[0]["contents"] == "What's the weather like?"


def test_response_text_is_stripped_of_surrounding_whitespace(monkeypatch):
    fake_client = _FakeClient(response=_FakeResponse("  padded answer  \n"))
    monkeypatch.setattr(cloud_chat, "client", fake_client)

    result = cloud_chat.answer_via_gemini("some question")

    assert result == "padded answer"


def test_empty_response_text_falls_back_to_fixed_response(monkeypatch):
    fake_client = _FakeClient(response=_FakeResponse(None))
    monkeypatch.setattr(cloud_chat, "client", fake_client)

    result = cloud_chat.answer_via_gemini("some question", language="en")

    assert result == CLOUD_CHAT_UNAVAILABLE_RESPONSE["en"]


def test_api_failure_falls_back_to_fixed_response_not_a_crash(monkeypatch):
    fake_client = _FakeClient(exception=httpx.HTTPError("connection failed"))
    monkeypatch.setattr(cloud_chat, "client", fake_client)

    result = cloud_chat.answer_via_gemini("some question", language="en")

    assert result == CLOUD_CHAT_UNAVAILABLE_RESPONSE["en"]


@pytest.mark.parametrize("language", ["en", "hi", "ta"])
def test_failure_fallback_is_picked_per_language(monkeypatch, language):
    fake_client = _FakeClient(exception=httpx.HTTPError("connection failed"))
    monkeypatch.setattr(cloud_chat, "client", fake_client)

    result = cloud_chat.answer_via_gemini("some question", language=language)

    assert result == CLOUD_CHAT_UNAVAILABLE_RESPONSE[language]


def test_unsupported_language_falls_back_to_default_response_on_failure(monkeypatch):
    fake_client = _FakeClient(exception=httpx.HTTPError("connection failed"))
    monkeypatch.setattr(cloud_chat, "client", fake_client)

    result = cloud_chat.answer_via_gemini("some question", language="fr")

    assert result == CLOUD_CHAT_UNAVAILABLE_RESPONSE["en"]


def test_system_prompt_names_the_correct_language(monkeypatch):
    fake_client = _FakeClient(response=_FakeResponse("ok"))
    monkeypatch.setattr(cloud_chat, "client", fake_client)

    cloud_chat.answer_via_gemini("ஏதோ ஒரு கேள்வி", language="ta")

    system_instruction = fake_client.models.calls[0]["config"].system_instruction
    assert "Tamil" in system_instruction
