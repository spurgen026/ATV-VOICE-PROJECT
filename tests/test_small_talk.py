import pytest

from resort_atv_voice.small_talk import try_small_talk_answer

GREETINGS = {"en": "hello there", "hi": "नमस्ते", "ta": "வணக்கம்"}
THANKS = {"en": "thank you so much", "hi": "धन्यवाद दोस्त", "ta": "நன்றி"}
IDENTITY = {"en": "who are you", "hi": "आप कौन हैं", "ta": "நீ யார்"}
CAPABILITY = {"en": "what can you do", "hi": "क्या कर सकते हो", "ta": "என்ன செய்ய முடியும்"}


@pytest.mark.parametrize("language", ["en", "hi", "ta"])
def test_greeting_matches(language):
    assert try_small_talk_answer(GREETINGS[language], language) is not None


@pytest.mark.parametrize("language", ["en", "hi", "ta"])
def test_thanks_matches(language):
    assert try_small_talk_answer(THANKS[language], language) is not None


@pytest.mark.parametrize("language", ["en", "hi", "ta"])
def test_identity_matches(language):
    assert try_small_talk_answer(IDENTITY[language], language) is not None


@pytest.mark.parametrize("language", ["en", "hi", "ta"])
def test_capability_matches(language):
    assert try_small_talk_answer(CAPABILITY[language], language) is not None


@pytest.mark.parametrize("language", ["en", "hi", "ta"])
def test_unrelated_question_does_not_match(language):
    # A real telemetry question must never accidentally get swallowed by
    # small talk - it needs to reach the router instead.
    unrelated = {
        "en": "What's the battery percentage?",
        "hi": "बैटरी कितनी है",
        "ta": "பேட்டரி எவ்வளவு இருக்கு",
    }
    assert try_small_talk_answer(unrelated[language], language) is None


def test_unknown_language_falls_back_to_english_table():
    # language codes outside en/hi/ta shouldn't crash - GREETING_RESPONSE.get()
    # etc. all fall back to DEFAULT_LANGUAGE's table.
    assert try_small_talk_answer("hello there", "fr") is not None


def test_empty_question_does_not_match():
    assert try_small_talk_answer("", "en") is None
