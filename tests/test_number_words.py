import pytest

from resort_atv_voice.number_words import to_words

# Root cause this module exists at all: MMS-TTS's tokenizer has an
# incomplete digit vocabulary per language and silently mispronounces
# numbers it can't spell (see number_words.py's docstring). These are
# regression tests for that specific failure mode - the exact digits that
# were confirmed missing from each tokenizer's vocabulary.
HI_MISSING_DIGITS = "56789"
TA_MISSING_DIGITS = "8"


@pytest.mark.parametrize("value", [0, 1, 5, 9, 10, 15, 20, 42, 78, 99])
def test_hindi_covers_the_real_dummy_schema_range(value):
    words = to_words(value, "hi")
    assert words
    assert not words[0].isdigit()


@pytest.mark.parametrize("value", [0, 1, 5, 8, 10, 15, 20, 42, 78, 99])
def test_tamil_covers_the_real_dummy_schema_range(value):
    words = to_words(value, "ta")
    assert words
    assert not words[0].isdigit()


@pytest.mark.parametrize("value", [n for n in range(100) if any(d in HI_MISSING_DIGITS for d in str(n))])
def test_hindi_output_never_contains_a_missing_digit(value):
    # The whole point of this module: values containing 5/6/7/8/9 must be
    # spelled out as words, never left as a raw digit MMS-TTS would drop.
    words = to_words(value, "hi")
    assert not any(d.isdigit() for d in words)


@pytest.mark.parametrize("value", [n for n in range(100) if "8" in str(n)])
def test_tamil_output_never_contains_a_missing_digit(value):
    words = to_words(value, "ta")
    assert not any(d.isdigit() for d in words)


def test_hundred_hindi():
    assert to_words(100, "hi") == "सौ"


def test_hundred_tamil():
    assert to_words(100, "ta") == "நூறு"


@pytest.mark.parametrize("value", [101, 150, 199])
def test_hindi_101_to_199(value):
    words = to_words(value, "hi")
    assert not any(d.isdigit() for d in words)


@pytest.mark.parametrize("value", [101, 150, 199])
def test_tamil_101_to_199(value):
    words = to_words(value, "ta")
    assert not any(d.isdigit() for d in words)


def test_above_199_falls_back_to_digit_by_digit_not_a_guess():
    # Documented, deliberate degradation - not a silent wrong answer, but
    # a real limitation if the telemetry schema grows 3-digit fields.
    words = to_words(250, "hi")
    assert len(words.split()) == 3  # one word per digit: 2, 5, 0


def test_english_is_left_as_a_plain_numeral():
    # Piper/espeak already reads English digits correctly - only hi/ta
    # need the word-spelling workaround.
    assert to_words(78, "en") == "78"
