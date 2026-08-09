"""Structural checks on the per-language phrase tables in config.py and
small_talk.py. These have no logic to speak of, which is exactly why they
need a test - it's easy to add a language or a telemetry field, update
three of the four dicts that need it, and never notice the fourth is
missing until a specific language/field combination is asked for live.
"""

from resort_atv_voice import config, small_talk

SUPPORTED_LANGUAGES = set(config.LANGUAGE_NAMES)


def test_language_names_is_en_hi_ta():
    assert SUPPORTED_LANGUAGES == {"en", "hi", "ta"}


def test_telemetry_field_phrases_covers_every_language():
    assert set(config.TELEMETRY_FIELD_PHRASES) == SUPPORTED_LANGUAGES


def test_telemetry_field_phrases_covers_every_field_for_every_language():
    field_sets = {lang: set(fields) for lang, fields in config.TELEMETRY_FIELD_PHRASES.items()}
    reference = field_sets["en"]
    for lang, fields in field_sets.items():
        assert fields == reference, f"{lang} covers {fields}, expected {reference}"


def test_telemetry_field_phrases_all_have_a_value_placeholder():
    for lang, fields in config.TELEMETRY_FIELD_PHRASES.items():
        for field, template in fields.items():
            assert "{value}" in template, f"{lang}/{field} template has no {{value}} placeholder"


def test_telemetry_unavailable_response_covers_every_language():
    assert set(config.TELEMETRY_UNAVAILABLE_RESPONSE) == SUPPORTED_LANGUAGES


def test_chat_unavailable_response_covers_every_language():
    assert set(config.CHAT_UNAVAILABLE_RESPONSE) == SUPPORTED_LANGUAGES


def test_small_talk_response_tables_cover_every_language():
    tables = [
        small_talk.THANKS_RESPONSE,
        small_talk.CAPABILITY_RESPONSE,
        small_talk.IDENTITY_RESPONSE,
        small_talk.GREETING_RESPONSE,
    ]
    for table in tables:
        assert set(table) == SUPPORTED_LANGUAGES


def test_small_talk_trigger_tables_cover_every_language():
    tables = [
        small_talk.GREETINGS,
        small_talk.THANKS,
        small_talk.IDENTITY,
        small_talk.CAPABILITY,
    ]
    for table in tables:
        assert set(table) == SUPPORTED_LANGUAGES


def test_no_response_table_has_an_empty_string():
    # An empty response would get passed straight to speak(), producing
    # dead air with no error - the same silent-failure shape as the
    # digit-vocabulary bug number_words.py exists to prevent.
    tables = [
        config.TELEMETRY_UNAVAILABLE_RESPONSE,
        config.CHAT_UNAVAILABLE_RESPONSE,
        small_talk.THANKS_RESPONSE,
        small_talk.CAPABILITY_RESPONSE,
        small_talk.IDENTITY_RESPONSE,
        small_talk.GREETING_RESPONSE,
    ]
    for table in tables:
        for lang, text in table.items():
            assert text.strip(), f"{lang} has an empty/whitespace-only response"


def test_vehicle_term_keywords_is_non_empty():
    assert len(config.VEHICLE_TERM_KEYWORDS) > 0
