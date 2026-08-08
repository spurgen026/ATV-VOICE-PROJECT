import time

import pytest

from resort_atv_voice.can_telemetry import (
    CAN_ID_TO_FIELD,
    FIELD_TO_CAN_ID,
    TelemetryCache,
    decode_frame,
    describe,
    encode_frame,
    start_fake_ecu,
    start_listener,
)


def test_can_id_mapping_is_a_bijection():
    # Every field maps to exactly one CAN ID and back - a duplicate or
    # missing entry here would silently corrupt telemetry lookups.
    assert len(CAN_ID_TO_FIELD) == len(FIELD_TO_CAN_ID)
    for can_id, field in CAN_ID_TO_FIELD.items():
        assert FIELD_TO_CAN_ID[field] == can_id


@pytest.mark.parametrize("field", list(FIELD_TO_CAN_ID))
@pytest.mark.parametrize("value", [0, 1, 78, 255])
def test_encode_decode_round_trip(field, value):
    msg = encode_frame(field, value)
    decoded_field, decoded_value = decode_frame(msg)
    assert decoded_field == field
    assert decoded_value == value


def test_decode_unknown_can_id_returns_none():
    import can

    msg = can.Message(arbitration_id=0x999, data=bytes([1]), is_extended_id=False)
    field, value = decode_frame(msg)
    assert field is None
    assert value is None


def test_decode_empty_payload_returns_none():
    import can

    msg = can.Message(arbitration_id=0x100, data=b"", is_extended_id=False)
    field, value = decode_frame(msg)
    assert field is None
    assert value is None


def test_encode_rounds_and_wraps_out_of_byte_range_values():
    # A single-byte CAN payload can't hold anything above 255 - this
    # documents the actual (wrapping) behavior rather than leaving it as
    # an unverified assumption. Not a scenario the real 4-field dummy
    # schema hits today, but worth knowing if the schema grows.
    msg = encode_frame("battery_percent", 300)
    _, value = decode_frame(msg)
    assert value == 300 & 0xFF


class TestTelemetryCache:
    def test_missing_field_returns_none(self):
        cache = TelemetryCache()
        assert cache.get("battery_percent") is None

    def test_update_then_get(self):
        cache = TelemetryCache()
        cache.update("battery_percent", 78)
        assert cache.get("battery_percent") == 78

    def test_snapshot_is_a_copy_not_a_live_view(self):
        cache = TelemetryCache()
        cache.update("battery_percent", 78)
        snap = cache.snapshot()
        cache.update("battery_percent", 50)
        assert snap["battery_percent"] == 78
        assert cache.get("battery_percent") == 50


@pytest.mark.slow
def test_fake_ecu_to_listener_end_to_end():
    # Real integration test of the virtual-bus round trip used as a
    # stand-in for the not-yet-existing real ATV. Marked slow since it
    # depends on real thread timing, not because it loads a model.
    cache = TelemetryCache()
    channel = "test_channel_fake_ecu"
    ecu_thread, ecu_stop = start_fake_ecu(channel=channel, interval=0.2)
    listener_thread, listener_stop = start_listener(cache, channel=channel)
    try:
        time.sleep(1.0)
        for field in FIELD_TO_CAN_ID:
            assert cache.get(field) is not None
    finally:
        ecu_stop.set()
        listener_stop.set()
        ecu_thread.join(timeout=2)
        listener_thread.join(timeout=2)


def test_describe_never_leaves_a_raw_digit_for_hindi_or_tamil():
    # describe() is the last line of defense against MMS-TTS's digit-
    # vocabulary gap (see number_words.py) - a regression here would
    # silently mispronounce readings again.
    for language in ("hi", "ta"):
        text = describe("battery_percent", 78, language)
        assert "78" not in text


def test_describe_english_is_untouched():
    text = describe("battery_percent", 78, "en")
    assert "78" in text
