import json
import threading

import can

from .config import DEFAULT_LANGUAGE, TELEMETRY_FIELD_PHRASES, VEHICLE_STATE_PATH
from .number_words import to_words

# python-can's "virtual" interface only shares messages between Bus
# instances in the *same process* (see can/interfaces/virtual.py) - fine
# for the real "socketcan" interface on actual hardware later, where
# there's no separate "fake ECU" process at all, just one real bus.
CAN_CHANNEL = "resort_atv_dummy"

CAN_ID_TO_FIELD = {
    0x100: "battery_percent",
    0x101: "motor_temp_c",
    0x102: "speed_kmh",
    0x103: "tire_pressure_psi",
}
FIELD_TO_CAN_ID = {field: can_id for can_id, field in CAN_ID_TO_FIELD.items()}


def encode_frame(field: str, value: float) -> can.Message:
    can_id = FIELD_TO_CAN_ID[field]
    return can.Message(
        arbitration_id=can_id, data=bytes([int(round(value)) & 0xFF]), is_extended_id=False
    )


def decode_frame(msg: can.Message):
    field = CAN_ID_TO_FIELD.get(msg.arbitration_id)
    if field is None or not msg.data:
        return None, None
    return field, msg.data[0]


class TelemetryCache:
    """In-memory latest-value cache, updated by the background listener
    thread. Lookups never touch the bus, so a slow/stalled bus can't hang
    a query for a telemetry value."""

    def __init__(self):
        self._lock = threading.Lock()
        self._values: dict[str, int] = {}

    def update(self, field: str, value: int) -> None:
        with self._lock:
            self._values[field] = value

    def get(self, field: str):
        with self._lock:
            return self._values.get(field)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._values)


def start_listener(
    cache: TelemetryCache, channel: str = CAN_CHANNEL, stop_event: threading.Event | None = None
) -> tuple[threading.Thread, threading.Event]:
    """Starts a background thread that decodes CAN frames into `cache`.
    Returns (thread, stop_event) - set the event to stop the thread."""
    stop_event = stop_event or threading.Event()
    bus = can.interface.Bus(channel=channel, interface="virtual")

    def _run():
        try:
            while not stop_event.is_set():
                msg = bus.recv(timeout=0.5)
                if msg is None:
                    continue
                field, value = decode_frame(msg)
                if field is not None:
                    cache.update(field, value)
        finally:
            bus.shutdown()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, stop_event


def start_fake_ecu(
    channel: str = CAN_CHANNEL, interval: float = 1.0, stop_event: threading.Event | None = None
) -> tuple[threading.Thread, threading.Event]:
    """Publishes data/vehicle_state.json on the virtual CAN bus periodically,
    standing in for a real ECU until real ATV hardware exists (see CLAUDE.md
    "Version 3 Plan"). In-process, for the same reason as start_listener()'s
    module-level note - the virtual interface doesn't cross process
    boundaries. Real hardware later just needs start_listener() pointed at
    the real "socketcan" bus; this function goes away entirely then."""
    stop_event = stop_event or threading.Event()
    bus = can.interface.Bus(channel=channel, interface="virtual")
    with open(VEHICLE_STATE_PATH) as f:
        state = json.load(f)

    def _run():
        try:
            while not stop_event.is_set():
                for field, value in state.items():
                    bus.send(encode_frame(field, value))
                stop_event.wait(interval)
        finally:
            bus.shutdown()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, stop_event


def describe(field: str, value, language: str = DEFAULT_LANGUAGE) -> str:
    """Turns a raw telemetry value into a spoken sentence, in the rider's
    detected language. The number always comes from the CAN cache, never
    from a model, so this can't hallucinate a reading - it can only
    misreport which field a query asked about (a router-level concern, not
    this function's).

    Numbers are spelled out as words for hi/ta (to_words()) rather than
    left as digits - MMS-TTS's tokenizer has an incomplete digit
    vocabulary per language and silently speaks the wrong number for
    missing digits otherwise (see number_words.py for how this was
    found)."""
    phrases = TELEMETRY_FIELD_PHRASES.get(language, TELEMETRY_FIELD_PHRASES[DEFAULT_LANGUAGE])
    return phrases[field].format(value=to_words(value, language))
