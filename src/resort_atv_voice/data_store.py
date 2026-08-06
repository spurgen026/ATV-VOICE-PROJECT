import json
from typing import Optional

from .config import DOCUMENTS_DIR, VEHICLE_STATE_PATH, VEHICLE_STATUS_DOC_PATH


def get_vehicle_state() -> dict:
    with open(VEHICLE_STATE_PATH, "r") as f:
        return json.load(f)


def try_local_answer(question: str) -> Optional[str]:
    """Keyword-matched answer straight from vehicle_state.json, no cloud
    call needed. Used as a fallback when Gemini is unreachable, so vehicle
    stats stay answerable offline even though document Q&A doesn't.
    Collects every matching stat rather than stopping at the first one, so
    compound questions ("battery and speed?") get a complete answer."""
    text = question.lower()
    state = get_vehicle_state()

    facts = []
    if "battery" in text:
        facts.append(f"the battery is at {state['battery_percent']} percent")
    if "motor" in text or "temperature" in text or "temp" in text:
        facts.append(f"the motor temperature is {state['motor_temp_c']} degrees Celsius")
    if "speed" in text or "fast" in text:
        facts.append(f"the current speed is {state['speed_kmh']} kilometers per hour")
    if "tire" in text or "tyre" in text or "pressure" in text:
        facts.append(f"the tire pressure is {state['tire_pressure_psi']} psi")

    if not facts:
        return None

    if len(facts) == 1:
        sentence = facts[0]
    else:
        sentence = ", ".join(facts[:-1]) + f", and {facts[-1]}"
    return sentence[0].upper() + sentence[1:] + "."


def write_vehicle_status_document() -> None:
    """Renders vehicle_state.json as a text document so it's indexable
    alongside other documents in the RAG corpus (see V2 plan)."""
    state = get_vehicle_state()
    text = (
        f"Battery level: {state['battery_percent']} percent.\n"
        f"Motor temperature: {state['motor_temp_c']} degrees Celsius.\n"
        f"Current speed: {state['speed_kmh']} kilometers per hour.\n"
        f"Tire pressure: {state['tire_pressure_psi']} psi.\n"
    )
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    VEHICLE_STATUS_DOC_PATH.write_text(text, encoding="utf-8")
