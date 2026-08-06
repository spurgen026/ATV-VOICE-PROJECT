import json
from typing import Optional

from .config import DOCUMENTS_DIR, VEHICLE_STATE_PATH, VEHICLE_STATUS_DOC_PATH


def get_vehicle_state() -> dict:
    with open(VEHICLE_STATE_PATH, "r") as f:
        return json.load(f)


def try_local_answer(question: str) -> Optional[str]:
    """Keyword-matched answer straight from vehicle_state.json, no cloud
    call needed. Used as a fallback when Gemini is unreachable, so vehicle
    stats stay answerable offline even though document Q&A doesn't."""
    text = question.lower()
    state = get_vehicle_state()

    if "battery" in text:
        return f"The battery is at {state['battery_percent']} percent."
    if "motor" in text or "temperature" in text or "temp" in text:
        return f"The motor temperature is {state['motor_temp_c']} degrees Celsius."
    if "speed" in text or "fast" in text:
        return f"The current speed is {state['speed_kmh']} kilometers per hour."
    if "tire" in text or "tyre" in text or "pressure" in text:
        return f"The tire pressure is {state['tire_pressure_psi']} psi."
    return None


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
