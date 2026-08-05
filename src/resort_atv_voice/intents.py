from .data_store import get_vehicle_state

FALLBACK_RESPONSE = (
    "I can only tell you the battery level, motor temperature, speed, "
    "or tire pressure right now."
)


def handle_query(text: str) -> str:
    text = text.lower()
    state = get_vehicle_state()

    if "battery" in text:
        return f"The battery is at {state['battery_percent']} percent."

    if "motor" in text or "temperature" in text or "temp" in text:
        return f"The motor temperature is {state['motor_temp_c']} degrees Celsius."

    if "speed" in text or "fast" in text:
        return f"The current speed is {state['speed_kmh']} kilometers per hour."

    if "tire" in text or "tyre" in text or "pressure" in text:
        return f"The tire pressure is {state['tire_pressure_psi']} psi."

    return FALLBACK_RESPONSE
