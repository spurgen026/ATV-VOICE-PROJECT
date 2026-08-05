import json

from .config import VEHICLE_STATE_PATH


def get_vehicle_state() -> dict:
    with open(VEHICLE_STATE_PATH, "r") as f:
        return json.load(f)
