"""
Standalone wrapper around can_telemetry.start_fake_ecu() for manual
testing. Simulates the ATV's ECU broadcasting telemetry on a CAN bus,
standing in for real vehicle hardware (no real ATV CAN bus exists yet -
see CLAUDE.md's "Version 3 Plan").

Caveat: python-can's "virtual" interface only shares messages between
Bus instances in the same process, so running this as a separate
process alongside a listener elsewhere won't connect them - main.py
starts its own fake ECU thread in-process instead of using this script.
This script is for manually inspecting bus traffic on its own. This
limitation goes away on real hardware, where "socketcan" is a real
shared bus, not an in-process queue.
"""

import time

from resort_atv_voice.can_telemetry import start_fake_ecu


def main():
    thread, stop_event = start_fake_ecu()
    print("Publishing data/vehicle_state.json on the virtual CAN channel every 1s. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
