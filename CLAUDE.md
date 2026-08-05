# Resort ATV Voice Assistant

A voice assistant embedded directly in an EV ATV (electric all-terrain
vehicle) — not a phone app. Mics and speakers are physically mounted on
the vehicle so anyone can talk to it directly. Wake-word triggered,
interaction model similar to Gemini Live.

This file is the seed context for the project. The canonical, evolving
record of this project lives in the Obsidian vault at
`D:\obsidian\crytonite\wiki\projects\resort-atv-voice-assistant.md` —
update that page (and this file, if it drifts) as the project
progresses, rather than letting the two go out of sync.

## Product vision (future — not being built yet)

The vehicle itself is being built toward full autonomy. The voice
assistant is meant to become the primary interface to that autonomy.
Planned features, all future scope:

- **Return to base station** — the ATV autonomously drives itself back
  to a charging base station.
- **Ride booking** — like Ola/Uber, but slot-based: riders pick a time
  slot rather than hailing on demand.
- **Sightseeing** — the assistant guides riders around resort locations.
- **Table reservations** — book a table at a hotel/restaurant within the
  resort, by voice.
- **Geofencing** — warns the rider if the vehicle leaves resort
  premises.

Positioning: the vehicle should feel like a **companion**, not just a
tool.

**Market:** resorts. ATVs are deployed on-site and need to be usable by
any guest there.

## What's actually being built now: V1

A deliberately minimal first prototype. Nothing above this list is in
scope yet.

- **Runs entirely on a laptop.** No real ATV hardware yet. Target
  production hardware (mini PC / Raspberry Pi / NVIDIA Jetson) is not
  decided.
- **Audio I/O:** headphones connected to the laptop (mic + speaker),
  standing in for the vehicle's future onboard mics/speakers.
- **Trigger:** wake word, Gemini-Live-style — the assistant listens for
  a wake word, then captures the user's spoken query.
- **Data:** no live vehicle telemetry yet. Battery percentage, motor
  temperature, speed, and tire pressure are **hardcoded dummy data** in
  a local folder/database, standing in for a real data feed.
- **Supported queries (v1):** battery percentage, motor temperature,
  speed, tire pressure.

## V1 implementation (decided 2026-08-05)

- **Stack:** openWakeWord (wake word) + faster-whisper (STT) + Piper
  (TTS) — all local/open-source, chosen over the Gemini Live API
  because its free tier is rate-capped/testing-only, unsuited to a
  continuously-listening device.
- **Language:** Python 3.9.
- **Layout:** `src/resort_atv_voice/` package (`wake_word.py`, `stt.py`,
  `tts.py`, `intents.py`, `data_store.py`, `config.py`, `main.py`);
  dummy data in `data/vehicle_state.json`; downloaded models in
  `models/` (gitignored).
- **Wake word:** placeholder pretrained `hey_jarvis_v0.1` ("Hey
  Jarvis") — no resort/vehicle-branded wake word trained yet. Threshold
  tuned to 0.3 (down from openWakeWord's 0.5 default) after live
  testing showed missed detections at 0.5.
- **Query capture:** records until trailing silence (RMS-threshold
  based), bounded by a max duration and a speech-start timeout, so it
  doesn't hang or clip a query — replaced the original fixed
  4-second recording after live testing showed clipping.
- Live-tested end to end on the dev laptop with real headphones: wake
  word, STT, intent matching, and TTS all confirmed working.
- Run with: `.venv\Scripts\python -m resort_atv_voice.main` (needs
  headphones connected, package installed via `pip install -e .`).

## Open questions (not yet decided)

- Target production hardware: mini PC vs. Raspberry Pi vs. NVIDIA
  Jetson.
- How the "accessible anywhere in the resort" requirement gets met
  eventually — single onboard unit vs. distributed mics/speakers — not
  addressed by the laptop prototype.
- A real, resort/vehicle-branded wake word.
- Not yet a git repo — no version control on this work yet.

## Unrelated sibling project — do not confuse

There's a separate, unrelated EV ATV voice assistant project at
`D:\EV ATV VOICE PROJECT\ev-atv-voice-assistant` (friend-collaboration,
Vosk STT + keyword-rule intents, built 2026-07-30 to 2026-08-02, no
resort/autonomy vision). Same domain, different product — don't pull
code, conventions, or assumptions from it without deliberately deciding
to.

## Kickoff date

2026-08-05.
