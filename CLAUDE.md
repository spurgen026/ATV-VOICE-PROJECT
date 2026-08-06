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

## V1 implementation (decided 2026-08-05, superseded 2026-08-06)

> Kept for history. V1's keyword-rule intent matching (`intents.py`) was
> deleted 2026-08-06 and replaced by the V2 RAG pipeline below — see
> "V2 implementation." Wake word, STT, TTS, and the dummy vehicle data
> all carried forward unchanged; only the query-answering mechanism
> changed.

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
- **Version control:** git repo initialized 2026-08-05, initial commit
  made. Git identity (`spurgensam14@gmail.com` / "Spurgen") set
  **local to this repo only** (not `--global`), at the user's explicit
  request.

## Open questions (not yet decided)

- Target production hardware: mini PC vs. Raspberry Pi vs. NVIDIA
  Jetson. (The V2 plan below assumes Raspberry Pi 3, but that's the
  planning doc's default, not a confirmed decision — the V2 build so
  far still runs on the laptop, same as v1.)
- How the "accessible anywhere in the resort" requirement gets met
  eventually — single onboard unit vs. distributed mics/speakers — not
  addressed by the laptop prototype.
- A real, resort/vehicle-branded wake word.

## Version 2 Plan (2026-08-06)

User dropped in a planning PDF — *Raspberry Pi 3 Voice Document
Assistant (Cloud AI Only)* — adopted as the direction for v2. Built the
same day; see "V2 implementation" below for what actually shipped vs.
this plan. Original PDF and full write-up:
`D:\obsidian\crytonite\Raw\2026-08-06-raspberry-pi-voice-document-assistant.pdf`
and the "Version 2 Plan" section of
`D:\obsidian\crytonite\wiki\projects\resort-atv-voice-assistant.md`.

### What the plan describes

A voice-enabled **document Q&A** assistant, backend-free:

- Documents (PDF/DOCX/TXT) live locally in `documents/` on the Pi, get
  chunked, embedded (cloud embeddings), and indexed into a **local
  FAISS vector store**.
- Query flow: user speaks → mic → STT → embed the question → search
  local FAISS index → top 5 matching chunks → prompt → **cloud LLM**
  (OpenAI, Gemini, or Claude) → answer → TTS → speaker.
- **No local LLM, no custom backend server** (no FastAPI/Node) — cloud
  AI handles embeddings and reasoning only; everything else is local.
- Answers are strictly grounded in the uploaded documents: the system
  prompt instructs the model to use *only* the supplied context and to
  reply "I couldn't find that information in the provided documents."
  rather than guessing when the answer isn't present.
- Target hardware per the doc: Raspberry Pi 3, Raspberry Pi OS, USB
  mic, USB speaker/headset, internet connection.
- Python stack per the doc: `openai`, `faiss-cpu`, `numpy`, `pypdf`,
  `python-docx`, `sounddevice`, `soundfile`, `python-dotenv`.
- The doc's own listed future enhancements (beyond even this v2 scope):
  wake word detection, conversation history, multiple document
  collections, multi-language support, automatic re-indexing, streaming
  speech, barge-in, speaker identification, OLED status display.

### Reconciliation decisions (decided 2026-08-06)

These were open questions until the user made explicit calls on each:

- **Replacement, not additive.** V2's document Q&A *replaces* v1's
  keyword-rule intent matching entirely — vehicle stats now answer
  through the same RAG pipeline as any other document, not a separate
  mechanism. `intents.py` was deleted.
- **Hybrid pipeline, not fully cloud.** V1's local wake-word/STT/TTS
  stack (openWakeWord, faster-whisper, Piper) was kept as-is; only
  document search + LLM reasoning go to the cloud. This preserves the
  original reasoning for going local in the first place (Gemini Live's
  free tier can't sustain continuous listening) while still getting
  cloud-quality reasoning for the actual Q&A.
- **Provider: Gemini**, not OpenAI — the user has a Gemini
  subscription and no OpenAI one. Both embeddings
  (`gemini-embedding-001`) and chat (`gemini-flash-latest`) go through
  Gemini.
- **Hardware: still the laptop.** Pi 3 remains the doc's target but
  wasn't independently confirmed as a decision — see "V2
  implementation" for what's actually running where.

## V2 implementation (built 2026-08-06)

Live-tested end to end on the dev laptop with real headphones: wake
word, STT, RAG search, Gemini answer, and TTS all confirmed working,
including correct grounded refusal ("I couldn't find that information
in the provided documents.") on out-of-scope questions.

- **New modules:** `gemini_client.py` (shared `genai.Client()`),
  `documents.py` (loads/chunks `.txt`/`.pdf`/`.docx` from
  `documents/`), `index_documents.py` (embeds chunks, builds/saves the
  FAISS index — rerun whenever documents change, same as the doc's
  Step 2), `rag.py` (embeds the question, searches the index, calls
  Gemini with a grounded system prompt, returns the answer).
- **Vehicle stats as a document:** `data_store.py` gained
  `write_vehicle_status_document()`, which renders
  `vehicle_state.json` as `documents/vehicle_status.txt` so it's
  indexed like any other document. Currently the *only* document in
  the corpus — no real resort FAQs/policies exist yet to add.
- **Index artifacts:** `index/vectors.faiss` + `index/metadata.json`
  (same filenames as the planning doc), gitignored — rebuilt from
  source documents, not meant to be committed.
- **New dependencies:** `google-genai`, `faiss-cpu`, `pypdf`,
  `python-docx`, `python-dotenv` — added to `requirements.txt` and
  installed.
- **Secrets:** Gemini API key lives in `.env` (gitignored;
  `.env.example` committed as the template). The user was first asked
  to set the key in `.env` themselves via their editor to keep it out
  of the chat; that didn't happen, and the user pasted the key
  directly into the conversation instead, so it was written to `.env`
  from there. Flagged to them that the key is now in the conversation
  transcript, not just the file, and that they may want to rotate it.
- **Config additions** in `config.py`: `GEMINI_EMBEDDING_MODEL =
  "gemini-embedding-001"`, `GEMINI_CHAT_MODEL = "gemini-flash-latest"`
  (a stable alias, not a pinned version — `gemini-2.5-flash` was
  already retired for new API keys by the time this was built, so
  pinning felt fragile), `CHUNK_SIZE_CHARS = 800`,
  `CHUNK_OVERLAP_CHARS = 100`, `TOP_K_CHUNKS = 5` (matches the doc).
- **Deviations from the doc, deliberate:** kept the wake word (the doc
  lists wake-word detection as a *future* enhancement, implying no
  wake word in its own core design); file layout follows v1's existing
  module boundaries (`main.py`/`rag.py`/`stt.py`/`tts.py`/
  `wake_word.py`) rather than the doc's suggested
  `app.py`/`search.py`/`speech.py`.
- **Known gaps vs. the doc:** not running on real Raspberry Pi 3
  hardware (still the laptop, same as v1); only one document exists
  (the auto-generated vehicle-status doc) — multi-document handling is
  built and untested-at-scale, just nothing else to index yet.
  Everything the doc itself lists under "Future Enhancements"
  (conversation history, multi-doc collections, multi-language,
  streaming, barge-in, speaker ID, OLED display) is intentionally not
  built.

## V2 hardening (2026-08-06)

Three fixes made after the initial V2 build, each live-tested or
verified by directly simulating the failure condition (not just
theorized):

- **Gemini failure handling.** `answer_query()` in `rag.py` now wraps
  both cloud calls in `try/except (errors.APIError, httpx.HTTPError)`
  — confirmed by pointing the client at an unreachable host and
  checking what actually got raised (`httpx.ConnectError`) rather than
  guessing. On failure it logs the real error and returns a spoken
  fallback instead of crashing the main loop.
- **Wake-word acknowledgment chime.** `tts.play_ack_chime()` plays a
  synthesized tone (`CHIME_FREQUENCY_HZ`, `CHIME_DURATION_MS`,
  `CHIME_VOLUME` in `config.py`) immediately on wake-word detection,
  before recording/the cloud round-trip — gives audio feedback during
  the wait V2 introduced. First version (880Hz/120ms/0.3 volume) was
  live-tested and reported inaudible; bumped to 1000Hz/250ms/0.7 and
  retested.
- **Offline fallback for vehicle stats.** `data_store.py` gained
  `try_local_answer()` — the old V1 keyword-matching logic, revived
  but demoted to a fallback only used when the Gemini call fails.
  `rag.py`'s except block tries it before giving up; vehicle-stat
  questions (battery/motor temp/speed/tire pressure) now still answer
  correctly with zero internet, while unrelated questions still
  honestly report they're unavailable. Directly addresses the gap
  identified when comparing V1 vs V2: previously *any* network drop
  meant zero answers, even for data that never needed the cloud.

## V2 companion polish (2026-08-06)

Follow-up work making the assistant feel less like a demo, prompted by
"is there any way to make it feel more real":

- **Personality.** `RAG_SYSTEM_PROMPT` in `config.py` now frames it as
  "the voice of a resort ATV — a friendly, easygoing companion," warm
  and conversational, while keeping the grounding rules and the exact
  required refusal string untouched. Verified live: e.g. "You got it!
  Your battery is at 78 percent and your tire pressure is sitting at
  32 psi." Also explicitly told not to add closing filler ("ready
  whenever you are," "let me know if you need anything else") —
  the first version added these consistently and the user asked for
  them removed; second version confirmed clean.
- **Follow-up window.** `main.py`'s loop now listens again after every
  answer without requiring the wake word — `stt.record_query()` gained
  a `speech_timeout_ms` parameter so the follow-up listen can use a
  shorter, silent-on-timeout window (`FOLLOWUP_SPEECH_TIMEOUT_MS`,
  4000ms) instead of the post-wake-word one
  (`QUERY_SPEECH_TIMEOUT_MS`, 3000ms, which still speaks "Sorry, I
  didn't catch that" on the *first* miss only). Live-tested working:
  multi-turn conversation without repeating "Hey Jarvis" between
  questions.
- **Voice changed twice.** First `en_US-lessac-medium` →
  `en_US-ryan-high` (user's pick from the real Piper voice catalog,
  fetched live rather than guessed from memory). User then said the
  American accent didn't fit — switched again to `en_GB-cori-high`
  (British), also picked from the real catalog.
- **TTS chunk-boundary bug fix.** User reported words "breaking
  towards the end" of responses. Root cause: Piper's `synthesize()`
  yields one audio chunk per sentence, and `tts.speak()` was calling
  `sd.play()` separately per chunk — each call reopens the audio
  stream, causing glitches at sentence boundaries, worst on short
  trailing sentences (which the new personality prompt produces more
  of). Fixed by concatenating all chunks into one buffer and playing
  once. Confirmed the failure mode first (multi-sentence text really
  does produce multiple same-sample-rate chunks) before fixing, rather
  than guessing.

## V1 vs V2 at a glance

Wake word, STT, TTS, and silence-based query capture are unchanged
between v1 and v2 — only *how the question gets answered* changed.

| | V1 | V2 |
|---|---|---|
| Answering mechanism | Keyword-rule matching (`"battery" in text`, etc.) in `intents.py` | RAG: embed question → search local FAISS index → retrieve matching text → Gemini generates a grounded answer |
| Data source | `data/vehicle_state.json` read directly | Same data, rendered into `documents/vehicle_status.txt` and indexed like any document |
| What you can ask | Exactly 4 fixed topics, rigid phrasing | Same 4 topics today (only doc that exists), but open-ended — any document dropped into `documents/` becomes answerable, phrasing is flexible |
| Network | Fully offline | Needs internet — embeddings + answer generation are cloud calls to Gemini |
| Unknown questions | Generic fallback message | Explicit grounded refusal: "I couldn't find that information in the provided documents." |

`intents.py` is deleted, not kept as a fallback.

## Tech stack (cumulative)

**Local/offline, unchanged since v1:**

- Python 3.9
- `openwakeword` 0.6.0 — wake word detection
- `faster-whisper` 1.2.1 — speech-to-text
- `piper-tts` 1.6.0 — text-to-speech
- `sounddevice` 0.5.5 — mic capture + audio playback
- `numpy` — audio array handling

**Cloud, added in v2:**

- `google-genai` 1.47.0 — Gemini SDK
  - `gemini-embedding-001` — embeddings
  - `gemini-flash-latest` — answer generation
- `faiss-cpu` 1.13.0 — local vector search index
- `pypdf` 6.14.2 / `python-docx` 1.2.0 — PDF/DOCX text extraction (unused so far, only a `.txt` document exists)
- `python-dotenv` 1.2.1 — loads the Gemini API key from `.env`

**Infra:** git, local-only identity, no remote configured.

## Unrelated sibling project — do not confuse

There's a separate, unrelated EV ATV voice assistant project at
`D:\EV ATV VOICE PROJECT\ev-atv-voice-assistant` (friend-collaboration,
Vosk STT + keyword-rule intents, built 2026-07-30 to 2026-08-02, no
resort/autonomy vision). Same domain, different product — don't pull
code, conventions, or assumptions from it without deliberately deciding
to.

## Kickoff date

2026-08-05.
