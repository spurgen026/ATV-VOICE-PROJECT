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

## V2 companion polish, round 2 (2026-08-06)

User asked for ideas on making it feel more like a companion, reviewed
them, and picked four to build (explicitly deferred: giving it a name,
proactive/unprompted check-ins — bigger scope, not done):

- **Small talk.** New `small_talk.py` — `try_small_talk_answer()`
  keyword-matches greetings, thanks, and identity/capability questions
  ("who are you," "what can you do") and answers them directly,
  checked at the top of `rag.answer_query()` *before* touching Gemini
  at all. Without this, "thanks" was falling through to the document
  search and getting the honest-but-jarring "I couldn't find that
  information" refusal. Verified live: small talk gets answered with
  no Gemini API call made (confirmed by the absence of the SDK's
  response warning in the log for those turns).
- **Short-term conversational memory.** `answer_query()` gained a
  `history` parameter (list of prior (question, answer) pairs,
  capped at `MAX_HISTORY_TURNS = 4`), formatted into the generation
  prompt so follow-ups like "what about the tire pressure too?"
  resolve correctly. Reset every wake-word cycle in `main.py` — it's
  memory for one conversation, not persistent. Note: only the
  *generation* step sees history, not retrieval — with a single
  document in the corpus this doesn't matter yet, but a genuinely
  pronoun-only follow-up ("what about that?") could retrieve the
  wrong chunk once there are multiple documents. Known limitation, not
  yet a problem.
- **Spoken startup greeting.** `STARTUP_GREETING` in `config.py`,
  spoken once right after models finish loading, before "Say the wake
  word." A companion should announce itself, not sit silently in a
  terminal.
- **Sleep chime.** `tts.play_sleep_chime()` — a descending two-tone
  (700Hz → 450Hz, mirroring the single-tone ack chime), played when
  the follow-up window times out silently, so it's audibly clear the
  assistant stopped listening even though nothing gets said. Tone
  generation in `tts.py` was refactored into a shared `_generate_tone()`
  helper used by both chimes.
- **Bug found via live use, not testing: compound-question fallback
  gap.** A real Gemini 503 (server overload) happened live during
  testing of the above. The offline fallback correctly caught it and
  answered — but the question asked for *both* battery and speed, and
  `try_local_answer()` only returned the first matching stat, silently
  dropping the rest. Fixed: it now collects every matching fact and
  joins them into one natural sentence instead of returning on the
  first match. Verified against the exact question that surfaced the
  bug, plus single-fact, all-four-facts, and no-match cases.

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

## Version 3 Plan (2026-08-07)

**This is a pivot, not an increment.** V2 as actually built (above) is
the Gemini cloud RAG system. A separate, more elaborate local
architecture — Qwen router + GBNF grammar + CAN bus microservice — was
designed in a *conversation* on 2026-08-07 but never implemented
against this codebase. V3 is that architecture, now adopted as the real
plan: it **replaces V2's cloud dependency for telemetry answers**, it
doesn't extend the Gemini pipeline. Full write-up, including the
target architecture diagram, phased execution plan, and open items, is
in `D:\obsidian\crytonite\wiki\projects\resort-atv-voice-assistant.md`
under "Version 3 Plan" — that's the canonical copy; summary below.

### Requirements (stated 2026-08-07)

- Runs on **Raspberry Pi 5 (8GB)** and **NVIDIA Jetson** (tier — Nano
  vs. Orin — not yet decided) as the real production hardware, not the
  laptop.
- **Everything local.** No cloud calls — explicit reversal of V2's
  Gemini dependency for telemetry.
- **Trilingual: Tamil, English, Hindi**, auto-detected per utterance,
  no manual mode switch.
- **Real-time** response.
- Model must be **light** and **reliable** (no hallucinated telemetry
  numbers).
- Must interpret **all vehicle telemetry details**, not just the
  current 4 dummy stats.

### Target architecture

Wake word (openWakeWord) → STT (`faster-whisper`, multilingual,
`language=None` auto-detect) → local LLM router (Qwen2.5, GBNF-grammar-
constrained via `llama.cpp`/`llama-cpp-python`, output restricted to
`{"action":"get_telemetry","target":...}` or `{"action":"chat",...}`)
→ CAN telemetry microservice (background thread decodes CAN frames
into an in-memory cache; lookups never block on bus I/O) → TTS (Piper,
kept — needs Tamil/Hindi voice packs; Kokoro-82M/MeloTTS as fallback
candidates).

The grammar-constrained router removes the hallucinated-number failure
mode entirely: the LLM only ever picks *which* value to fetch, a
deterministic Python lookup supplies the actual number — it never
generates the number itself.

### What happens to V2's Gemini/RAG code

**Parked, not deleted.** `rag.py`, `documents.py`, `gemini_client.py`,
`index_documents.py`, and the FAISS index still have real value for
open-ended resort-knowledge Q&A (hours, policies, sightseeing) once
actual resort documents exist — a different job than telemetry
lookups. Reviving it locally would need a local embedding model to swap
out Gemini's — not scoped into V3, tracked as an open item.

### Blocking config fact found while planning

`WHISPER_MODEL_SIZE = "small.en"` in `config.py` is an **English-only**
Whisper variant — it cannot transcribe Tamil or Hindi at all. Must
switch to a non-`.en` multilingual size before any Tamil/Hindi STT
testing can begin. This is a Phase 2 blocker, not optional polish.

### Open items (block "production ready," not the plan)

1. ~~Full telemetry parameter list~~ — **decided 2026-08-07:** build V3
   against the current 4-field dummy schema first (battery, motor temp,
   speed, tire pressure); expand to the full power/drivetrain/motion/
   safety draft later without redoing the router/grammar architecture.
2. Jetson tier (Nano vs. Orin) — **deferred 2026-08-07** at the user's
   call: get the whole pipeline working on the laptop first (Phases
   1-7), decide Pi 5 vs. Jetson (and which Jetson tier) afterward.
   `llama.cpp` CPU baseline remains the default until then.
3. The 2026-08-07 conversation's draft `telemetry_router.gbnf` and
   `can_telemetry_service.py` (currently only in a scratch directory,
   not in this repo) use placeholder field names that don't match the
   real dummy schema — conceptual references, not drop-in code.
4. Tamil/Hindi Piper voice availability — unresearched.
5. Whether/when to revive the Gemini RAG path locally for non-telemetry
   resort-knowledge Q&A.
6. ~~Python 3.9 EOL~~ — **resolved 2026-08-07**, see "V3 environment
   setup" below.

### V3 environment setup (2026-08-07)

Phase 1 of the execution plan below, done:

- **Interpreter upgraded 3.9 → 3.12.** Python 3.9 was EOL and, more
  concretely, its Windows/pip combo couldn't even attempt building
  `llama-cpp-python` (no prebuilt wheel on PyPI for any Python version
  — it always builds from source). Installed Python 3.12.10 via
  `winget install Python.Python.3.12` (landed at
  `C:\Users\Spurgen\AppData\Local\Programs\Python\Python312`, a
  per-user install, ~155MB — not a cache, so it doesn't conflict with
  the [[feedback_c_drive_low_storage]] convention of keeping pip/HF
  caches off C:). New venv built at `.venv` (Python 3.12.10); the old
  3.9 venv was kept as `.venv39-old` rather than deleted, as a
  rollback. All V1/V2 dependencies reinstalled clean — every native
  package (`ctranslate2`, `onnxruntime`, `scipy`, `pydantic-core`,
  `lxml`) had a prebuilt `cp312` wheel, nothing built from source.
  V1/V2 re-verified via import smoke test + `pip install -e .` on the
  new interpreter (not yet re-verified with real audio hardware — that
  still needs a live test before V3 work is considered fully safe).
- **`llama-cpp-python` install blocker found and fixed.** First
  attempt (on the old 3.9 venv) failed outright: pip fell back to the
  71.6MB sdist (no PyPI wheel exists for *any* Python version on
  Windows) and the source unpack itself crashed — the vendored
  `llama.cpp` tree has paths deep enough to exceed Windows' default
  260-char `MAX_PATH`. Fixed by enabling Windows long paths
  (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled
  = 1`, needs admin — user did this via an elevated PowerShell, since
  this machine is Windows 11 **Home** and has no `gpedit.msc`).
  Retried on the new 3.12 venv: this time it built from source
  successfully (`llama_cpp_python-0.3.34`) — turned out Visual Studio
  Build Tools were already installed on this machine (found via
  `vswhere`), so CMake could invoke MSBuild directly without needing
  `cl.exe` on `PATH`. If this is ever repeated on a clean machine,
  Visual Studio Build Tools (C++ workload) would need installing first.
- **`python-can` installed clean** (`4.6.1`, pure-Python wheel, no
  build issues).
- **`requirements.txt` updated** with `llama-cpp-python==0.3.34` and
  `python-can==4.6.1` under a new "V3: local Qwen router + CAN
  telemetry" section.
- **Still open from Phase 1:** picking the actual Qwen2.5 GGUF quant
  source (1.5B vs. 3B) — not yet done, needs benchmarking (Phase 3),
  not just an install.

### Phase 2 progress (2026-08-07) — model swapped, live-tested, see below

- `WHISPER_MODEL_SIZE` swapped `"small.en"` → (eventually) `"medium"`;
  `stt.transcribe()` now calls `model.transcribe(audio, language=None)`
  for per-utterance auto-detect instead of the old hardcoded
  `language="en"`. See "Phase 2+3 live multilingual pipeline test"
  below for the live-testing history that drove `small` → `medium`.

### Phase 3 progress (2026-08-07) — router smoke test, English-only

- **Model:** Qwen2.5-1.5B-Instruct, `Qwen/Qwen2.5-1.5B-Instruct-GGUF`,
  `q4_k_m` quant (~1GB, downloaded to
  `D:\DevTools\huggingface-cache`). Picked the 1.5B tier over 3B first
  since the router's job is grammar-constrained classification, not
  open reasoning — benchmark results below confirmed this was
  sufficient, no need to escalate.
- **New files:** `grammars/telemetry_router.gbnf` (GBNF grammar
  restricting output to exactly `{"action": "get_telemetry", "target":
  "<one of the 4 real dummy-schema field names>"}` or `{"action":
  "chat"}` — supersedes the placeholder-schema draft referenced in
  open item 3, now built against the real field names in
  `data/vehicle_state.json`); `src/resort_atv_voice/router.py`
  (`load_router_model()`, `load_grammar()`, `route()` — loads the GGUF
  via `Llama.from_pretrained`, applies the grammar via
  `llama_cpp.LlamaGrammar`). `config.py` gained `QWEN_MODEL_REPO`,
  `QWEN_MODEL_FILENAME`, `ROUTER_GRAMMAR_PATH`, `ROUTER_CONTEXT_SIZE`,
  `ROUTER_SYSTEM_PROMPT`.
- **Benchmark results** (18 hand-written English test queries — 3 per
  telemetry field × 4 fields, plus 6 chat/off-topic queries; CPU only,
  laptop, `benchmark_router.py`, not committed): **18/18 routing
  accuracy**. Latency: avg 0.85s, min 0.39s (chat, short output), max
  1.78s (first telemetry call, includes some warmup). Comfortably
  real-time for a voice interaction.

### Phase 2+3 live multilingual pipeline test (2026-08-07)

Combined test (`test_stt_router_pipeline.py`, not committed): user
spoke real English/Hindi/Tamil phrases through the actual STT → router
chain. Two rounds, findings compounding on each other:

**Round 1, `WHISPER_MODEL_SIZE = "small"`:** STT consistently mangled
the English loanwords "battery"/"tire" embedded in Hindi/Tamil speech
into garbled non-words (e.g. "battery" → `பாற்றி`, not a real word).
One case was a genuine safety bug, not just a missed answer: a garbled
Hindi battery question got routed to `tire_pressure_psi` — a
**confidently wrong telemetry field**, worse than "didn't understand."

**Fix attempted: escalate to `WHISPER_MODEL_SIZE = "medium"`.**
Re-tested the same phrases. Real improvement: Hindi "motor kitni garam
hai" now transcribes and routes correctly (failed outright before);
the Tamil tire-pressure phrase now transcribes *correctly* as
`தயர் பிரச்சர்...`, fixing the STT half of that case. Tamil "battery"
still mis-transcribes to `பாட்டி` (which actually means "grandmother").
Importantly, no confidently-wrong answers this round — remaining
failures fell back to safe `chat` rather than a wrong number.

**New problem surfaced: correctly-transcribed Tamil still misrouted.**
The Tamil tire-pressure phrase, now correctly transcribed, still got
routed to `chat` instead of `tire_pressure_psi` — a router-side gap,
not an STT gap: Qwen wasn't recognizing Tamil-script phrasing of an
English loanword it should know.

**Fix attempted: few-shot Tamil/Hindi examples in `ROUTER_SYSTEM_PROMPT`
(`ROUTER_FEWSHOT_EXAMPLES` in `config.py`, deliberately different
phrasings from anything tested, to measure real generalization).**
Tested against a held-out set built from the actual transcripts
captured in both live rounds above (`benchmark_router_v2.py`, not
committed) — 9 real multilingual transcripts plus the original 18
English cases as a regression check. Result: **multilingual went
3/9 → 6/9 correct, but 3 NEW failures appeared that are worse in kind
than what was there before** — cases that used to safely fall back to
`chat` now confidently guess a *wrong* telemetry field instead. Plus a
genuine English regression: "What's the weather like today?" started
routing to `battery_percent` instead of `chat` (17/18, was 18/18).
Net effect: better raw accuracy, worse safety profile — trading "I
don't know" for "confidently wrong," which is the opposite of what the
whole grammar-constrained-router design is for.

**Fix attempted: explicit "if unsure, say chat, don't guess" instruction
added to `ROUTER_SYSTEM_PROMPT`.** Fixed the English regression cleanly
(back to 18/18) but did **not** fix the Tamil wrong-field-guessing
problem — multilingual held-out actually dropped slightly to 5/9, same
3 dangerous failures still present (still guessing a specific wrong
field, never falling back to chat for these three). Prompt wording
alone doesn't seem to unlock "admit uncertainty" behavior in Tamil at
this model size.

**Fix attempted: escalate router model 1.5B → Qwen2.5-3B-Instruct**
(one-off comparison script, not committed, config.py still points at
1.5B). Result: **no improvement** — identical 3 failures on the
multilingual held-out set (6/9, same pattern as few-shot-only), *plus
two new English regressions* (became too conservative, missed two
legitimate telemetry requests: "is the engine overheating," "are my
tires properly inflated" — both fell back to chat when they shouldn't
have). Latency also roughly doubled (1.6-1.9s vs 0.85s). Scaling the
router model up made things strictly worse here, not better.

**Current committed state:** `WHISPER_MODEL_SIZE = "medium"`, router
stays on Qwen2.5-**1.5B**, with both the few-shot examples and the
"don't guess when unsure" instruction kept — this combination scored
best overall (English 18/18, multilingual 5/9) even though it doesn't
fully solve the multilingual case.

**Open, unresolved:** 3 specific real-world transcripts consistently
fail across every combination tried (small/medium Whisper × no-few-shot/
few-shot/few-shot+confidence prompts × 1.5B/3B router) — all involve a
mistranscribed Tamil loanword where the router confidently commits to
a wrong field instead of admitting uncertainty. Two rounds of prompt
engineering and a model-size escalation did not fix this. This looks
like a genuine capability limit of text-only grammar-constrained
routing once the input word itself is unrecognizable garbage, not
something more prompting alone will solve — likely needs either
further STT improvement, or a different mitigation (e.g. fuzzy/phonetic
matching of key vehicle terms before routing). Tracked here rather than
silently accepted; not resolved as of 2026-08-07.

### Phase 4 progress (2026-08-07) — dummy CAN service, smoke-tested

- **New files:** `src/resort_atv_voice/can_telemetry.py`
  (`CAN_ID_TO_FIELD`/`FIELD_TO_CAN_ID` mapping the 4 real dummy-schema
  fields to arbitrary CAN IDs `0x100`-`0x103`; `encode_frame()`/
  `decode_frame()`; `TelemetryCache` - a lock-protected latest-value
  dict; `start_listener()` - background thread that decodes bus frames
  into the cache, so lookups never block on bus I/O, per the target
  architecture above). `scripts/fake_ecu_publisher.py` - stands in for
  real ATV hardware (none exists yet), publishes
  `data/vehicle_state.json` on the virtual CAN channel every second,
  mimicking a real ECU's periodic broadcast.
- **Important limitation found by reading `python-can`'s own source**
  (`can/interfaces/virtual.py`), not assumed: the `"virtual"` interface
  only shares messages between `Bus` instances **in the same process**
  - confirmed via the library's own docstring. This means the fake-ECU
  publisher and a listener running as two separate processes will
  *not* see each other's messages; the smoke test below runs both in
  one process for that reason. This is a dev/testing-only limitation -
  the real `"socketcan"` interface on actual hardware (Phase 8+) is a
  real shared bus with no such restriction, so this isn't a design
  problem to carry forward, just a fact about how the dummy stand-in
  has to be exercised.
- **Smoke test** (`test_can_telemetry.py`, not committed): publisher
  writes all 4 fields onto the virtual bus, background listener thread
  decodes them into `TelemetryCache`, lookups read them back correctly.
  All 4 fields matched `vehicle_state.json` exactly; pre-publish
  lookups returned `None` immediately rather than hanging; post-publish
  lookups completed in ~0.00001s (never touch the bus). Fully automated,
  no live audio/human needed for this phase.

### Phased execution plan

0. This planning doc — done 2026-08-07.
1. Environment — **done 2026-08-07**, see "V3 environment setup"
   above.
2. Multilingual STT validation — **live-tested 2026-08-07**, escalated
   `small` → `medium` after real loanword-transcription failures; not
   fully solved, see "Phase 2+3 live multilingual pipeline test" above.
3. Router smoke test — **live-tested 2026-08-07**: English is solid
   (18/18, sub-2s), multilingual is real but incomplete (5/9 on a
   held-out set of actual captured transcripts, 3 recurring failures
   unresolved after few-shot examples, a confidence instruction, and a
   1.5B→3B escalation). See "Phase 2+3 live multilingual pipeline
   test" above for the full investigation before treating this as
   closed.
4. Dummy CAN service — **done 2026-08-07**, smoke-tested. See "Phase 4
   progress" above, including the found-not-assumed limitation that
   `python-can`'s virtual interface is process-local.
5. TTS language coverage — **done 2026-08-07**, live-tested. Piper's
   catalog has no Tamil at all and neither named fallback (Kokoro-82M,
   MeloTTS) covers Tamil or Hindi either - adopted MMS-TTS instead. See
   "Phase 5 progress" above.
6. Integration into `main.py` — **done 2026-08-07**, and it *replaces*
   the live V2 Gemini path rather than running alongside it (a
   deliberate deviation from this line's original wording, at the
   user's explicit direction to make the running app actually local,
   not just prove the pieces work in isolation). `rag.py` etc. remain
   in the repo, parked, just no longer imported by `main.py`. See
   "Phase 6 progress" above, including one real unresolved issue
   (local chat can invent fake resort amenities).
7. End-to-end laptop testing: all three languages × full schema, live
   speech, same rigor as V1/V2. Partially covered already (Phase 5+6
   live audio tests), but not yet a dedicated full pass.
8. Pi 5 port: compile `llama.cpp` natively (NEON), re-benchmark real
   latency on-device.
9. Jetson port once tier is confirmed: `llama.cpp` baseline either way;
   GPU offload or vLLM/TensorRT-LLM only if Orin.
10. Parked/future: real `python-can` wiring to the actual ATV, custom
    resort-branded wake word, local-embeddings RAG revival. Autonomy
    features stay out of scope (see product vision above).

### Phase 5 progress (2026-08-07) — Tamil/Hindi TTS, live-tested

- **Piper has zero Tamil voices** - confirmed by calling
  `huggingface_hub.list_repo_files('rhasspy/piper-voices')` directly
  rather than assuming: its 50 language directories include `hi` (3
  voices - pratham, priyamvada, rohan, all "medium" tier) but no `ta`
  at all. **Kokoro-82M and MeloTTS also don't cover Tamil or Hindi** -
  checked each model's own HF tags directly (Kokoro-82M is tagged `en`
  only; MeloTTS ships separate per-language repos, none for
  Tamil/Hindi). So neither of the plan's two named fallback candidates
  actually works for this requirement.
- **Adopted: Meta's MMS-TTS** (`facebook/mms-tts-tam`,
  `facebook/mms-tts-hin`, VITS via `transformers`) - genuinely covers
  both languages. Verified real, not assumed, via `HfApi.model_info()`
  before committing to it. New dependencies: `torch` (CPU-only wheel
  via `download.pytorch.org/whl/cpu`, confirmed `2.13.0+cpu` not the
  multi-GB CUDA build) and `transformers`. Benchmarked at ~0.25x
  realtime factor on CPU (synthesis takes ~25% of the audio's own
  duration) - comfortably fast enough for a voice assistant.
- **Live-tested with the user twice**: first the standalone Tamil/Hindi
  clips played directly through their speakers (confirmed "good enough
  to use"), then the full pipeline (text query -> local answer in that
  language -> spoken with the matching voice) for English/Hindi/Tamil
  battery questions plus an English motor-temp question - all four
  played correctly and were confirmed acceptable.
- **`tts.py`** gained `VoiceBundle` (Piper for English + two MMS-TTS
  models for Hindi/Tamil) and `load_voices()`/`speak(voices, text,
  language)`, replacing the old English-only `load_voice()`/
  `speak(voice, text)`.

### Phase 6 progress (2026-08-07) — main.py now fully local, Gemini removed from the live path

At the user's explicit direction ("ensure everything in the V3 plan
operates locally... the entire goal is to bring everything to operate
locally in an effective way") - until this point `main.py` still
imported `rag.py` and called Gemini for every single query, V3's local
pieces (STT, router, CAN service, TTS) existed but were never wired
together into the actual app.

- **New: `local_qa.py`** - `answer_query()`, the direct local
  replacement for `rag.py`'s Gemini-backed one. Order: small talk (a
  known English-only gap, see below) -> grammar-constrained router ->
  either a CAN cache lookup (`get_telemetry`) or a free-form local chat
  reply (`chat`) via the same Qwen model, no grammar constraint.
- **`router.py`** gained `generate_chat_reply()` - answers anything
  classified as chat, in the rider's own detected language
  (`LOCAL_CHAT_SYSTEM_PROMPT` takes a `{language_name}` slot). Without
  this, the app would auto-detect Hindi/Tamil input correctly and then
  always answer back in English regardless - detecting the input
  language is not the same as generating output in it.
- **`stt.transcribe()` signature changed**: now returns `(text,
  language_code)` instead of just `text` - callers need the language to
  pick response templates and the TTS voice, not just to log it.
- **`can_telemetry.describe()` and `TELEMETRY_UNAVAILABLE_RESPONSE`
  became per-language dicts** (`en`/`hi`/`ta`) in `config.py`, written
  directly rather than machine-translated at runtime, so a battery
  question asked in Tamil gets answered in Tamil, not English.
- **`main.py` rewritten**: no longer imports `rag.py` or anything
  Gemini-related. Loads the router model+grammar and voice bundle at
  startup, starts `start_fake_ecu()` + `start_listener()` (standing in
  for real ATV hardware), and threads the detected language through
  the whole turn (transcribe -> answer -> speak).
- **Verified structurally, not just functionally**: `main.py`'s import
  graph no longer includes `gemini_client.py` at all - there is no code
  path left in the live app that can reach the network. `rag.py`,
  `gemini_client.py`, `documents.py`, `index_documents.py` are
  untouched, still there, still "parked, not deleted" per the earlier
  decision, just genuinely unused now instead of nominally-parked-but-
  still-imported.
- **Fully automated integration test** (`test_local_qa.py`, not
  committed): small talk, all 4 telemetry fields, and 2 free-form chat
  queries, all through the exact `answer_query()` `main.py` calls.
  Small talk and telemetry all correct. **One real quality issue found
  and not yet fixed**: asked to recommend a resort activity, the local
  chat model invented specific, non-existent amenities ("a scenic boat
  ride around the lake," "a hot spring bath") stated as fact, despite
  `LOCAL_CHAT_SYSTEM_PROMPT` explicitly saying not to guess. Same
  underlying pattern as the router's uncertainty problem from Phase 3 -
  this model size doesn't reliably follow "admit you don't know"
  instructions. For a resort-facing assistant, confidently inventing
  amenities is a real trust problem, tracked here as unresolved, not
  papered over.
- **Verified with network access physically forced off, not just
  structurally** - the user asked directly "is everything running
  locally now?" and this deserved an actual test, not an assumption.
  Running with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` initially
  **failed**: `llama_cpp.Llama.from_pretrained()` (used for the router
  model) always calls the HF Hub API to list/resolve the repo's files,
  even when the exact file is already fully cached locally - neither
  switching from a glob filename to an exact one, nor the offline env
  vars, stopped it. Fixed by not using that convenience method at all:
  `router.load_router_model()` now calls `huggingface_hub.
  hf_hub_download()` directly to resolve the local cache path (which
  *does* work fully offline once cached) and passes that path straight
  to `Llama(model_path=...)`. Re-ran the same offline test after the
  fix: wake word, Whisper, both TTS engines, and the router all loaded
  and `answer_query()` answered correctly with network access
  genuinely off. This is the strongest evidence so far that the live
  app has no hidden network dependency.
- **Known, tracked gap**: `small_talk.py`'s keyword matching is
  English-only (checks literal English words like "hello" against the
  transcript). A Hindi/Tamil greeting doesn't match it and falls
  through to the router/local-chat path instead - which still answers
  correctly (confirmed - Tamil/Hindi battery questions typed directly,
  i.e. not run through STT, routed and answered correctly in Phase
  5+6 testing), just without the canned instant response. Not fixed
  this session.

### Phase 5 hardening (2026-08-07) — MMS-TTS was speaking the wrong number

User reported live: the Hindi/Tamil voices "don't say the reading, they
just say the sentence." Investigated rather than guessed at a fix:

- **Root cause, confirmed via each tokenizer's `get_vocab()` directly**:
  `facebook/mms-tts-hin`/`-tam` are character-level VITS models with an
  *incomplete* digit vocabulary per language - Hindi's is missing
  `5,6,7,9`, Tamil's is missing `8`. Feeding raw digits like "78"
  silently drops whichever digit isn't in that language's vocab, with
  no error - it doesn't skip the number, it **confidently speaks the
  wrong one** (Hindi said "8" for 78; Tamil said "7"). `num2words` (the
  standard fix) doesn't support Hindi or Tamil at all - checked its
  `CONVERTER_CLASSES` directly rather than assuming.
- **Fix: `number_words.py`** - hand-written number-to-words for both
  languages, covering 0-199 with real number words (more than the
  current 4-field dummy schema's value range ever needs), falling back
  to reading digits one at a time above that rather than guessing an
  unverified compound hundreds form. Not verified by a native speaker -
  flagged as such in the module's own docstring. `can_telemetry.
  describe()` now spells out the number via `to_words()` for hi/ta
  before formatting the template, instead of leaving it as a raw digit.
- **Verified two ways**: tokenizer round-trip (encode then decode) is
  now lossless for all 4 fields in both languages, confirmed
  programmatically; then played the actual audio through the user's
  speakers and had them confirm the numbers were now correct by ear,
  not just by checking tokens.
- **Known limitation, explicitly not fixed**: the user separately asked
  whether the Tamil/Hindi phrasing could sound more colloquial (like
  how people actually talk at home) rather than the current formal/
  written register `TELEMETRY_FIELD_PHRASES` and `LOCAL_CHAT_SYSTEM_
  PROMPT` use. Discussed as options, not yet acted on: rewriting the
  fixed templates in colloquial Tamil (reliable, deterministic) vs.
  instructing Qwen's chat generation to use spoken-register Tamil
  (less reliable - same class of risk as the "don't guess" instruction
  that didn't fully work for Tamil in Phase 2+3). The MMS-TTS voice's
  own delivery style/formality is fixed by its training data and can't
  be changed by prompting either way. Also unresolved as of this
  session: whether `WHISPER_MODEL_SIZE` needs escalating past `medium`
  to `large-v3-turbo` for the persistent Tamil loanword-mistranscription
  problem (Phase 2+3) - downloaded and a live comparison test was
  handed to the user, but not yet run/reported back.

### Tech stack additions for V3

`llama-cpp-python`, `python-can`, Qwen2.5-1.5B-Instruct GGUF (q4_k_m -
3B tested and rejected, see Phase 2+3 investigation above),
`facebook/mms-tts-tam`/`facebook/mms-tts-hin` via `torch`+`transformers`
for Tamil/Hindi TTS (Piper kept for English; Kokoro-82M/MeloTTS ruled
out, neither covers Tamil or Hindi), multilingual `faster-whisper`
(`medium`, replaces `small.en`).

## Unrelated sibling project — do not confuse

There's a separate, unrelated EV ATV voice assistant project at
`D:\EV ATV VOICE PROJECT\ev-atv-voice-assistant` (friend-collaboration,
Vosk STT + keyword-rule intents, built 2026-07-30 to 2026-08-02, no
resort/autonomy vision). Same domain, different product — don't pull
code, conventions, or assumptions from it without deliberately deciding
to.

## Kickoff date

2026-08-05.
