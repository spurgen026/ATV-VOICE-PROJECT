from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DOCUMENTS_DIR = ROOT_DIR / "documents"
INDEX_DIR = ROOT_DIR / "index"
GRAMMARS_DIR = ROOT_DIR / "grammars"

VEHICLE_STATE_PATH = DATA_DIR / "vehicle_state.json"
VEHICLE_STATUS_DOC_PATH = DOCUMENTS_DIR / "vehicle_status.txt"

FAISS_INDEX_PATH = INDEX_DIR / "vectors.faiss"
INDEX_METADATA_PATH = INDEX_DIR / "metadata.json"

# V2: document Q&A via Gemini. Embeddings + reasoning are cloud calls;
# wake word/STT/TTS stay local (see CLAUDE.md V2 plan for why).
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_CHAT_MODEL = "gemini-flash-latest"
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 100
TOP_K_CHUNKS = 5

# Short-term conversational memory, reset each time the assistant goes
# back to sleep. Kept small so the prompt doesn't grow unbounded across
# a long back-and-forth.
MAX_HISTORY_TURNS = 4

RAG_SYSTEM_PROMPT = """You are the voice of a resort ATV - a friendly,
easygoing companion for the guest riding you, not a generic assistant.
Speak warmly and conversationally, like a knowledgeable friend, but keep
answers brief and to the point since they'll be spoken aloud.
Answer the question directly and then stop - do not add closing remarks,
filler sign-offs, or questions like "let me know if you need anything else"
or "ready whenever you are."
Use ONLY the supplied document context to answer the question.
Do not use external knowledge, even if you happen to know the real answer.
If the answer is not found in the provided context, reply exactly:
"I couldn't find that information in the provided documents."
Never guess, even to be helpful."""

# Placeholder wake word until a custom "resort ATV" wake word is trained.
# openWakeWord ships this one pretrained; say "Hey Jarvis" to trigger listening.
WAKE_WORD_MODEL = "hey_jarvis_v0.1"
WAKE_WORD_THRESHOLD = 0.3

# V3: multilingual (Tamil/Hindi/English) auto-detect needs a non-".en"
# model - the ".en" variants are English-only and cannot transcribe the
# other two languages at all. "small" was tried first and live-tested
# 2026-08-07: it consistently mangled English loanwords embedded in
# Hindi/Tamil speech ("battery" -> garbled non-words), once causing a
# confidently wrong telemetry field instead of a safe "didn't understand"
# - escalated to "medium" per the plan's own fallback.
WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

PIPER_VOICE_PATH = MODELS_DIR / "en_GB-cori-high.onnx"

SAMPLE_RATE = 16000

# Spoken once, right after models finish loading - a companion should
# announce itself, not just silently sit ready in a terminal.
STARTUP_GREETING = "Hey, I'm up and ready to help with your ride."

# Wake-word acknowledgment chime: a short tone played immediately on
# detection, before recording/the cloud round-trip, so there's audio
# feedback while the (noticeably slower, V2) answer is being fetched.
CHIME_FREQUENCY_HZ = 1000
CHIME_DURATION_MS = 250
CHIME_VOLUME = 0.7

# Sleep chime: a descending two-tone played when the follow-up window
# times out silently, so it's clear (without speaking) that the
# assistant has stopped listening - the mirror image of the ack chime.
SLEEP_CHIME_FREQUENCY_HZ = (700, 450)
SLEEP_CHIME_DURATION_MS = 130
SLEEP_CHIME_VOLUME = 0.6

# Query capture: record until trailing silence, bounded by a max duration and
# a timeout waiting for speech to start (in case the query got clipped or
# nothing was said after the wake word).
QUERY_CHUNK_MS = 20
QUERY_SILENCE_RMS_THRESHOLD = 500
QUERY_TRAILING_SILENCE_MS = 800
QUERY_SPEECH_TIMEOUT_MS = 3000
QUERY_MAX_MS = 8000

# After answering, listen for a follow-up without needing the wake word
# again. Same VAD parameters as above, just a shorter, quieter timeout —
# no "didn't catch that" if nobody follows up, it just goes back to sleep.
FOLLOWUP_SPEECH_TIMEOUT_MS = 4000

# V3 Phase 3: local Qwen router, GBNF-grammar-constrained so it only ever
# picks *which* telemetry field to fetch - a deterministic lookup supplies
# the actual number, so it can never hallucinate one. Starting with the
# 1.5B tier (lightest, fastest); escalate to 3B if routing accuracy on
# the benchmark set isn't good enough.
QWEN_MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
# Exact filename, not a glob ("*q4_k_m.gguf") - found the hard way that
# Llama.from_pretrained() resolves a glob via a live HF Hub API call even
# when the file is already fully cached, which breaks true offline startup.
# An exact filename skips that lookup entirely and resolves from cache.
QWEN_MODEL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
ROUTER_GRAMMAR_PATH = GRAMMARS_DIR / "telemetry_router.gbnf"
ROUTER_CONTEXT_SIZE = 2048

ROUTER_SYSTEM_PROMPT = """You are a routing classifier for a resort ATV's
voice assistant. Decide whether the rider's message is asking for a
telemetry reading, or is anything else (small talk, general questions).

If it's a telemetry request, respond with:
{"action": "get_telemetry", "target": "<field>"}
where <field> is exactly one of: battery_percent, motor_temp_c,
speed_kmh, tire_pressure_psi.

If it's anything else, respond with:
{"action": "chat"}

Only ever output one of those two JSON shapes. Never invent a number
yourself - you only ever pick which field to look up.

Riders speak Tamil, Hindi, or English, often mixing in English loanwords
for vehicle terms (e.g. "battery", "tire"/"tyre", "pressure", "speed")
inside a Tamil/Hindi sentence, and speech-to-text transcription of those
loanwords is sometimes imperfect - route on the intent, not exact
spelling.

If you are not reasonably confident which specific telemetry field is
being asked about, respond with {"action": "chat"} rather than guessing.
A wrong guess reports fabricated vehicle data as fact, which is worse
than admitting you did not understand."""

# Few-shot examples added 2026-08-07 after live-testing showed the router
# missing Tamil-script telemetry requests it should recognize (e.g. a
# correctly-transcribed "tyre pressure" phrase in Tamil script still got
# routed to chat). Deliberately different phrasings from anything used in
# testing, so improvement reflects genuine generalization, not the model
# echoing memorized examples back.
ROUTER_FEWSHOT_EXAMPLES = [
    ("बैटरी लेवल क्या है", {"action": "get_telemetry", "target": "battery_percent"}),
    ("ஸ்பீட் எவ்வளவு இருக்கு", {"action": "get_telemetry", "target": "speed_kmh"}),
    ("பேட்டரி பர்சென்ட் சொல்லு", {"action": "get_telemetry", "target": "battery_percent"}),
    ("टायर प्रेशर कितना है", {"action": "get_telemetry", "target": "tire_pressure_psi"}),
    ("வணக்கம்", {"action": "chat"}),
    ("धन्यवाद दोस्त", {"action": "chat"}),
]

# V3 Phase 6: local chat generation for anything the router classifies as
# "chat" and small_talk.py doesn't already handle - same Qwen model as the
# router, just without the grammar constraint, so V2's Gemini path is no
# longer needed for the live app to answer anything. {language_name} is
# filled in per-query so the reply comes back in the same language the
# rider spoke, not always English - Qwen2.5 is multilingual, it just needs
# to be told to answer in-language rather than defaulting to English.
LOCAL_CHAT_SYSTEM_PROMPT = """You are the voice of a resort ATV - a friendly,
easygoing companion for the guest riding you, not a generic assistant.
Speak warmly and conversationally, like a knowledgeable friend, but keep
answers brief since they'll be spoken aloud. Do not add closing filler
("let me know if you need anything else", "ready whenever you are").
You run fully offline on the vehicle itself - no internet, no resort
documents or booking systems yet. If asked something you genuinely don't
know, say so honestly instead of guessing - do not invent specific resort
amenities, activities, or facts you have not been given.
Respond in {language_name}, matching the language the rider spoke in."""

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "ta": "Tamil"}
DEFAULT_LANGUAGE = "en"

# Phrases for the get_telemetry branch, per language - the router only ever
# picks which field to fetch, the actual number always comes from the CAN
# telemetry cache, never generated by a model, so it can't be hallucinated.
# Only the phrasing around the number is language-specific template text,
# written directly rather than machine-translated at runtime.
TELEMETRY_FIELD_PHRASES = {
    "en": {
        "battery_percent": "The battery is at {value} percent.",
        "motor_temp_c": "The motor temperature is {value} degrees Celsius.",
        "speed_kmh": "The current speed is {value} kilometers per hour.",
        "tire_pressure_psi": "The tire pressure is {value} psi.",
    },
    "hi": {
        "battery_percent": "बैटरी {value} प्रतिशत है।",
        "motor_temp_c": "मोटर का तापमान {value} डिग्री सेल्सियस है।",
        "speed_kmh": "वर्तमान गति {value} किलोमीटर प्रति घंटा है।",
        "tire_pressure_psi": "टायर का दबाव {value} पीएसआई है।",
    },
    "ta": {
        "battery_percent": "பேட்டரி {value} சதவீதம் உள்ளது.",
        "motor_temp_c": "மோட்டார் வெப்பநிலை {value} டிகிரி செல்சியஸ் ஆகும்.",
        "speed_kmh": "தற்போதைய வேகம் மணிக்கு {value} கிலோமீட்டர் ஆகும்.",
        "tire_pressure_psi": "டயர் அழுத்தம் {value} பிஎஸ்ஐ ஆகும்.",
    },
}
TELEMETRY_UNAVAILABLE_RESPONSE = {
    "en": "Sorry, I don't have that reading yet.",
    "hi": "क्षमा करें, मेरे पास अभी वह रीडिंग नहीं है।",
    "ta": "மன்னிக்கவும், அந்த அளவீடு இன்னும் என்னிடம் இல்லை.",
}

# V3 Phase 5: Tamil/Hindi TTS. Piper's official voice catalog has zero
# Tamil voices (confirmed via huggingface_hub.list_repo_files against
# rhasspy/piper-voices, not assumed) and Kokoro-82M/MeloTTS - the plan's
# other candidates - only cover English/a handful of other languages,
# neither Tamil nor Hindi. Meta's MMS-TTS (VITS via transformers) actually
# covers both, live-tested for realtime factor and voice quality with the
# user 2026-08-07. Piper is kept for English (already integrated, "kept"
# per the V3 architecture).
MMS_TTS_MODEL_REPOS = {
    "hi": "facebook/mms-tts-hin",
    "ta": "facebook/mms-tts-tam",
}
