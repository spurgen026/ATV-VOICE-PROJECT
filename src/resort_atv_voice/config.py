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
# - escalated to "medium" per the plan's own fallback. Escalated again to
# "large-v3-turbo" 2026-08-08 after a live head-to-head comparison
# (compare_whisper_sizes.py, 8 real spoken phrases, same audio fed to both
# models): 7/8 vs medium's 5/8, and specifically fixed the two dangerous
# confidently-wrong-field cases (Hindi and Tamil "battery" misrouting).
# One case (Tamil tire-pressure -> speed_kmh) still fails on both models
# despite a legible transcription from either - confirmed router-side, not
# an STT problem; see CLAUDE.md for the open item this leaves.
#
# WHISPER_DEVICE moved "cpu" -> "cuda" 2026-08-08: this laptop has an
# unused NVIDIA RTX 3050 - discovered while chasing the CPU latency problem
# above (medium ~8s/query, large-v3-turbo ~11s/query on CPU, essentially
# flat regardless of utterance length, confirmed via benchmark_whisper_
# latency.py). GPU inference measured at ~0.9s/query for large-v3-turbo
# (RTF 0.51, i.e. faster than real-time) - solves the latency problem
# without giving up the accuracy win, no CPU-vs-accuracy tradeoff needed.
# Requires nvidia-cublas-cu12/nvidia-cudnn-cu12/nvidia-cuda-runtime-cu12
# (pip-installable redistributable DLLs, no full CUDA Toolkit installer
# needed) - see stt.py's PATH setup at import time for why plain `pip
# install` isn't enough on its own (CTranslate2 loads them via LoadLibrary(),
# which only honors PATH, not os.add_dll_directory()).
WHISPER_MODEL_SIZE = "large-v3-turbo"
WHISPER_DEVICE = "cuda"
# int8_float16 over plain float16 2026-08-08: measured lower VRAM (2.85GB
# vs 3.79GB - this 4GB card only has ~1.2GB free after OS/desktop overhead
# once Whisper is loaded) AND lower latency (0.80s vs 1.09s on a 2s clip) -
# a genuine free win, not a tradeoff. Transcription quality checked on 5
# synthesized phrases: 4/5 identical output, the 1 difference was on an
# already-garbled synthetic Tamil sample in both versions (synthesis
# artifact, not a real quality regression).
WHISPER_COMPUTE_TYPE = "int8_float16"

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

# Echo/feedback mitigation (2026-08-08): every test of this app so far has
# used headphones, which physically isolate the speaker output from the
# mic - the eventual target (onboard vehicle-mounted mics/speakers, see
# product vision in this file's header) has no such isolation, and the mic
# will pick up the assistant's own TTS/chime output. record_query() briefly
# pauses before it starts actively listening for speech, so an acoustic
# echo/reverb tail from whatever just played has time to dissipate before
# the RMS-based speech-start detection begins - otherwise that tail risks
# being misread as the start of a new query.
#
# This is a partial mitigation, not real acoustic echo cancellation (AEC) -
# Windows does have OS-level AEC (WASAPI Audio Processing Objects), but
# `sounddevice` doesn't expose it, and building against a Windows-specific
# API here would be wasted effort anyway: the actual target hardware
# (Raspberry Pi 5) runs Linux, not Windows, and would need a genuinely
# different, portable AEC approach (e.g. WebRTC's AEC module) - one that
# also needs real speaker/mic placement to tune against, which doesn't
# exist yet. Tracked as a real open item, not solved here - see CLAUDE.md.
MIC_SETTLE_MS = 300

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

# V3 hardening round 2 (2026-08-08): separate, larger model just for
# free-form chat ("make it more knowledgeable locally without cloud").
# The router stays on the 1.5B model above - it's a strict classification
# task the small model already handles well (Phase 3: 18/18 English,
# CPU latency ~0.85s avg), no reason to pay a bigger model's cost there.
# Chat is a different task (open-ended generation), where model size
# genuinely matters more. Tested three CPU options head-to-head on the
# same technical questions before choosing: 1.5B (already had it, decent
# but rambled on one multi-scenario question), 7B (better tone, but
# 3.5-12s latency - too slow for a voice interface), 3B (best of both -
# cleaner, more concise answers than either 1.5B or 7B, latency 1.5-4.6s,
# close to what 1.5B already cost). A tighter system prompt asking the 1.5B
# model directly for "one clear answer, not a list of scenarios" was tried
# first, for free, before downloading anything - it didn't work, same
# unreliable-instruction-following pattern already seen for the router's
# "don't guess" instruction (Phase 2+3) and the chat model's
# fabricated-amenities problem - confirming this needed an actual model
# change, not just better wording.
CHAT_MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
CHAT_MODEL_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
CHAT_MODEL_CONTEXT_SIZE = 2048

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
    # Added 2026-08-08 from two real live misroutes captured during an
    # actual main.py run (see CLAUDE.md "V3 hardening, round 2") - garbled
    # STT transcripts, not clean scripted text, deliberately kept as-is
    # (garbling included) since that's what the router actually has to
    # handle in practice.
    ("ஐயர் பிரசர் எவலவு இருக்கிறு", {"action": "get_telemetry", "target": "tire_pressure_psi"}),
    ("வெளியா வதர் எப்படி இருக்கு?", {"action": "chat"}),
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
When speaking Hindi or Tamil, use everyday spoken language, the way
people actually talk in conversation - not formal written/literary
Hindi or Tamil.
Respond in {language_name}, matching the language the rider spoke in."""
# Added the Hindi/Tamil spoken-register line 2026-08-08, low confidence -
# this codebase has repeatedly found prompt-only instructions unreliable
# at this model size (the router's "don't guess" instruction in Phase 2+3,
# and this same prompt's "don't invent amenities" line both needed a
# deterministic gate, not just wording, to actually work). The templates
# in TELEMETRY_FIELD_PHRASES/small_talk.py are the real, reliable fix for
# colloquial phrasing since they're fixed text, not generated - this line
# is a low-cost bonus attempt for the free-form chat path specifically,
# where there's no deterministic alternative.

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "ta": "Tamil"}
DEFAULT_LANGUAGE = "en"

# Phrases for the get_telemetry branch, per language - the router only ever
# picks which field to fetch, the actual number always comes from the CAN
# telemetry cache, never generated by a model, so it can't be hallucinated.
# Only the phrasing around the number is language-specific template text,
# written directly rather than machine-translated at runtime.
#
# Hindi/Tamil rewritten to colloquial/spoken register 2026-08-08 (were
# formal/written Tamil and Hindi before - user feedback: "how people
# actually talk at home", tracked as an open item since Phase 5 hardening).
# Not verified by a native speaker - same caveat as number_words.py -
# needs an actual listen-through to confirm, not just a code review.
# Tamil: swapped the literary ஆகும்/உள்ளது verb endings for spoken இருக்கு -
# this specific formal-vs-spoken split is one of the most well-documented
# features of Tamil diglossia, so this is the highest-confidence change
# here. Also dropped தற்போதைய ("current", a written-register adjective) for
# இப்போ ("now", spoken), and அழுத்தம்/வெப்பநிலை (formal Tamil nouns) for the
# loanwords பிரஷர்/டெம்பரேச்சர் where the app's own router few-shot examples
# already show riders naturally saying "பிரஷர்"/"ஸ்பீட்" instead of the
# formal Tamil words. Hindi: swapped प्रतिशत/दबाव for the loanwords
# पर्सेंट/प्रेशर (already how बैटरी/मोटर/टायर are used elsewhere in this
# app), dropped the formal वर्तमान ("current") and क्षमा करें ("forgive me",
# quite formal) for a plainer, more spoken tone.
TELEMETRY_FIELD_PHRASES = {
    "en": {
        "battery_percent": "The battery is at {value} percent.",
        "motor_temp_c": "The motor temperature is {value} degrees Celsius.",
        "speed_kmh": "The current speed is {value} kilometers per hour.",
        "tire_pressure_psi": "The tire pressure is {value} psi.",
    },
    "hi": {
        "battery_percent": "बैटरी {value} पर्सेंट है।",
        "motor_temp_c": "मोटर का टेम्परेचर {value} डिग्री सेल्सियस है।",
        "speed_kmh": "अभी स्पीड {value} किलोमीटर प्रति घंटा है।",
        "tire_pressure_psi": "टायर प्रेशर {value} पीएसआई है।",
    },
    "ta": {
        "battery_percent": "பேட்டரி {value} சதவீதம் இருக்கு.",
        "motor_temp_c": "மோட்டர் டெம்பரேச்சர் {value} டிகிரி இருக்கு.",
        "speed_kmh": "இப்போ ஸ்பீட் மணிக்கு {value} கிலோமீட்டர் இருக்கு.",
        "tire_pressure_psi": "டயர் பிரஷர் {value} பிஎஸ்ஐ இருக்கு.",
    },
}
TELEMETRY_UNAVAILABLE_RESPONSE = {
    "en": "Sorry, I don't have that reading yet.",
    "hi": "सॉरी, अभी वो रीडिंग मेरे पास नहीं है।",
    "ta": "சாரி, அது இன்னும் என்கிட்ட இல்ல.",
}

# V3 hardening round 2 (production hardening pass): main.py's per-turn
# error handling speaks this instead of crashing the whole app when
# something in transcribe()/answer_query()/speak() throws unexpectedly -
# a stack trace and dead process is a much worse outcome mid-ride than an
# apology and going back to sleep. Colloquial register, matching the
# other Hindi/Tamil rewrites - same not-native-verified caveat.
UNEXPECTED_ERROR_RESPONSE = {
    "en": "Sorry, something went wrong there. Try again?",
    "hi": "सॉरी, कुछ गड़बड़ हो गई। फिर से बोलिए?",
    "ta": "சாரி, ஏதோ தவறு நடந்துடுச்சு. மறுபடி சொல்லுங்க?",
}

# V3 hardening round 2 continued (2026-08-08): deterministic safety net for
# the router's remaining known-dangerous failure - a Tamil non-telemetry
# question ('வெளியா வதர் எப்படி இருக்கு?', reads as "how's the weather
# outside") still misrouted to speed_kmh even as a verbatim few-shot
# example (see CLAUDE.md "V3 hardening, round 2"). Same fix philosophy as
# RESORT_KNOWLEDGE_TRIGGERS: don't trust the LLM's classification alone,
# gate deterministically. This one only ever downgrades get_telemetry ->
# chat, never picks a field itself and never upgrades chat -> telemetry -
# it's a safety net, not a second router.
#
# Threshold tuned empirically against real transcripts (difflib.
# SequenceMatcher.ratio(), character-level): the weakest genuine
# vehicle-term token seen so far ('ச்பீட்டில', a garbled "speed") scores
# 0.80 against its keyword; the highest-scoring *non*-vehicle token seen
# ('வதர்', "weather", against "தயர்"/tire) scores 0.75. 0.8 cleanly
# separates the two in this test set, but it's a small set - a stricter
# threshold is the safe failure direction here (a missed real vehicle term
# just falls back to chat, an honest "didn't understand," rather than a
# wrong field), so err toward missing over false-triggering.
VEHICLE_TERM_KEYWORDS = (
    "battery", "बैटरी", "பேட்டரி", "பாட்டரி", "பாட்டிரி", "बाट्री", "बाटरी",
    "tire", "tyre", "टायर", "டயர்", "தயர்", "ஐயர்",
    "pressure", "प्रेशर", "प्रशर", "பிரஷர்", "பிரசர்", "ப்ரச்சர்",
    "motor", "मोटर", "மோட்டர்", "மோட்டார்",
    "temperature", "टेम्परेचर", "तापमान", "டெம்பரேச்சர்", "வெப்பநிலை",
    "speed", "स्पीड", "ஸ்பீட்", "ச்பீட்",
    "percent", "पर्सेंट", "சதவீதம்",
)
VEHICLE_TERM_FUZZY_THRESHOLD = 0.8

# V3 hardening: LOCAL_CHAT_SYSTEM_PROMPT already told the model not to
# invent resort amenities, and Phase 6 testing showed it did so anyway
# ("a scenic boat ride around the lake") - the same unreliable-uncertainty
# pattern documented for the router in Phase 2+3, where prompt wording
# alone didn't unlock honest "I don't know" behavior at this model size.
# Same fix as the router's grammar constraint: don't trust the LLM's
# self-restraint, gate deterministically before generation is ever called.
# English-only keyword match and not an exhaustive phrase list, so some
# resort-knowledge questions will still slip through to free generation -
# same known-limitation shape as small_talk.py's language coverage.
RESORT_KNOWLEDGE_TRIGGERS = (
    "activity", "activities", "recommend", "sightsee", "restaurant",
    "dining", "dinner", "lunch", "breakfast", "menu", "book a table",
    "reservation", "hotel", "amenity", "amenities", "pool", "spa",
    "trail", "hike", "hiking", "boat ride", "lake", "waterfall",
    "things to do", "what should i do", "where can i",
)

# Rewritten colloquial 2026-08-08, same caveat as TELEMETRY_FIELD_PHRASES
# above - not native-verified. Kept the respectful आप/உங்கள் address form
# used elsewhere in this app (a resort guest, not a close friend) -
# colloquial vocabulary/structure, not informal-address - rather than
# switching to तुम/உன் which would read as presumptuous for a guest.
RESORT_KNOWLEDGE_UNAVAILABLE_RESPONSE = {
    "en": "I don't have real information about resort activities or amenities yet - I can only help with your vehicle for now.",
    "hi": "रिसॉर्ट की एक्टिविटीज़ या सुविधाओं के बारे में मुझे अभी सही जानकारी नहीं है - फिलहाल मैं बस आपकी गाड़ी के बारे में मदद कर सकता हूं।",
    "ta": "ரிசார்ட் ஆக்டிவிட்டீஸ் பத்தி இன்னும் சரியா எனக்குத் தெரியல - தற்போதைக்கு உங்க வண்டி பத்தி மட்டும்தான் உதவ முடியும்.",
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
