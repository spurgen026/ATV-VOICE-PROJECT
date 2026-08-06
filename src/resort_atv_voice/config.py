from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DOCUMENTS_DIR = ROOT_DIR / "documents"
INDEX_DIR = ROOT_DIR / "index"

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

WHISPER_MODEL_SIZE = "small.en"
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
