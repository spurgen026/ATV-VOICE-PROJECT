from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

VEHICLE_STATE_PATH = DATA_DIR / "vehicle_state.json"

# Placeholder wake word until a custom "resort ATV" wake word is trained.
# openWakeWord ships this one pretrained; say "Hey Jarvis" to trigger listening.
WAKE_WORD_MODEL = "hey_jarvis_v0.1"
WAKE_WORD_THRESHOLD = 0.3

WHISPER_MODEL_SIZE = "small.en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

PIPER_VOICE_PATH = MODELS_DIR / "en_US-lessac-medium.onnx"

SAMPLE_RATE = 16000

# Query capture: record until trailing silence, bounded by a max duration and
# a timeout waiting for speech to start (in case the query got clipped or
# nothing was said after the wake word).
QUERY_CHUNK_MS = 20
QUERY_SILENCE_RMS_THRESHOLD = 500
QUERY_TRAILING_SILENCE_MS = 800
QUERY_SPEECH_TIMEOUT_MS = 3000
QUERY_MAX_MS = 8000
