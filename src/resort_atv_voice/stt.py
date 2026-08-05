import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from .config import (
    QUERY_CHUNK_MS,
    QUERY_MAX_MS,
    QUERY_SILENCE_RMS_THRESHOLD,
    QUERY_SPEECH_TIMEOUT_MS,
    QUERY_TRAILING_SILENCE_MS,
    SAMPLE_RATE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)

CHUNK_SAMPLES = int(SAMPLE_RATE * QUERY_CHUNK_MS / 1000)
SILENCE_CHUNKS_TO_STOP = QUERY_TRAILING_SILENCE_MS // QUERY_CHUNK_MS
SPEECH_TIMEOUT_CHUNKS = QUERY_SPEECH_TIMEOUT_MS // QUERY_CHUNK_MS
MAX_CHUNKS = QUERY_MAX_MS // QUERY_CHUNK_MS


def load_model() -> WhisperModel:
    return WhisperModel(
        WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
    )


def record_query() -> np.ndarray:
    """Records from the default input device after the wake word fires, stopping
    once trailing silence is detected. Gives up if speech never starts, or after
    QUERY_MAX_MS regardless, so a stuck mic can't hang the assistant forever."""
    chunks = []
    speech_started = False
    silence_run = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        for chunks_read in range(MAX_CHUNKS):
            frame, _ = stream.read(CHUNK_SAMPLES)
            frame = frame.reshape(-1)
            rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))

            if rms >= QUERY_SILENCE_RMS_THRESHOLD:
                speech_started = True
                silence_run = 0
                chunks.append(frame)
            elif speech_started:
                silence_run += 1
                chunks.append(frame)
                if silence_run >= SILENCE_CHUNKS_TO_STOP:
                    break
            elif chunks_read >= SPEECH_TIMEOUT_CHUNKS:
                break

    if not chunks:
        return np.array([], dtype=np.float32)

    audio_int16 = np.concatenate(chunks)
    return audio_int16.astype(np.float32) / 32768.0


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    segments, _ = model.transcribe(audio, language="en")
    return " ".join(segment.text.strip() for segment in segments).strip()
