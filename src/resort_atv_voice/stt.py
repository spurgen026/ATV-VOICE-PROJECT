import os
import sys

# V3 hardening round 2 (2026-08-08): CTranslate2's CUDA backend loads
# cublas/cudnn via plain LoadLibrary(), which only honors PATH, not
# os.add_dll_directory() - confirmed by testing both directly (ctypes.WinDLL
# could load the same DLL via add_dll_directory, CTranslate2 still couldn't).
# No full CUDA Toolkit is installed on this machine, just the pip-installable
# redistributable DLL wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12,
# nvidia-cuda-runtime-cu12), which land in the venv's site-packages (D: drive,
# not C:) rather than a multi-GB installer. Must run before the faster_whisper
# import below, since that's what pulls in ctranslate2.
_venv_site_packages = os.path.abspath(
    os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages")
)
_cuda_dll_dirs = [
    os.path.join(_venv_site_packages, "nvidia", pkg, "bin")
    for pkg in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc")
]
_cuda_dll_dirs = [d for d in _cuda_dll_dirs if os.path.isdir(d)]
if _cuda_dll_dirs:
    os.environ["PATH"] = os.pathsep.join(_cuda_dll_dirs) + os.pathsep + os.environ["PATH"]

import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from .config import (
    DEFAULT_LANGUAGE,
    LANGUAGE_NAMES,
    MIC_SETTLE_MS,
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
MAX_CHUNKS = QUERY_MAX_MS // QUERY_CHUNK_MS


# Used only if WHISPER_DEVICE fails to load - matches the compute type
# this app used before GPU acceleration was added (V3 hardening round 2).
CPU_FALLBACK_COMPUTE_TYPE = "int8"


def load_model() -> WhisperModel:
    """Loads Whisper on WHISPER_DEVICE, falling back to CPU if that fails.

    A GPU-specific failure (missing CUDA runtime DLLs, no compatible GPU,
    a driver mismatch) would otherwise crash the whole app at startup
    instead of degrading to a slower-but-working CPU path. This is a real
    risk, not a theoretical one: this app's eventual target hardware
    (Raspberry Pi 5, see CLAUDE.md "Version 3 Plan") has no discrete GPU
    at all, so this fallback is what keeps a laptop-tuned
    WHISPER_DEVICE = "cuda" from being a hard requirement everywhere this
    code runs."""
    try:
        return WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    except Exception as exc:
        if WHISPER_DEVICE == "cpu":
            raise
        print(f"Whisper failed to load on device={WHISPER_DEVICE!r} ({exc}) - falling back to CPU.")
        return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=CPU_FALLBACK_COMPUTE_TYPE)


def record_query(speech_timeout_ms: int = QUERY_SPEECH_TIMEOUT_MS) -> np.ndarray:
    """Records from the default input device, stopping once trailing silence
    is detected. Gives up if speech never starts within speech_timeout_ms, or
    after QUERY_MAX_MS regardless, so a stuck mic can't hang the assistant
    forever. A shorter speech_timeout_ms is used for the post-answer
    follow-up window vs. the initial post-wake-word listen.

    Waits MIC_SETTLE_MS before opening the input stream - every call site
    (after a chime, after a spoken response) follows something that just
    played through the speaker, and on real (non-headphone) hardware the
    mic would otherwise risk picking up that output's own acoustic tail as
    if it were the start of a new query. See MIC_SETTLE_MS in config.py."""
    time.sleep(MIC_SETTLE_MS / 1000)

    speech_timeout_chunks = speech_timeout_ms // QUERY_CHUNK_MS
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
            elif chunks_read >= speech_timeout_chunks:
                break

    if not chunks:
        return np.array([], dtype=np.float32)

    audio_int16 = np.concatenate(chunks)
    return audio_int16.astype(np.float32) / 32768.0


def transcribe(model: WhisperModel, audio: np.ndarray) -> tuple[str, str]:
    """Returns (text, language_code). language_code drives which response
    templates/TTS voice get used downstream, so callers need it, not just
    the text."""
    if audio.size == 0:
        return "", DEFAULT_LANGUAGE

    # Live testing 2026-08-08 found unconstrained auto-detect (language=
    # None across Whisper's full ~99-language set) picking irrelevant
    # languages on short/ambiguous audio - e.g. "Bye" detected as Urdu,
    # a mumbled utterance hallucinated as Spanish - producing text nothing
    # downstream can handle. This app only ever supports English/Hindi/
    # Tamil (V3 trilingual requirement, no manual mode switch), so
    # constrain detection to just those three: detect_language() returns
    # per-language probabilities for the full set, pick the best-scoring
    # of our 3 supported ones, then force transcribe() to that language
    # rather than letting it re-detect from the full set.
    _, _, language_probs = model.detect_language(audio)
    language = max(
        (entry for entry in language_probs if entry[0] in LANGUAGE_NAMES),
        key=lambda entry: entry[1],
        default=(DEFAULT_LANGUAGE, 0.0),
    )[0]

    segments, info = model.transcribe(audio, language=language)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, info.language
