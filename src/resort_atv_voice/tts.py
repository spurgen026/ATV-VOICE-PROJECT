import numpy as np
import sounddevice as sd
from piper import PiperVoice

from .config import (
    CHIME_DURATION_MS,
    CHIME_FREQUENCY_HZ,
    CHIME_VOLUME,
    PIPER_VOICE_PATH,
    SAMPLE_RATE,
)


def load_voice() -> PiperVoice:
    return PiperVoice.load(str(PIPER_VOICE_PATH))


def speak(voice: PiperVoice, text: str) -> None:
    # Piper yields one chunk per sentence. Playing each chunk with its own
    # sd.play() call reopens the audio stream every time, which can click or
    # clip audio at sentence boundaries - most noticeable on a short trailing
    # sentence. Concatenating into one buffer plays the whole response as a
    # single continuous stream instead.
    chunks = list(voice.synthesize(text))
    if not chunks:
        return
    audio = np.concatenate([chunk.audio_int16_array for chunk in chunks])
    sd.play(audio, samplerate=chunks[0].sample_rate, blocking=True)


def play_ack_chime() -> None:
    """Plays a short tone right on wake-word detection, so there's
    immediate audio feedback while the slower recording + cloud
    round-trip happens."""
    num_samples = int(SAMPLE_RATE * CHIME_DURATION_MS / 1000)
    t = np.linspace(0, CHIME_DURATION_MS / 1000, num_samples, endpoint=False)
    tone = CHIME_VOLUME * np.sin(2 * np.pi * CHIME_FREQUENCY_HZ * t)

    fade_samples = max(1, int(num_samples * 0.1))
    envelope = np.ones(num_samples)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    sd.play((tone * envelope).astype(np.float32), samplerate=SAMPLE_RATE, blocking=True)
