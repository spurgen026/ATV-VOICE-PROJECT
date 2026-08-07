import numpy as np
import sounddevice as sd
import torch
from piper import PiperVoice
from transformers import AutoTokenizer, VitsModel

from .config import (
    CHIME_DURATION_MS,
    CHIME_FREQUENCY_HZ,
    CHIME_VOLUME,
    DEFAULT_LANGUAGE,
    MMS_TTS_MODEL_REPOS,
    PIPER_VOICE_PATH,
    SAMPLE_RATE,
    SLEEP_CHIME_DURATION_MS,
    SLEEP_CHIME_FREQUENCY_HZ,
    SLEEP_CHIME_VOLUME,
)


def _generate_tone(frequency_hz: float, duration_ms: float, volume: float) -> np.ndarray:
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
    tone = volume * np.sin(2 * np.pi * frequency_hz * t)

    fade_samples = max(1, int(num_samples * 0.1))
    envelope = np.ones(num_samples)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    return (tone * envelope).astype(np.float32)


class VoiceBundle:
    """English via Piper (kept from V1/V2), Hindi/Tamil via MMS-TTS
    (V3 Phase 5 - Piper's own voice catalog has zero Tamil voices and
    Kokoro-82M/MeloTTS cover neither language, confirmed by checking
    each catalog directly rather than assuming)."""

    def __init__(self, piper_voice: PiperVoice, mms_models: dict):
        self.piper_voice = piper_voice
        self.mms_models = mms_models  # {language: (VitsModel, tokenizer)}


def load_voices() -> VoiceBundle:
    piper_voice = PiperVoice.load(str(PIPER_VOICE_PATH))
    mms_models = {}
    for language, repo_id in MMS_TTS_MODEL_REPOS.items():
        model = VitsModel.from_pretrained(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        mms_models[language] = (model, tokenizer)
    return VoiceBundle(piper_voice, mms_models)


def _speak_piper(voice: PiperVoice, text: str) -> None:
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


def _speak_mms(model: VitsModel, tokenizer, text: str) -> None:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    audio = output.squeeze().numpy()
    sd.play(audio, samplerate=model.config.sampling_rate, blocking=True)


def speak(voices: VoiceBundle, text: str, language: str = DEFAULT_LANGUAGE) -> None:
    if not text:
        return
    if language in voices.mms_models:
        model, tokenizer = voices.mms_models[language]
        _speak_mms(model, tokenizer, text)
    else:
        _speak_piper(voices.piper_voice, text)


def play_ack_chime() -> None:
    """Plays a short tone right on wake-word detection, so there's
    immediate audio feedback while the slower recording + cloud
    round-trip happens."""
    tone = _generate_tone(CHIME_FREQUENCY_HZ, CHIME_DURATION_MS, CHIME_VOLUME)
    sd.play(tone, samplerate=SAMPLE_RATE, blocking=True)


def play_sleep_chime() -> None:
    """Plays a descending two-tone when the follow-up window times out
    silently, so it's audibly clear the assistant stopped listening even
    though nothing gets said."""
    high, low = SLEEP_CHIME_FREQUENCY_HZ
    tones = np.concatenate(
        [
            _generate_tone(high, SLEEP_CHIME_DURATION_MS, SLEEP_CHIME_VOLUME),
            _generate_tone(low, SLEEP_CHIME_DURATION_MS, SLEEP_CHIME_VOLUME),
        ]
    )
    sd.play(tones, samplerate=SAMPLE_RATE, blocking=True)
