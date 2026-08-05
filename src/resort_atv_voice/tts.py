import sounddevice as sd
from piper import PiperVoice

from .config import PIPER_VOICE_PATH


def load_voice() -> PiperVoice:
    return PiperVoice.load(str(PIPER_VOICE_PATH))


def speak(voice: PiperVoice, text: str) -> None:
    for chunk in voice.synthesize(text):
        sd.play(chunk.audio_int16_array, samplerate=chunk.sample_rate, blocking=True)
