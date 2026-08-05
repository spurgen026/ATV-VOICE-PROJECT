import numpy as np
import sounddevice as sd
from openwakeword.model import Model
from openwakeword.utils import download_models

from .config import SAMPLE_RATE, WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD

FRAME_SAMPLES = 1280  # 80ms at 16kHz, openWakeWord's expected chunk size


def load_model() -> Model:
    download_models([WAKE_WORD_MODEL])
    return Model(wakeword_models=[WAKE_WORD_MODEL], inference_framework="onnx")


def wait_for_wake_word(model: Model) -> None:
    """Blocks until the wake word is heard on the default input device."""
    model.reset()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while True:
            frame, _ = stream.read(FRAME_SAMPLES)
            audio = frame.reshape(-1)
            predictions = model.predict(audio.astype(np.int16))
            if predictions.get(WAKE_WORD_MODEL, 0.0) >= WAKE_WORD_THRESHOLD:
                return
