import sounddevice as sd

from . import stt, tts, wake_word
from .intents import handle_query

NO_SPEECH_RESPONSE = "Sorry, I didn't catch that."


def run() -> None:
    print("Loading models...")
    ww_model = wake_word.load_model()
    whisper_model = stt.load_model()
    voice = tts.load_voice()
    print("Ready. Say the wake word...")

    try:
        while True:
            wake_word.wait_for_wake_word(ww_model)
            print("Wake word detected, listening...")

            audio = stt.record_query()
            query = stt.transcribe(whisper_model, audio)
            print(f"Heard: {query!r}")

            if not query:
                print("No speech detected, going back to sleep.")
                tts.speak(voice, NO_SPEECH_RESPONSE)
                continue

            response = handle_query(query)
            print(f"Responding: {response!r}")
            tts.speak(voice, response)
    except sd.PortAudioError as exc:
        print(f"Audio device error, stopping: {exc}")


if __name__ == "__main__":
    run()
