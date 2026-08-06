import sounddevice as sd

from . import stt, tts, wake_word
from .config import FOLLOWUP_SPEECH_TIMEOUT_MS, QUERY_SPEECH_TIMEOUT_MS
from .rag import answer_query, load_index

NO_SPEECH_RESPONSE = "Sorry, I didn't catch that."


def run() -> None:
    print("Loading models...")
    ww_model = wake_word.load_model()
    whisper_model = stt.load_model()
    voice = tts.load_voice()
    rag_index, rag_metadata = load_index()
    print("Ready. Say the wake word...")

    try:
        while True:
            wake_word.wait_for_wake_word(ww_model)
            print("Wake word detected, listening...")
            tts.play_ack_chime()

            # First listen requires the wake word; after any answer, keep
            # listening for a follow-up without it, for as long as the user
            # keeps talking. Silence on the follow-up just goes back to
            # sleep quietly - only the very first miss gets a spoken nudge.
            speech_timeout_ms = QUERY_SPEECH_TIMEOUT_MS
            heard_anything = False
            while True:
                audio = stt.record_query(speech_timeout_ms=speech_timeout_ms)
                query = stt.transcribe(whisper_model, audio)

                if not query:
                    if not heard_anything:
                        print("No speech detected, going back to sleep.")
                        tts.speak(voice, NO_SPEECH_RESPONSE)
                    else:
                        print("No follow-up, going back to sleep.")
                    break

                print(f"Heard: {query!r}")
                heard_anything = True

                response = answer_query(rag_index, rag_metadata, query)
                print(f"Responding: {response!r}")
                tts.speak(voice, response)

                speech_timeout_ms = FOLLOWUP_SPEECH_TIMEOUT_MS
    except sd.PortAudioError as exc:
        print(f"Audio device error, stopping: {exc}")


if __name__ == "__main__":
    run()
