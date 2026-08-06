import sounddevice as sd

from . import stt, tts, wake_word
from .config import FOLLOWUP_SPEECH_TIMEOUT_MS, QUERY_SPEECH_TIMEOUT_MS, STARTUP_GREETING
from .rag import answer_query, load_index

NO_SPEECH_RESPONSE = "Sorry, I didn't catch that."


def run() -> None:
    print("Loading models...")
    ww_model = wake_word.load_model()
    whisper_model = stt.load_model()
    voice = tts.load_voice()
    rag_index, rag_metadata = load_index()
    print("Ready.")
    tts.speak(voice, STARTUP_GREETING)
    print("Say the wake word...")

    try:
        while True:
            wake_word.wait_for_wake_word(ww_model)
            print("Wake word detected, listening...")
            tts.play_ack_chime()

            # First listen requires the wake word; after any answer, keep
            # listening for a follow-up without it, for as long as the user
            # keeps talking. Silence on the follow-up just goes back to
            # sleep quietly - only the very first miss gets a spoken nudge.
            # History resets every wake-word cycle - it's short-term memory
            # for one conversation, not carried across sleep.
            speech_timeout_ms = QUERY_SPEECH_TIMEOUT_MS
            heard_anything = False
            history = []
            while True:
                audio = stt.record_query(speech_timeout_ms=speech_timeout_ms)
                query = stt.transcribe(whisper_model, audio)

                if not query:
                    if not heard_anything:
                        print("No speech detected, going back to sleep.")
                        tts.speak(voice, NO_SPEECH_RESPONSE)
                    else:
                        print("No follow-up, going back to sleep.")
                        tts.play_sleep_chime()
                    break

                print(f"Heard: {query!r}")
                heard_anything = True

                response = answer_query(rag_index, rag_metadata, query, history=history)
                print(f"Responding: {response!r}")
                tts.speak(voice, response)
                history.append((query, response))

                speech_timeout_ms = FOLLOWUP_SPEECH_TIMEOUT_MS
    except sd.PortAudioError as exc:
        print(f"Audio device error, stopping: {exc}")


if __name__ == "__main__":
    run()
