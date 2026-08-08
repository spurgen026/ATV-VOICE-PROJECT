import sounddevice as sd

from . import stt, tts, wake_word
from .can_telemetry import TelemetryCache, start_fake_ecu, start_listener
from .config import (
    DEFAULT_LANGUAGE,
    FOLLOWUP_SPEECH_TIMEOUT_MS,
    QUERY_SPEECH_TIMEOUT_MS,
    STARTUP_GREETING,
    UNEXPECTED_ERROR_RESPONSE,
)
from .local_qa import answer_query
from .router import load_chat_model, load_grammar, load_router_model, load_tamil_chat_model

NO_SPEECH_RESPONSE = "Sorry, I didn't catch that."


def run() -> None:
    print("Loading models...")
    ww_model = wake_word.load_model()
    whisper_model = stt.load_model()
    voices = tts.load_voices()
    router_llm = load_router_model()
    router_grammar = load_grammar()
    chat_llm = load_chat_model()
    tamil_chat_llm = load_tamil_chat_model()

    # No real ATV CAN bus exists yet - start_fake_ecu() stands in for one.
    # On real hardware this line goes away; start_listener() just points
    # at the real "socketcan" bus instead.
    telemetry_cache = TelemetryCache()
    start_fake_ecu()
    start_listener(telemetry_cache)

    print("Ready.")
    tts.speak(voices, STARTUP_GREETING)
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
                language = DEFAULT_LANGUAGE
                try:
                    query, language = stt.transcribe(whisper_model, audio)
                except sd.PortAudioError:
                    raise  # a dead audio device is fatal - let the outer handler stop the app
                except Exception as exc:
                    # A single bad turn (a model hiccup, a malformed
                    # decode) must not take down an app meant to run
                    # continuously in a moving vehicle - log it, go back
                    # to sleep, and let the next wake word start clean.
                    print(f"Error transcribing speech, recovering: {exc}")
                    break

                if not query:
                    if not heard_anything:
                        print("No speech detected, going back to sleep.")
                        tts.speak(voices, NO_SPEECH_RESPONSE)
                    else:
                        print("No follow-up, going back to sleep.")
                        tts.play_sleep_chime()
                    break

                print(f"Heard ({language}): {query!r}")
                heard_anything = True

                try:
                    response = answer_query(
                        router_llm,
                        router_grammar,
                        telemetry_cache,
                        query,
                        chat_llm=chat_llm,
                        tamil_chat_llm=tamil_chat_llm,
                        history=history,
                        language=language,
                    )
                    print(f"Responding: {response!r}")
                    tts.speak(voices, response, language)
                    history.append((query, response))
                except sd.PortAudioError:
                    raise
                except Exception as exc:
                    print(f"Error answering or speaking, recovering: {exc}")
                    try:
                        tts.speak(
                            voices,
                            UNEXPECTED_ERROR_RESPONSE.get(language, UNEXPECTED_ERROR_RESPONSE[DEFAULT_LANGUAGE]),
                            language,
                        )
                    except Exception:
                        pass  # even the apology failed - just go back to sleep quietly
                    break

                speech_timeout_ms = FOLLOWUP_SPEECH_TIMEOUT_MS
    except sd.PortAudioError as exc:
        print(f"Audio device error, stopping: {exc}")


if __name__ == "__main__":
    run()
