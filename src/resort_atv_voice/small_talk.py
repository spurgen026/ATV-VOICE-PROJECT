
from .config import DEFAULT_LANGUAGE

# Checked before the RAG pipeline runs at all, so greetings/thanks/meta
# questions don't fall through to "I couldn't find that information in
# the provided documents" - that refusal is honest for real questions,
# but jarring for small talk that was never going to be in a document.
#
# Hindi/Tamil phrase and response sets: hand-written, not verified by a
# native speaker - same caveat as number_words.py. Matching is keyword-only
# per detected language, so code-switched or unexpected phrasing can still
# miss and fall through to the router/chat path, same as English always
# could.
#
# Responses rewritten to colloquial/spoken register 2026-08-08 (were
# formal/written Tamil and Hindi before - see config.py's
# TELEMETRY_FIELD_PHRASES comment for the reasoning and same not-yet-
# native-verified caveat). Kept the respectful आप/உங்கள் address form (a
# resort guest, not a close friend) throughout - colloquial vocabulary and
# sentence structure, not informal-address.

GREETINGS = {
    "en": ("hello", "hi there", " hi ", "hi ", "hey there", "good morning", "good afternoon", "good evening"),
    "hi": ("नमस्ते", "नमस्कार"),
    "ta": ("வணக்கம்",),
}
THANKS = {
    "en": ("thank", "appreciate it", "you're the best", "good job", "well done"),
    "hi": ("धन्यवाद", "शुक्रिया"),
    "ta": ("நன்றி",),
}
IDENTITY = {
    "en": ("who are you", "what are you"),
    "hi": ("आप कौन हैं", "तुम कौन हो"),
    "ta": ("நீ யார்", "நீங்கள் யார்"),
}
CAPABILITY = {
    "en": ("what can you do", "what can you help", "what do you know", "how can you help"),
    "hi": ("क्या कर सकते हो", "क्या मदद कर सकते हो"),
    "ta": ("என்ன செய்ய முடியும்", "எப்படி உதவ முடியும்"),
}

THANKS_RESPONSE = {
    "en": "You're welcome!",
    "hi": "कोई बात नहीं!",
    "ta": "பரவாயில்ல!",
}
CAPABILITY_RESPONSE = {
    "en": (
        "I can tell you about your vehicle - battery, motor temperature, "
        "speed, tire pressure - and chat about anything else, all "
        "without needing an internet connection."
    ),
    "hi": (
        "मैं आपकी गाड़ी के बारे में बता सकता हूं - बैटरी, मोटर टेम्परेचर, "
        "स्पीड, टायर प्रेशर - और बिना इंटरनेट के और भी बातें कर सकता हूं।"
    ),
    "ta": (
        "உங்க வண்டியைப் பத்தி நான் சொல்லுவேன் - பேட்டரி, மோட்டர் "
        "டெம்பரேச்சர், ஸ்பீட், டயர் பிரஷர் - இணையம் இல்லாமயே வேற "
        "எதைப் பத்தியும் பேசலாம்."
    ),
}
IDENTITY_RESPONSE = {
    "en": "I'm your resort ATV's voice assistant - your companion for the ride.",
    "hi": "मैं आपकी रिसॉर्ट एटीवी हूं - आपकी सवारी का साथी!",
    "ta": "நான் உங்க ரிசார்ட் ஏடிவி - உங்க பயணத்துல தோழன்!",
}
GREETING_RESPONSE = {
    "en": "Hey there! What can I help you with?",
    "hi": "नमस्ते! बताइए, क्या मदद चाहिए?",
    "ta": "வணக்கம்! சொல்லுங்க, என்ன வேணும்?",
}


def _matches(text: str, language: str, table: dict) -> bool:
    phrases = table.get(language, table[DEFAULT_LANGUAGE])
    return any(phrase in text for phrase in phrases)


def try_small_talk_answer(question: str, language: str = DEFAULT_LANGUAGE) -> str | None:
    text = f" {question.lower().strip()} "

    if _matches(text, language, THANKS):
        return THANKS_RESPONSE.get(language, THANKS_RESPONSE[DEFAULT_LANGUAGE])

    if _matches(text, language, CAPABILITY):
        return CAPABILITY_RESPONSE.get(language, CAPABILITY_RESPONSE[DEFAULT_LANGUAGE])

    if _matches(text, language, IDENTITY):
        return IDENTITY_RESPONSE.get(language, IDENTITY_RESPONSE[DEFAULT_LANGUAGE])

    if _matches(text, language, GREETINGS):
        return GREETING_RESPONSE.get(language, GREETING_RESPONSE[DEFAULT_LANGUAGE])

    return None
