from typing import Optional

# Checked before the RAG pipeline runs at all, so greetings/thanks/meta
# questions don't fall through to "I couldn't find that information in
# the provided documents" - that refusal is honest for real questions,
# but jarring for small talk that was never going to be in a document.

GREETINGS = ("hello", "hi there", " hi ", "hi ", "hey there", "good morning", "good afternoon", "good evening")
THANKS = ("thank", "appreciate it", "you're the best", "good job", "well done")
IDENTITY = ("who are you", "what are you")
CAPABILITY = ("what can you do", "what can you help", "what do you know", "how can you help")


def try_small_talk_answer(question: str) -> Optional[str]:
    text = f" {question.lower().strip()} "

    if any(phrase in text for phrase in THANKS):
        return "You're welcome!"

    if any(phrase in text for phrase in CAPABILITY):
        return (
            "I can tell you about your vehicle - battery, motor temperature, "
            "speed, tire pressure - and answer anything from the documents "
            "loaded on board."
        )

    if any(phrase in text for phrase in IDENTITY):
        return "I'm your resort ATV's voice assistant - your companion for the ride."

    if any(phrase in text for phrase in GREETINGS):
        return "Hey there! What can I help you with?"

    return None
