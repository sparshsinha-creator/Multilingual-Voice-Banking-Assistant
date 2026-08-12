"""
Phase 2 of 7 -- Multilingual Speech Loop

Extends Phase 1's speech loop (mic -> Whisper -> gTTS -> speaker) to handle
Hindi, Kannada, Tamil, and English instead of English only. On top of
Phase 1, this phase adds:
  - Auto language detection: Whisper is no longer forced to language="en",
    so it detects the spoken language itself (plus a confidence score).
  - Translation to English via deep-translator's GoogleTranslator, since
    English is the internal processing language later phases (RAG/agent
    logic) will operate on.
  - Translation back from English into the detected language before
    speaking the reply, with a small map from Whisper's language codes to
    the gTTS language codes needed for playback.

There is still no real answer logic here -- Phase 2 just echoes back what
was said, translated round-trip, to prove the multilingual audio pipeline
works before Phase 3+ add scheme knowledge and eligibility reasoning.

Run with: python phase2_multilingual/main.py
"""

import os
import sys
import tempfile

import speech_recognition as sr

# Windows consoles default to a codepage (e.g. cp1252) that can't encode
# Devanagari/Kannada/Tamil script; force UTF-8 so printing transcripts
# doesn't crash.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
from gtts import gTTS

MODEL_SIZE = "small"
RECORD_SECONDS = 5

# Whisper language code -> gTTS language code
GTTS_LANGUAGE_MAP = {
    "en": "en",
    "hi": "hi",
    "kn": "kn",
    "ta": "ta",
}


def record_audio() -> str:
    """Record ~5 seconds of audio from the default mic and save it to a temp wav file."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print(f"Listening for {RECORD_SECONDS} seconds -- speak now...")
        audio = recognizer.record(source, duration=RECORD_SECONDS)

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with open(wav_path, "wb") as f:
        f.write(audio.get_wav_data())
    return wav_path


def transcribe_auto(wav_path: str, model: WhisperModel) -> tuple[str, str, float]:
    """Transcribe a wav file letting Whisper auto-detect the language.

    Returns (text, detected_language, language_probability).
    """
    segments, info = model.transcribe(wav_path)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, info.language, info.language_probability


def to_english(text: str, source_language: str) -> str:
    """Translate text into English using deep-translator, unless it's already English."""
    if source_language == "en":
        return text
    try:
        return GoogleTranslator(source=source_language, target="en").translate(text)
    except Exception as e:
        print(f"Translation to English failed ({e}). Using original text instead.")
        return text


def from_english(text: str, target_language: str) -> str:
    """Translate English text into the target language using deep-translator."""
    if target_language == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_language).translate(text)
    except Exception as e:
        print(f"Translation to '{target_language}' failed ({e}). Reply will stay in English.")
        return text


def speak(text: str, language_code: str) -> None:
    """Speak text aloud with gTTS in the given language, falling back to printing the file path."""
    gtts_lang = GTTS_LANGUAGE_MAP.get(language_code, "en")

    fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    gTTS(text=text, lang=gtts_lang).save(mp3_path)

    try:
        from playsound import playsound

        playsound(mp3_path)
        os.remove(mp3_path)
    except Exception as e:
        print(f"Couldn't play audio automatically ({e}). Reply saved at: {mp3_path}")


if __name__ == "__main__":
    print("Loading Whisper model...")
    whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    wav_path = record_audio()
    transcript, detected_language, language_probability = transcribe_auto(wav_path, whisper_model)
    os.remove(wav_path)

    print(f"Detected language: {detected_language} (confidence: {language_probability:.2f})")
    print(f"Transcribed text: {transcript}")

    if not transcript:
        print("Didn't catch that -- try again.")
    else:
        english_text = to_english(transcript, detected_language)
        print(f"English translation: {english_text}")

        english_reply = f"You said: {english_text}. This is Phase 2, echoing back only."
        reply = from_english(english_reply, detected_language)
        print(f"Reply ({detected_language}): {reply}")

        speak(reply, detected_language)
