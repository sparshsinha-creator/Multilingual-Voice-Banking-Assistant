"""
Phase 2 of 7 -- Multilingual Speech Loop

Extends Phase 1's speech loop (mic -> Whisper -> gTTS -> speaker) to handle
Hindi, Kannada, Tamil, and English instead of English only. On top of
Phase 1, this phase adds:
  - Auto language detection: Whisper is no longer forced to language="en",
    so it detects the spoken language itself (plus a confidence score).
  - Translation to English via deep-translator's MyMemoryTranslator, since
    English is the internal processing language later phases (RAG/agent
    logic) will operate on. Long text is chunked under MyMemory's 500-char
    request limit and translated piece by piece.
  - Translation back from English into the detected language before
    speaking the reply, with a small map from Whisper's language codes to
    the gTTS language codes needed for playback.

There is still no real answer logic here -- Phase 2 just echoes back what
was said, translated round-trip, to prove the multilingual audio pipeline
works before Phase 3+ add scheme knowledge and eligibility reasoning.

Run with: python phase2_multilingual/main.py
"""

import os
import re
import sys
import tempfile

import speech_recognition as sr

# Windows consoles default to a codepage (e.g. cp1252) that can't encode
# Devanagari/Kannada/Tamil script; force UTF-8 so printing transcripts
# doesn't crash.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from deep_translator import MyMemoryTranslator
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

# Whisper language code -> MyMemoryTranslator locale code. GoogleTranslator's
# free scraper backend became unreliable (frequently returns "No translation
# was found"), so translation runs on MyMemory instead, which needs full
# locale codes rather than Whisper's bare ISO codes.
MYMEMORY_LANGUAGE_MAP = {
    "en": "en-GB",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "ta": "ta-IN",
}

# MyMemory's free tier rejects any single request over 500 characters, but
# RAG answers routinely run well past that -- so long text is split into
# chunks under this limit (leaving headroom) and translated piece by piece.
MYMEMORY_MAX_CHARS = 450


def _split_into_chunks(text: str, max_chars: int = MYMEMORY_MAX_CHARS) -> list[str]:
    """Split text into <= max_chars pieces, breaking on sentence boundaries
    where possible so translation isn't cut off mid-sentence."""
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    chunks = []
    current = ""
    for sentence in sentences:
        # A single sentence longer than the limit still needs a hard split.
        while len(sentence) > max_chars:
            head, sentence = sentence[:max_chars], sentence[max_chars:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _translate(text: str, source_locale: str, target_locale: str) -> str:
    """Translate text via MyMemory, chunking first if it exceeds the API's
    500-character request limit."""
    if len(text) <= MYMEMORY_MAX_CHARS:
        return MyMemoryTranslator(source=source_locale, target=target_locale).translate(text)
    translator = MyMemoryTranslator(source=source_locale, target=target_locale)
    return " ".join(translator.translate(chunk) for chunk in _split_into_chunks(text))


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
    # Without VAD, Whisper hallucinates text (e.g. "you", stray foreign-script
    # fragments) over silence/background noise instead of returning nothing --
    # vad_filter drops non-speech segments before transcription runs on them.
    try:
        segments, info = model.transcribe(wav_path, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, info.language, info.language_probability
    except ValueError:
        # VAD found no speech at all, leaving nothing for language detection
        # to run on -- treat that the same as an empty transcript.
        return "", "en", 0.0


def to_english(text: str, source_language: str) -> str:
    """Translate text into English using deep-translator, unless it's already English."""
    if source_language == "en":
        return text
    source_locale = MYMEMORY_LANGUAGE_MAP.get(source_language)
    if source_locale is None:
        print(f"Unrecognized source language '{source_language}'. Using original text instead.")
        return text
    try:
        return _translate(text, source_locale, "en-GB")
    except Exception as e:
        print(f"Translation to English failed ({e}). Using original text instead.")
        return text


def from_english(text: str, target_language: str) -> str:
    """Translate English text into the target language using deep-translator."""
    if target_language == "en":
        return text
    target_locale = MYMEMORY_LANGUAGE_MAP.get(target_language)
    if target_locale is None:
        print(f"Unrecognized target language '{target_language}'. Reply will stay in English.")
        return text
    try:
        return _translate(text, "en-GB", target_locale)
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
