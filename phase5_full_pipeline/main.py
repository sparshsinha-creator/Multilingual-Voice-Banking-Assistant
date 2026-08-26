"""
Phase 5 of 7 -- Full pipeline

Chains every phase built so far into one voice-in, voice-out loop:

  mic -> Whisper auto-detect + transcribe (phase2) -> translate to English
  (phase2) -> Phase 4 agent, which internally routes to RAG or the
  eligibility tool (phase4) -> translate back to the detected language
  (phase2) -> gTTS speech out (phase2)

This file only sequences existing functions -- no speech, translation,
RAG, or agent logic is duplicated here.

If Whisper detects a language outside the four this project supports
(English, Hindi, Kannada, Tamil), the reply falls back to English rather
than silently failing to translate/speak it.

Run with: python phase5_full_pipeline/main.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from faster_whisper import WhisperModel

from phase2_multilingual.main import (
    GTTS_LANGUAGE_MAP,
    MODEL_SIZE,
    from_english,
    record_audio,
    speak,
    to_english,
    transcribe_auto,
)
from phase4_agent.agent import answer_question

SUPPORTED_LANGUAGES = set(GTTS_LANGUAGE_MAP)  # {"en", "hi", "kn", "ta"}


def run_pipeline(whisper_model: WhisperModel) -> None:
    wav_path = record_audio()
    transcript, detected_language, language_probability = transcribe_auto(wav_path, whisper_model)
    os.remove(wav_path)

    print(f"Detected language: {detected_language} (confidence: {language_probability:.2f})")
    print(f"Transcribed text: {transcript}")

    if not transcript:
        print("Didn't catch that -- try again.")
        return

    reply_language = detected_language
    if detected_language not in SUPPORTED_LANGUAGES:
        print(
            f"Detected language '{detected_language}' isn't one of the supported "
            f"languages ({', '.join(sorted(SUPPORTED_LANGUAGES))}) -- "
            "falling back to English for the reply."
        )
        reply_language = "en"

    english_text = to_english(transcript, detected_language)
    print(f"English translation: {english_text}")

    print("Thinking...")
    english_answer = answer_question(english_text)
    print(f"Agent answer (English): {english_answer}")

    translated_answer = from_english(english_answer, reply_language)
    print(f"Translated answer ({reply_language}): {translated_answer}")

    speak(translated_answer, reply_language)


if __name__ == "__main__":
    print("Loading Whisper model...")
    whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    run_pipeline(whisper_model)
