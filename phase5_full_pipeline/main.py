"""
Phase 5 of 7 -- Full pipeline

Chains every phase built so far into one voice-in, voice-out loop:

  audio -> Whisper auto-detect + transcribe (phase2) -> translate to English
  (phase2) -> Phase 4 agent, which internally routes to RAG or the
  eligibility tool (phase4) -> translate back to the detected language
  (phase2) -> gTTS speech out (phase2)

This file only sequences existing functions -- no speech, translation,
RAG, or agent logic is duplicated here.

process_audio_query(audio_filepath) is the reusable core: given a path to
an existing audio file, it runs the full pipeline and returns a dict of
every intermediate result, so Phase 6's UI (or anything else) can call it
without needing a live mic. The CLI's run_pipeline() records from the mic
and is a thin wrapper around that same function.

If Whisper detects a language outside the four this project supports
(English, Hindi, Kannada, Tamil), the reply falls back to English rather
than silently failing to translate/speak it.

Run with: python phase5_full_pipeline/main.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

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
    synthesize_speech,
    to_english,
    transcribe_auto,
)
from phase4_agent.agent import answer_question

SUPPORTED_LANGUAGES = set(GTTS_LANGUAGE_MAP)  # {"en", "hi", "kn", "ta"}

_WHISPER_MODEL: Optional[WhisperModel] = None


def _get_whisper_model() -> WhisperModel:
    """Lazily load and cache the Whisper model, shared across calls."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        print("Loading Whisper model...")
        _WHISPER_MODEL = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _WHISPER_MODEL


def process_audio_query(audio_filepath: str) -> dict:
    """Run the full pipeline on an existing audio file (no live recording).

    Returns a dict with detected_language, transcript, english_translation,
    answer_text (the agent's English answer), translated_answer (in the
    reply language), and output_audio_path (spoken reply, or None if
    nothing was transcribed). Also includes language_probability for
    callers that want Whisper's detection confidence.
    """
    whisper_model = _get_whisper_model()
    transcript, detected_language, language_probability = transcribe_auto(
        audio_filepath, whisper_model
    )

    result = {
        "detected_language": detected_language,
        "language_probability": language_probability,
        "transcript": transcript,
        "english_translation": "",
        "answer_text": "",
        "translated_answer": "",
        "output_audio_path": None,
    }

    if not transcript:
        no_speech_message = "Didn't catch that -- try again."
        result["english_translation"] = no_speech_message
        result["answer_text"] = no_speech_message
        result["translated_answer"] = no_speech_message
        return result

    reply_language = detected_language if detected_language in SUPPORTED_LANGUAGES else "en"

    english_text = to_english(transcript, detected_language)
    result["english_translation"] = english_text

    english_answer = answer_question(english_text)
    result["answer_text"] = english_answer

    translated_answer = from_english(english_answer, reply_language)
    result["translated_answer"] = translated_answer

    result["output_audio_path"] = synthesize_speech(translated_answer, reply_language)
    return result


def run_pipeline() -> None:
    _get_whisper_model()  # load up front so there's no lag right after recording

    wav_path = record_audio()
    result = process_audio_query(wav_path)
    os.remove(wav_path)

    print(
        f"Detected language: {result['detected_language']} "
        f"(confidence: {result['language_probability']:.2f})"
    )
    print(f"Transcribed text: {result['transcript']}")

    if not result["transcript"]:
        print("Didn't catch that -- try again.")
        return

    if result["detected_language"] not in SUPPORTED_LANGUAGES:
        print(
            f"Detected language '{result['detected_language']}' isn't one of the supported "
            f"languages ({', '.join(sorted(SUPPORTED_LANGUAGES))}) -- "
            "falling back to English for the reply."
        )

    print(f"English translation: {result['english_translation']}")
    print(f"Agent answer (English): {result['answer_text']}")
    print(f"Translated answer: {result['translated_answer']}")

    output_audio_path = result["output_audio_path"]
    if output_audio_path:
        try:
            from playsound import playsound

            playsound(output_audio_path)
            os.remove(output_audio_path)
        except Exception as e:
            print(f"Couldn't play audio automatically ({e}). Reply saved at: {output_audio_path}")


if __name__ == "__main__":
    run_pipeline()
