"""
Phase 2 synthetic test harness -- validates ASR + language detection
without a live microphone or a human speaker of Hindi/Kannada/Tamil.

For each test phrase, gTTS synthesizes a "known good" spoken version of a
sentence we already know the text and language of. That audio is fed
straight into transcribe_auto() from phase2_multilingual/main.py (the same
function the live mic loop uses) to check whether Whisper both detects the
right language and recovers a reasonable transcription.

No mp3 -> wav conversion is needed: faster-whisper decodes audio itself via
PyAV (which bundles its own FFmpeg libraries), so it accepts the gTTS mp3
directly -- no system ffmpeg/pydub dependency required.

Run with: python phase2_multilingual/test_synthetic.py
"""

import os
import sys
import tempfile

from gtts import gTTS

# Windows consoles default to a codepage (e.g. cp1252) that can't encode
# Devanagari/Kannada/Tamil script; force UTF-8 so printing transcripts
# doesn't crash.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from main import GTTS_LANGUAGE_MAP, MODEL_SIZE, transcribe_auto
from faster_whisper import WhisperModel

TEST_PHRASES = {
    "hi": "नमस्ते, आप कैसे हैं?",
    "kn": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
    "ta": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
}


def synthesize(text: str, language_code: str) -> str:
    """Synthesize text to a temp mp3 using gTTS and return the file path."""
    gtts_lang = GTTS_LANGUAGE_MAP.get(language_code, "en")
    fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    gTTS(text=text, lang=gtts_lang).save(mp3_path)
    return mp3_path


if __name__ == "__main__":
    print("Loading Whisper model...")
    whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    results = []
    for expected_language, original_text in TEST_PHRASES.items():
        print(f"\nSynthesizing test audio for '{expected_language}'...")
        mp3_path = synthesize(original_text, expected_language)
        try:
            transcript, detected_language, confidence = transcribe_auto(mp3_path, whisper_model)
        finally:
            os.remove(mp3_path)

        results.append(
            {
                "expected_language": expected_language,
                "original_text": original_text,
                "detected_language": detected_language,
                "confidence": confidence,
                "transcript": transcript,
                "match": detected_language == expected_language,
            }
        )

    print("\n" + "=" * 100)
    header = f"{'Expected':<10}{'Detected':<12}{'Confidence':<12}{'Match':<8}{'Original Text':<28}Transcribed Text"
    print(header)
    print("-" * 100)
    for r in results:
        print(
            f"{r['expected_language']:<10}"
            f"{r['detected_language']:<12}"
            f"{r['confidence']:<12.2f}"
            f"{'YES' if r['match'] else 'NO':<8}"
            f"{r['original_text']:<28} "
            f"{r['transcript']}"
        )
    print("=" * 100)

    passed = sum(1 for r in results if r["match"])
    print(f"\n{passed}/{len(results)} language detections matched expected.")
