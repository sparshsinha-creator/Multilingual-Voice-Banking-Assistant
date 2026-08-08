"""
Phase 1 of 7 -- Speech Loop (English only)

This is the simplest possible version of the voice assistant: mic ->
Whisper speech-to-text -> gTTS text-to-speech -> speaker. No translation,
no RAG, no agent -- just an echo. Running this successfully proves the
core audio round trip (recording, transcription, and playback) works
before Phase 2 adds multilingual support and later phases add the
RAG/agent intelligence on top.

Run with: python phase1_speech_loop/main.py
"""

import os
import tempfile

import speech_recognition as sr
from faster_whisper import WhisperModel
from gtts import gTTS

MODEL_SIZE = "small"
RECORD_SECONDS = 5


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


def transcribe(wav_path: str, model: WhisperModel) -> tuple[str, str]:
    """Transcribe a wav file with Whisper (English) and return (text, detected_language)."""
    segments, info = model.transcribe(wav_path, language="en")
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, info.language


def speak(text: str) -> None:
    """Speak text aloud with gTTS, falling back to printing the file path if playback fails."""
    fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    gTTS(text=text, lang="en").save(mp3_path)

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
    transcript, detected_language = transcribe(wav_path, whisper_model)
    os.remove(wav_path)

    print(f"Detected language: {detected_language}")
    print(f"Transcribed text: {transcript}")

    if not transcript:
        print("Didn't catch that -- try again.")
    else:
        reply = f"You said: {transcript}. This is Phase 1, echoing back only."
        print(f"Reply: {reply}")
        speak(reply)
