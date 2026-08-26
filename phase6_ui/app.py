"""
Phase 6 of 7 -- UI

A Gradio browser UI wrapping the Phase 5 pipeline for live demos without a
terminal or a working local mic setup. Records a question from the
browser's microphone, runs it through phase5_full_pipeline.process_audio_query
(ASR -> translate -> agent -> translate -> TTS), and displays every
intermediate stage plus a playable audio reply.

No pipeline logic lives here -- this file only wires Phase 5's existing
process_audio_query() to Gradio components.

Run with: python phase6_ui/app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import gradio as gr

from phase5_full_pipeline.main import process_audio_query

TITLE = "Multilingual Voice Assistant -- Indian Fintech Rules & Schemes"

DESCRIPTION = """
Ask a question by voice in **English, Hindi, Kannada, or Tamil** about RBI/SEBI
regulation, Union Budget content, or government schemes -- or ask about your
own Atal Pension Yojana eligibility. Answers are grounded in real government
PDFs (RAG) or a rules-based eligibility tool, then spoken back in your language.

Try asking:
- "What is Atal Pension Yojana?"
- "Am I eligible for APY if I'm 25 with a bank account and I don't pay tax?"
- "What is the Budget at a Glance for 2026-27?"
"""


def handle_query(audio_filepath: str | None):
    if audio_filepath is None:
        return "", "", "", "Please record a question first.", None

    result = process_audio_query(audio_filepath)
    detected_language_display = (
        f"{result['detected_language']} (confidence: {result['language_probability']:.2f})"
    )
    return (
        detected_language_display,
        result["transcript"],
        result["english_translation"],
        result["translated_answer"],
        result["output_audio_path"],
    )


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Ask your question",
    )
    submit_btn = gr.Button("Submit", variant="primary")

    detected_language_output = gr.Textbox(label="Detected language")
    transcript_output = gr.Textbox(label="Transcript")
    english_translation_output = gr.Textbox(label="English translation")
    answer_output = gr.Textbox(label="Answer (in your language)")
    answer_audio_output = gr.Audio(label="Spoken answer", type="filepath")

    submit_btn.click(
        fn=handle_query,
        inputs=[audio_input],
        outputs=[
            detected_language_output,
            transcript_output,
            english_translation_output,
            answer_output,
            answer_audio_output,
        ],
    )

if __name__ == "__main__":
    # share=False keeps this local-only. To get a temporary public link for
    # a remote demo, change this to share=True (creates a gradio.live URL
    # valid for 72 hours) -- don't do this by default since it exposes the
    # app (and your Groq API usage) to the public internet.
    demo.launch(share=False)
