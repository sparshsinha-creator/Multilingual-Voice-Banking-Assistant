# Multilingual Voice Assistant for Indian Fintech Regulation & Government Schemes

A voice-in, voice-out assistant that answers questions about Indian fintech
regulation and government financial schemes in **Hindi, Kannada, Tamil, or
English** — spoken in, spoken back, in the same language.

Ask about RBI rules (digital lending, UPI limits, KYC/AML, Account
Aggregator, Payment Aggregator licensing, NBFC scale-based regulation,
co-lending, e-Rupee/CBDC), FDI policy for insurance/NBFC/fintech, SEBI
robo-advisory licensing, IRDAI's insurtech sandbox, PFRDA/Ministry of Finance
schemes (Atal Pension Yojana, PMJDY, PMJJBY, PMSBY, Sovereign Gold Bonds), or
cross-cutting topics (DPDP Act 2023, crypto/VDA taxation, ONDC). Answers are
grounded in a 27-item RAG FAQ dataset. One question type — Atal Pension
Yojana eligibility — is handled by a real agentic tool instead of retrieval.

## Why this project

This is a portfolio/interview build, not a product. It's designed to
demonstrate the same core capability stack as **Sarvam AI**'s products —
multilingual ASR, multilingual TTS, RAG, and agentic orchestration over
Indian-language voice input — without copying any specific Sarvam product.
Concretely, it maps to:

| This project uses | Sarvam's equivalent product | Swap-in point |
|---|---|---|
| faster-whisper (ASR) | **Saaras** (speech-to-text) | `phase1`/`phase5` ASR call |
| gTTS (TTS) | **Bulbul** (text-to-speech) | `phase1`/`phase5` TTS call |
| deep-translator | **Sarvam-Translate** | `phase2` translation step |
| Groq-hosted LLM | **Sarvam-M / Sarvam 30B** | `phase3`/`phase4` LLM call |

The build is staged in 7 phases so each capability (speech round-trip →
multilingual → RAG → agentic tool routing → full pipeline → UI → landing
page) can be run and demoed independently, mirroring how you'd actually
build this kind of system incrementally.

## One-time setup (VS Code)

1. **Clone/open the repo in VS Code**, then open an integrated terminal.

2. **Create and activate a virtual environment:**

   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

   (macOS/Linux: `python3 -m venv venv && source venv/bin/activate`)

3. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

   **PyAudio troubleshooting:** `PyAudio` (used by `SpeechRecognition` for
   mic input) depends on the `portaudio` C library and often fails to build
   from a plain `pip install` on Windows/Linux.

   - **Windows:** if `pip install pyaudio` fails, install the prebuilt wheel
     instead: `pip install pipwin && pipwin install pyaudio`.
   - **macOS:** `brew install portaudio` before `pip install pyaudio`.
   - **Linux (Debian/Ubuntu):** `sudo apt-get install portaudio19-dev` before
     `pip install pyaudio`.

4. **Set up your API key:**

   ```powershell
   copy .env.example .env
   ```

   Edit `.env` and add a free [Groq API key](https://console.groq.com/keys)
   (Groq's free tier is enough to run every phase in this repo):

   ```
   GROQ_API_KEY=your_key_here
   ```

5. Select the `venv` interpreter in VS Code (`Ctrl+Shift+P` →
   *Python: Select Interpreter*) so the terminal and any run/debug configs
   use it by default.

## Phases

Each phase is self-contained under its own folder and can be run without the
later phases existing. Run them from the project root with the venv active.

### Phase 1 — Speech loop (`phase1_speech_loop/`)

Bare-bones English-only round trip: mic → Whisper ASR → gTTS → speaker.
Proves the audio pipeline works before adding any language or intelligence
complexity.

```powershell
python phase1_speech_loop/main.py
```

### Phase 2 — Multilingual (`phase2_multilingual/`)

Adds language auto-detection and translation (`deep-translator`) so the same
loop round-trips correctly in Hindi, Kannada, Tamil, and English.

```powershell
python phase2_multilingual/main.py
```

### Phase 3 — RAG (`phase3_rag/`)

Builds a FAISS index over `data/fintech_schemes_faq.json` using
`sentence-transformers`, then retrieves top chunks and calls a Groq LLM for
a grounded answer.

```powershell
python phase3_rag/build_index.py
python phase3_rag/query.py
```

### Phase 4 — Agent (`phase4_agent/`)

A LangGraph agent (`agent.py`) that routes each question to either the
Phase 3 RAG pipeline (general FAQ) or a rules-engine tool (`tools.py`) that
checks Atal Pension Yojana eligibility (age 18–40, bank account required,
excludes income-tax payers enrolling after 30 Sept 2022).

```powershell
python phase4_agent/agent.py
```

### Phase 5 — Full pipeline (`phase5_full_pipeline/`)

Combines Phases 1–4 into one voice-in, voice-out flow: speak a question in
any supported language, get a spoken, grounded answer back in that language.

```powershell
python phase5_full_pipeline/main.py
```

### Phase 6 — UI (`phase6_ui/`)

A Gradio browser UI wrapping Phase 5, for live demos without a terminal.

```powershell
python phase6_ui/app.py
```

### Phase 7 — Landing page (`phase7_landing_page/`)

A React + Vite + Tailwind landing page (hero, the 5-phase pipeline,
features grid, tech stack) for a LinkedIn post or portfolio link. Not part
of the voice pipeline — pure presentation layer.

```powershell
cd phase7_landing_page
npm install
npm run dev
```

## Tech stack

- **Speech:** Whisper via `faster-whisper`, `SpeechRecognition`, `gTTS`
- **Translation:** `deep-translator`
- **RAG:** `sentence-transformers`, `FAISS`
- **LLM:** Groq-hosted model
- **Agentic orchestration:** LangGraph
- **Backend structure:** FastAPI-style Python modules
- **Demo UI:** Gradio
- **Landing page:** React, Vite, Tailwind CSS, `lucide-react`

## What's simplified for this build

This is a demo, not a production system. Deliberately out of scope:

- **No auth** — no login, no API keys per user, no rate limiting.
- **No multi-user sessions** — single in-process session, no session store.
- **No production deployment** — runs locally; no containerization, no
  cloud hosting, no CI/CD.
- **Not full 22-language coverage** — only Hindi, Kannada, Tamil, and
  English, chosen to demonstrate multilingual capability without building
  out every scheduled Indian language.
- **Eligibility tool uses fixed demo answers** — the Atal Pension Yojana
  checker in `phase4_agent/tools.py` applies the rules to values you give it
  in a single turn; it does not do multi-turn slot-filling (asking follow-up
  questions if age or bank-account status is missing).

## Known limitations

- **Code-switched proper nouns can get mangled.** When an English proper
  noun (e.g. "Reserve Bank of India") is spoken inside an otherwise Hindi/
  Kannada/Tamil sentence, Whisper sometimes phonetically transliterates it
  into that language's script instead of keeping it in Latin script, and
  the translation step then translates that phonetic spelling literally
  instead of recognizing the original English term. This is a known
  limitation of combining Whisper with dictionary-style machine
  translation on mixed-language (code-switched) speech, not a pipeline
  bug — retrieval and the agent still behave correctly (they either match
  real content or honestly say they don't have it) given whatever text
  they're handed.

## Data accuracy disclaimer

The FAQ dataset (`data/fintech_schemes_faq.json`) and the Atal Pension
Yojana eligibility rules in this repo are **demo/informational content
only**. They are **not regulatory or financial advice** and may not reflect
the latest rule changes. Before relying on any of this for a real decision,
verify against the official sources:

- [RBI](https://www.rbi.org.in/) — banking, NBFC, digital lending, UPI, KYC/AML, Account Aggregator, Payment Aggregator, co-lending, e-Rupee
- [SEBI](https://www.sebi.gov.in/) — robo-advisory / investment adviser regulation
- [IRDAI](https://irdai.gov.in/) — insurtech, insurance sandbox
- [PFRDA](https://www.pfrda.org.in/) — Atal Pension Yojana and pension scheme rules

## Swapping in Sarvam's APIs

The LangGraph structure in `phase4_agent/agent.py` is intentionally
provider-agnostic — nodes call out to speech/translation/LLM functions, not
inline SDK code — so swapping providers means changing the implementation
behind those calls, not the graph itself:

| Step | Current | Replace with |
|---|---|---|
| ASR (`phase1`/`phase5`) | faster-whisper | Sarvam **Saaras** API |
| TTS (`phase1`/`phase5`) | gTTS | Sarvam **Bulbul** API |
| Translation (`phase2`) | deep-translator | Sarvam **Sarvam-Translate** API |
| LLM (`phase3`/`phase4`) | Groq | **Sarvam-M** or **Sarvam 30B** |

Each swap is a drop-in replacement of the function body that makes the
external call (same input/output shape: audio in → text out for ASR, text
in → audio out for TTS, text in → text out for translation/LLM) — the
LangGraph routing between the RAG node and the eligibility-tool node
doesn't need to change at all.
