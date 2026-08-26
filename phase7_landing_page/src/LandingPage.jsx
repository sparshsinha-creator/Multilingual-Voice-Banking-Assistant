import {
  Mic,
  Languages,
  FileSearch,
  Workflow,
  Waypoints,
  LayoutDashboard,
  AudioLines,
  ShieldCheck,
  Route,
  Globe2,
  FolderGit2,
  ExternalLink,
  Sparkles,
} from "lucide-react";

const PIPELINE_PHASES = [
  {
    icon: Mic,
    title: "Speech Loop Foundation",
    description: "Mic in, Whisper transcribes, gTTS speaks the reply back out.",
  },
  {
    icon: Languages,
    title: "Multilingual Understanding",
    description: "Auto-detects English, Hindi, Kannada, or Tamil and translates to English.",
  },
  {
    icon: FileSearch,
    title: "Document-Grounded RAG",
    description: "FAISS retrieval over real RBI, SEBI, and Union Budget PDFs, not a canned FAQ.",
  },
  {
    icon: Workflow,
    title: "Agentic Eligibility Tool-Calling",
    description: "A LangGraph agent routes to RAG or a rules-based Atal Pension Yojana checker.",
  },
  {
    icon: Waypoints,
    title: "Full Multilingual Pipeline",
    description: "Chains speech, translation, retrieval, and the agent into one voice-in, voice-out loop.",
  },
  {
    icon: LayoutDashboard,
    title: "Gradio Demo UI",
    description: "A browser UI wrapping the full pipeline for live, no-terminal demos.",
  },
];

const FEATURES = [
  {
    icon: AudioLines,
    title: "Multilingual voice in/out",
    description: "Ask a question out loud and hear the answer spoken back in the same language.",
  },
  {
    icon: ShieldCheck,
    title: "Real-document-grounded answers",
    description: "Every answer is retrieved from actual regulatory PDFs, not hallucinated by the LLM.",
  },
  {
    icon: Route,
    title: "Automatic FAQ-vs-eligibility routing",
    description: "The agent decides on its own whether to retrieve a fact or run an eligibility check.",
  },
  {
    icon: Globe2,
    title: "Four Indian languages",
    description: "Works end to end in English, Hindi, Kannada, and Tamil.",
  },
];

const TECH_STACK = [
  "Whisper",
  "gTTS",
  "deep-translator",
  "FAISS + sentence-transformers",
  "Groq (Llama 3.1)",
  "LangGraph",
  "Gradio",
  "React + Tailwind",
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Hero */}
      <header className="mx-auto max-w-5xl px-6 pt-24 pb-20 text-center">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-gray-800 bg-gray-900 px-4 py-1.5 text-sm text-gray-400">
          <Sparkles className="h-4 w-4 text-indigo-400" />
          Voice-first fintech assistant
        </div>
        <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
          Multilingual Voice Assistant
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-lg text-gray-300">
          A multilingual voice assistant that answers questions about Indian fintech
          rules and government schemes — in English, Hindi, Kannada, and Tamil.
        </p>
        <p className="mx-auto mt-4 max-w-xl text-sm text-gray-500">
          Answers are grounded in real RBI, SEBI, and Union Budget documents — not
          guesswork from an LLM.
        </p>
      </header>

      {/* Pipeline */}
      <section className="mx-auto max-w-5xl px-6 py-16">
        <h2 className="text-center text-2xl font-semibold text-white">
          Six phases, built incrementally
        </h2>
        <p className="mx-auto mt-2 max-w-xl text-center text-sm text-gray-500">
          Each phase is a working, independently runnable step toward the full pipeline.
        </p>
        <ol className="mt-12 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {PIPELINE_PHASES.map((phase, index) => {
            const Icon = phase.icon;
            return (
              <li
                key={phase.title}
                className="relative rounded-xl border border-gray-800 bg-gray-900 p-6"
              >
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-indigo-500/10 text-indigo-400">
                    <Icon className="h-5 w-5" />
                  </div>
                  <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                    Phase {index + 1}
                  </span>
                </div>
                <h3 className="mt-4 font-medium text-white">{phase.title}</h3>
                <p className="mt-2 text-sm text-gray-400">{phase.description}</p>
              </li>
            );
          })}
        </ol>
      </section>

      {/* Features */}
      <section className="mx-auto max-w-5xl px-6 py-16">
        <h2 className="text-center text-2xl font-semibold text-white">What it does</h2>
        <div className="mt-12 grid grid-cols-1 gap-6 sm:grid-cols-2">
          {FEATURES.map((feature) => {
            const Icon = feature.icon;
            return (
              <div
                key={feature.title}
                className="flex gap-4 rounded-xl border border-gray-800 bg-gray-900 p-6"
              >
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-400">
                  <Icon className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="font-medium text-white">{feature.title}</h3>
                  <p className="mt-1 text-sm text-gray-400">{feature.description}</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Tech stack */}
      <section className="mx-auto max-w-5xl px-6 py-16">
        <h2 className="text-center text-2xl font-semibold text-white">Tech stack</h2>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          {TECH_STACK.map((tech) => (
            <span
              key={tech}
              className="rounded-full border border-gray-800 bg-gray-900 px-4 py-2 text-sm text-gray-300"
            >
              {tech}
            </span>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800">
        <div className="mx-auto flex max-w-5xl flex-col items-center gap-4 px-6 py-12 text-center">
          <a
            href="#"
            className="inline-flex items-center gap-2 rounded-lg border border-gray-800 bg-gray-900 px-4 py-2 text-sm text-gray-300 transition hover:border-gray-700 hover:text-white"
          >
            <FolderGit2 className="h-4 w-4" />
            View on GitHub
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
          <p className="text-sm text-gray-500">Built for [interview context].</p>
        </div>
      </footer>
    </div>
  );
}
