"""
Phase 3 of 7 -- RAG query

Loads the FAISS index + docs.json built by build_index.py (chunks from the
real RBI, SEBI, Union Budget, and Government Schemes PDFs in
phase3_rag_datasource/), retrieves the top-k chunks for a question, and
asks a Groq-hosted LLM to answer using only that retrieved context --
citing which source document(s) it drew from.

Run with: python phase3_rag/query.py "your question here"
"""

import json
import os
import sys
from pathlib import Path

import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INDEX_DIR = Path(__file__).resolve().parent / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
DOCS_PATH = INDEX_DIR / "docs.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

SYSTEM_PROMPT = """You are an assistant answering questions about Indian fintech \
regulation and government schemes. You are given excerpts extracted directly \
from real RBI, SEBI, Union Budget, and government scheme PDF documents -- not \
summaries or FAQs. Each excerpt is labeled with the source folder, filename, \
and page it came from.

Rules:
- Answer only using the provided context excerpts. Do not rely on outside \
knowledge, even if you believe it is correct.
- When you use a fact from an excerpt, name the document it came from (folder \
and filename).
- If the context does not contain enough information to answer, say so \
plainly instead of guessing.
"""


def load_index():
    if not FAISS_INDEX_PATH.exists() or not DOCS_PATH.exists():
        raise SystemExit("Index not found -- run `python phase3_rag/build_index.py` first.")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs


def retrieve(question: str, index, docs, embed_model, top_k: int = TOP_K) -> list[dict]:
    query_vec = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({**docs[idx], "score": float(score)})
    return results


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        header = f"[Source: {c['folder']} / {c['filename']}, page {c['page']}]"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, client: Groq, index, docs, embed_model) -> tuple[str, list[dict]]:
    chunks = retrieve(question, index, docs, embed_model)
    if not chunks:
        return "No relevant context was found in the indexed documents.", []

    context = build_context(chunks)
    user_prompt = f"Context excerpts:\n\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content, chunks


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python phase3_rag/query.py "your question here"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])

    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise SystemExit("GROQ_API_KEY not set -- copy .env.example to .env and add a key.")

    print("Loading index...")
    index, docs = load_index()
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    client = Groq(api_key=groq_api_key)

    print(f"\nQuestion: {question}\n")
    answer, chunks = answer_question(question, client, index, docs, embed_model)

    print("Answer:")
    print(answer)

    if chunks:
        print("\nRetrieved from:")
        seen = set()
        for c in chunks:
            key = (c["folder"], c["filename"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  - {c['folder']} / {c['filename']}")


if __name__ == "__main__":
    main()
