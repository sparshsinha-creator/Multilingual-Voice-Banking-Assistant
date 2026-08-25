"""
Phase 3 of 7 -- RAG index build

Walks phase3_rag_datasource/<FolderName>/*.pdf -- real RBI, SEBI, Union
Budget, and Government Schemes PDFs -- extracts page text with pypdf,
splits each document into overlapping word-count chunks, embeds every
chunk with all-MiniLM-L6-v2, and writes a FAISS index plus a docs.json
sidecar (chunk text + source metadata) that query.py loads at retrieval
time.

data/fintech_schemes_faq.json is no longer read here; it's kept in the
repo as an unused fallback reference only.

Run with: python phase3_rag/build_index.py
"""

import json
import sys
from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASOURCE_DIR = Path(__file__).resolve().parent.parent / "phase3_rag_datasource"
INDEX_DIR = Path(__file__).resolve().parent / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
DOCS_PATH = INDEX_DIR / "docs.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_WORDS = 650
CHUNK_OVERLAP_WORDS = 75


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) for a PDF, 1-indexed."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"    WARNING: failed to extract page {i}: {e}")
            text = ""
        pages.append((i, text))
    return pages


def chunk_document(
    pages: list[tuple[int, str]],
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[dict]:
    """Flatten a document's pages into a single word stream (tracking each
    word's source page), then slide a fixed-size overlapping window over it
    so no chunk is cut off mid-sentence at a page boundary."""
    words: list[str] = []
    page_of_word: list[int] = []
    for page_num, text in pages:
        page_words = text.split()
        words.extend(page_words)
        page_of_word.extend([page_num] * len(page_words))

    if not words:
        return []

    chunks = []
    step = chunk_words - overlap_words
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_words, n)
        chunk_pages = page_of_word[start:end]
        page_start, page_end = chunk_pages[0], chunk_pages[-1]
        page_label = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        chunks.append({"text": " ".join(words[start:end]), "page": page_label})
        if end == n:
            break
        start += step
    return chunks


def build_index() -> None:
    if not DATASOURCE_DIR.exists():
        raise SystemExit(f"Datasource folder not found: {DATASOURCE_DIR}")

    pdf_paths = sorted(
        p for p in DATASOURCE_DIR.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        raise SystemExit(f"No PDFs found under {DATASOURCE_DIR}")

    records = []
    for pdf_path in pdf_paths:
        folder = pdf_path.parent.name
        filename = pdf_path.name
        print(f"Processing {folder}/{filename}...")
        try:
            pages = extract_pages(pdf_path)
        except Exception as e:
            print(f"  WARNING: could not open {filename}: {e}")
            continue

        chunks = chunk_document(pages)
        if not chunks:
            print(f"  WARNING: no extractable text in {filename} (skipped)")
            continue

        for i, chunk in enumerate(chunks):
            records.append(
                {
                    "id": len(records),
                    "text": chunk["text"],
                    "folder": folder,
                    "filename": filename,
                    "page": chunk["page"],
                    "chunk_index": i,
                }
            )
        print(f"  -> {len(chunks)} chunks from {len(pages)} pages")

    if not records:
        raise SystemExit("No text extracted from any PDF -- nothing to index.")

    print(f"\nEmbedding {len(records)} chunks with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        [r["text"] for r in records], show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved FAISS index -> {FAISS_INDEX_PATH}")
    print(f"Saved chunk metadata -> {DOCS_PATH} ({len(records)} chunks)")


if __name__ == "__main__":
    build_index()
