import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

from config import PDF_DIR, CACHE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


def split_into_sentences(text: str) -> List[str]:
    sentence_pattern = r"(?<=[.!?])\s+"
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def get_overlap_text(text: str, overlap_size: int) -> str:
    if len(text) <= overlap_size:
        return text

    tail = text[-overlap_size:]
    idx = max(tail.rfind(". "), tail.rfind("! "), tail.rfind("? "))
    if idx >= 0:
        return tail[idx + 2 :].strip()
    return tail.strip()


def split_long_sentence(sentence: str, max_size: int) -> List[str]:
    words = sentence.split()
    chunks = []
    current = []
    current_len = 0

    for w in words:
        add_len = len(w) + 1
        if current_len + add_len <= max_size:
            current.append(w)
            current_len += add_len
        else:
            if current:
                chunks.append(" ".join(current).strip())
            current = [w]
            current_len = len(w)

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


def chunk_text_recursive(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text or not text.strip():
        return []

    sentences = split_into_sentences(text)
    chunks: List[str] = []
    current = ""

    for sent in sentences:
        test = (current + " " + sent).strip() if current else sent

        if len(test) <= chunk_size:
            current = test
            continue

        if current:
            chunks.append(current.strip())
            overlap_text = get_overlap_text(current, overlap)
            current = (overlap_text + " " + sent).strip() if overlap_text else sent
            continue

        if len(sent) > chunk_size:
            subs = split_long_sentence(sent, chunk_size)
            chunks.extend(subs[:-1])
            current = subs[-1] if subs else ""
        else:
            current = sent

    if current.strip():
        chunks.append(current.strip())

    return chunks


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_chunk_page(
    page_text: str,
    embed_model: SentenceTransformer,
    sim_threshold: float = 0.55,
    min_chars: int = 300,
    max_chars: int = 1800,
) -> List[str]:
    """
    Semantic chunking per page.

    Idea:
    - Embed each sentence
    - Compare adjacent sentences via cosine similarity
    - If similarity drops below threshold and chunk is big enough, split
    - Force split if chunk grows too large
    - Merge tiny chunks
    """
    sentences = split_into_sentences(page_text)
    if not sentences:
        return []

    joined = " ".join(sentences).strip()
    if len(joined) <= max_chars and len(joined) >= min_chars:
        return [joined]

    sent_vecs = embed_model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)

    chunks: List[str] = []
    current_sents: List[str] = []
    current_len = 0

    def flush_current():
        nonlocal current_sents, current_len
        if current_sents:
            chunks.append(" ".join(current_sents).strip())
        current_sents = []
        current_len = 0

    for i, sent in enumerate(sentences):
        s = sent.strip()
        if not s:
            continue

        if current_sents:
            proposed_len = current_len + 1 + len(s)
        else:
            proposed_len = len(s)

        if current_sents and proposed_len > max_chars:
            flush_current()

        current_sents.append(s)
        current_len = (current_len + 1 + len(s)) if len(current_sents) > 1 else len(s)

        if i < len(sentences) - 1:
            sim = cosine_sim(sent_vecs[i], sent_vecs[i + 1])

            if sim < sim_threshold and current_len >= min_chars:
                flush_current()

    flush_current()

    merged: List[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue

        if not merged:
            merged.append(c)
            continue

        if len(c) < min_chars:
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)

    final: List[str] = []
    for c in merged:
        if len(c) > max_chars:
            final.extend(chunk_text_recursive(c, chunk_size=max_chars, overlap=0))
        else:
            final.append(c)

    return [c for c in final if c.strip()]


def extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    doc = fitz.open(pdf_path)
    pages: List[Tuple[int, str]] = []
    for page_number in range(len(doc)):
        text = doc[page_number].get_text() or ""
        text = text.strip()
        if text:
            pages.append((page_number + 1, text))
    doc.close()
    return pages


def pdf_to_chunks(
    pdf_path: Path,
    chunk_size: int,
    overlap: int,
    embed_model: SentenceTransformer,
) -> List[Dict]:
    """
    Produce BOTH fixed and semantic chunks for each page.
    Each chunk record includes chunk_type + chunker so downstream can filter.
    """
    pdf_name = pdf_path.stem
    pages = extract_pages(pdf_path)

    records: List[Dict] = []

    fixed_id = 0
    semantic_id = 0

    for page_number, page_text in pages:
        # Fixed chunks (existing)
        fixed_chunks = chunk_text_recursive(page_text, chunk_size=chunk_size, overlap=overlap)
        for ctext in fixed_chunks:
            records.append(
                {
                    "chunk_id": f"{pdf_name}_fixed_p{page_number}_c{fixed_id}",
                    "pdf_name": pdf_name,
                    "page_number": page_number,
                    "char_count": len(ctext),
                    "text": ctext,
                    "chunk_type": "fixed",
                    "chunker": "fixed_recursive_v1",
                }
            )
            fixed_id += 1

        # Semantic chunks (new)
        semantic_chunks = semantic_chunk_page(
            page_text,
            embed_model=embed_model,
            sim_threshold=0.55,
            min_chars=300,
            max_chars=1800,
        )
        for ctext in semantic_chunks:
            records.append(
                {
                    "chunk_id": f"{pdf_name}_semantic_p{page_number}_c{semantic_id}",
                    "pdf_name": pdf_name,
                    "page_number": page_number,
                    "char_count": len(ctext),
                    "text": ctext,
                    "chunk_type": "semantic",
                    "chunker": "semantic_adjacent_v1",
                }
            )
            semantic_id += 1

    return records


def write_jsonl(records: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_cache_for_all_pdfs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in: {PDF_DIR}")
        return

    print(f"Loading embedding model once: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Found {len(pdf_files)} PDF(s).")
    for pdf in pdf_files:
        print(f"\nExtracting and chunking: {pdf.name}")
        records = pdf_to_chunks(
            pdf,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            embed_model=embed_model,
        )
        out_file = CACHE_DIR / f"{pdf.stem}.jsonl"
        write_jsonl(records, out_file)

        fixed_count = sum(1 for r in records if r.get("chunk_type") == "fixed")
        semantic_count = sum(1 for r in records if r.get("chunk_type") == "semantic")
        print(f"Wrote {len(records)} total records: fixed={fixed_count}, semantic={semantic_count}")
        print(f"Cache: {out_file}")


if __name__ == "__main__":
    build_cache_for_all_pdfs()

