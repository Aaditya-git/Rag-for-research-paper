import os
from typing import List, Dict

from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from config import (
    MILVUS_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENAI_MODEL,
    TOP_K,
    MAX_CONTEXT_CHUNKS,
)


def setup():
    print(f"Connecting to Milvus at {MILVUS_DB_PATH}")
    connections.connect(alias="default", uri=MILVUS_DB_PATH)
    collection = Collection(COLLECTION_NAME)
    collection.load()
    print(f"Loaded collection: {COLLECTION_NAME}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    client = OpenAI(api_key=api_key)

    return collection, embed_model, client


def retrieve_top_chunks(query, collection, model, top_k=3, mode="mixed"):
    query_vec = model.encode([query])
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    expr = None
    mode = (mode or "mixed").strip().lower()
    if mode == "fixed":
        expr = 'chunk_type == "fixed"'
    elif mode == "semantic":
        expr = 'chunk_type == "semantic"'
    elif mode == "mixed":
        expr = None
    else:
        # unknown mode -> fall back to mixed
        expr = None
        mode = "mixed"

    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=[
            "chunk_id",
            "pdf_name",
            "page_number",
            "char_count",
            "text",
            "chunk_type",
            "chunker",
        ],
    )

    hits = []
    if results and results[0]:
        for hit in results[0]:
            entity = hit.entity
            hits.append(
                {
                    "distance": float(hit.distance),
                    "chunk_id": entity.get("chunk_id"),
                    "pdf_name": entity.get("pdf_name"),
                    "page_number": entity.get("page_number"),
                    "char_count": entity.get("char_count"),
                    "chunk_type": entity.get("chunk_type"),
                    "chunker": entity.get("chunker"),
                    "text": entity.get("text") or "",
                }
            )
    return hits, mode

def build_prompt(question: str, hits: List[Dict], max_chunks: int) -> str:
    selected = hits[:max_chunks]
    context_parts = []
    for h in selected:
        header = f"[Source: {h['pdf_name']}, page {h['page_number']}, chunk {h['chunk_id']}]"
        context_parts.append(header + "\n" + h["text"])

    context_text = "\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant answering questions about research papers.\n"
        "Use ONLY the provided context.\n"
        "If the answer is not clearly in the context, say you do not know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer clearly and concisely.\n"
    )
    return prompt


def ask_openai(client: OpenAI, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Answer only using the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def main():
    collection, embed_model, client = setup()

    print("\nRAG CLI ready. Type q to quit.\n")

    while True:
        try:
            question = input("Question: ").strip()
            mode_in = input("Mode [mixed/fixed/semantic] (default mixed): ").strip().lower()
            mode = mode_in if mode_in else "mixed"
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"q", "quit", "exit"}:
            break

        hits, mode = retrieve_top_chunks(question, collection, embed_model, top_k=3, mode=mode)

        if not hits:
            print("\nNo relevant context found.\n")
            continue

        prompt = build_prompt(question, hits, max_chunks=MAX_CONTEXT_CHUNKS)

        try:
            answer = ask_openai(client, prompt)
        except Exception as e:
            print(f"\nOpenAI error: {e}\n")
            continue

        print("\nAnswer:\n" + answer + "\n")

        print("Sources used:")
        for h in hits[:2]:
            print(
                f"- {h['pdf_name']} page {h['page_number']} "
                f"type {h.get('chunk_type')} chunk {h['chunk_id']} "
                f"distance {h['distance']:.4f}"
            )

        print()

    collection.release()


if __name__ == "__main__":
    main()

