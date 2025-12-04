import os

from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from openai import OpenAI

MILVUS_DB = "./milvus_rag.db"
COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o-mini"


def setup():
    """Connect to Milvus, load collection + embedding model, init OpenAI client."""
    print(f"Connecting to Milvus at {MILVUS_DB} ...")
    connections.connect(alias="default", uri=MILVUS_DB)
    collection = Collection(COLLECTION_NAME)
    collection.load()
    print(f"Connected. Collection '{COLLECTION_NAME}' loaded.")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    client = OpenAI(api_key=api_key)
    print("OpenAI client initialized.")

    return collection, embed_model, client


def retrieve_top_chunks(query, collection, model, top_k=3):
    """Embed the query and retrieve top_k similar chunks from Milvus."""
    query_vec = model.encode([query])
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_id", "pdf_name", "page_number", "char_count", "text"],
    )

    hits = []
    if results and results[0]:
        for hit in results[0]:
            entity = hit.entity
            hits.append(
                {
                    "score": float(hit.distance),
                    "chunk_id": entity.get("chunk_id"),
                    "pdf_name": entity.get("pdf_name"),
                    "page_number": entity.get("page_number"),
                    "char_count": entity.get("char_count"),
                    "text": entity.get("text") or "",
                }
            )
    return hits


def build_prompt(question, hits, max_chunks=2):
    """Build a simple RAG prompt from the top chunks and the question."""
    # Use only the top N chunks for context
    selected = hits[:max_chunks]

    context_parts = []
    for h in selected:
        header = f"[From PDF {h['pdf_name']}, page {h['page_number']}]"
        context_parts.append(header + "\n" + h["text"])

    context_text = "\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant that answers questions about research papers.\n"
        "You are given some context extracted from the papers. Use ONLY this context.\n"
        "If the answer is not clearly in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer in clear, concise natural language.\n"
    )

    return prompt


def ask_openai(client, prompt):
    """Send the RAG prompt to OpenAI and return the answer text."""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You answer user questions based only on the provided context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def main():
    collection, embed_model, client = setup()

    print("\nRAG Q&A CLI ready.")
    print("Ask questions about your indexed papers.")
    print("Type 'q' or 'quit' to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"q", "quit", "exit"}:
            print("Goodbye.")
            break

        print("\nRetrieving context from Milvus...")
        hits = retrieve_top_chunks(question, collection, embed_model, top_k=3)

        if not hits:
            print("No relevant context found in Milvus. Cannot answer.\n")
            continue

        prompt = build_prompt(question, hits, max_chunks=2)

        print("Calling OpenAI...")
        try:
            answer = ask_openai(client, prompt)
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            continue

        print("\nAnswer:")
        print(answer)
        print("\n---\n")

    collection.release()


if __name__ == "__main__":
    main()

