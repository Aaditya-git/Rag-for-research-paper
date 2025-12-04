from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

MILVUS_DB = "./milvus_rag.db"
COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def setup():
    """Connect to Milvus and load collection + embedding model."""
    print(f"Connecting to Milvus at {MILVUS_DB} ...")
    connections.connect(alias="default", uri=MILVUS_DB)
    collection = Collection(COLLECTION_NAME)
    collection.load()
    print(f"Connected. Collection '{COLLECTION_NAME}' loaded.")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.")

    return collection, model


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


def main():
    collection, model = setup()

    print("\nRetrieval CLI ready.")
    print("Type your question. Type 'q' or 'quit' to exit.\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"q", "quit", "exit"}:
            print("Goodbye.")
            break

        print("\nSearching...")
        hits = retrieve_top_chunks(query, collection, model, top_k=3)

        if not hits:
            print("No results found.\n")
            continue

        print("\nTop results:")
        for idx, h in enumerate(hits, start=1):
            preview = h["text"][:300].replace("\n", " ")
            print(f"\nRank {idx}:")
            print(f"  Distance:   {h['score']:.4f}")
            print(f"  PDF:        {h['pdf_name']}")
            print(f>  Page:       {h['page_number']}")
            print(f"  Char count: {h['char_count']}")
            print(f"  Text:       {preview}...")

        print()

    collection.release()



if __name__ == "__main__":
    main()


