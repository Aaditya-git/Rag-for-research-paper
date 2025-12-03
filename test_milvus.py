from pymilvus import connections, utility, Collection

MILVUS_DB = "./milvus_rag.db"
COLLECTION_NAME = "rag_chunks"

def main():
    print(f"Connecting to Milvus at {MILVUS_DB} ...")
    connections.connect(alias="default", uri=MILVUS_DB)
    print("Connected.\n")

    # List all collections
    collections = utility.list_collections()
    print("Collections in Milvus:", collections)

    # Check if `rag_chunks` exists
    if COLLECTION_NAME not in collections:
        print(f"Collection '{COLLECTION_NAME}' does NOT exist.")
        return

    # Load collection
    collection = Collection(COLLECTION_NAME)
    count = collection.num_entities
    print(f"\nCollection name: {COLLECTION_NAME}")
    print(f"Number of records: {count}")

    if count == 0:
        print("Collection is empty, nothing to query.")
        return

    # Load for querying
    collection.load()

    print("\nQuerying one record...")
    try:
        # expr matches anything since page_number >= -10000 is always true
        results = collection.query(
            expr="page_number >= -10000",
            output_fields=["chunk_id", "pdf_name", "page_number", "char_count","text"],
            limit=1
        )

        print("\nResult:")
        print(results[0] if results else "No results returned.")

    except Exception as e:
        print("Query error:", e)

if __name__ == "__main__":
    main()

