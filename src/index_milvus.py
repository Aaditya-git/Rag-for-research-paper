import json
from pathlib import Path
from typing import Dict, List, Iterable

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from sentence_transformers import SentenceTransformer

from config import (
    CACHE_DIR,
    MILVUS_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)


def setup_milvus() -> None:
    print(f"Connecting to Milvus Lite at {MILVUS_DB_PATH}")
    connections.connect(alias="default", uri=MILVUS_DB_PATH)
    print("Connected.")


def create_or_reset_collection() -> Collection:
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} exists, dropping.")
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="pdf_name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="page_number", dtype=DataType.INT64),
        FieldSchema(name="char_count", dtype=DataType.INT64),
        FieldSchema(name='chunk_type',dtype=DataType.VARCHAR,max_length=200),
        FieldSchema(name='chunker',dtype=DataType.VARCHAR,max_length=200)
    ]

    schema = CollectionSchema(fields=fields, description="RAG chunks")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Created collection: {COLLECTION_NAME}")
    return collection


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def insert_chunks_for_file(collection: Collection, model: SentenceTransformer, jsonl_file: Path) -> int:
    rows = read_jsonl(jsonl_file)
    if not rows:
        print(f"Empty cache file: {jsonl_file.name}")
        return 0

    texts = [r["text"] for r in rows]
    embeddings = model.encode(texts, show_progress_bar=True)

    chunk_id_col = [r["chunk_id"] for r in rows]
    pdf_name_col = [r["pdf_name"] for r in rows]
    text_col = [r["text"] for r in rows]
    emb_col = embeddings
    page_col = [int(r.get("page_number", -1)) for r in rows]
    char_col = [int(r.get("char_count", len(r.get("text", "")))) for r in rows]
    chunk_type_col = [r.get("chunk_type", "fixed") for r in rows]
    chunker_col = [r.get("chunker", "fixed_recursive_v1") for r in rows]

    collection.insert(
        [
            chunk_id_col,
            pdf_name_col,
            text_col,
            emb_col,
            page_col,
            char_col,
            chunk_type_col,
            chunker_col
        ]
    )
    return len(rows)


def build_index() -> None:
    setup_milvus()
    collection = create_or_reset_collection()

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    cache_files = sorted(CACHE_DIR.glob("*.jsonl"))
    if not cache_files:
        print(f"No cache files found in {CACHE_DIR}")
        print("Run: python src/extract_chunk.py")
        return

    total = 0
    for f in cache_files:
        print(f"\nIndexing: {f.name}")
        count = insert_chunks_for_file(collection, model, f)
        total += count
        print(f"Inserted {count} chunks.")

    collection.flush()
    collection.load()

    print("\nDone.")
    print(f"Total chunks indexed: {total}")
    print(f"Collection: {collection.name}")


if __name__ == "__main__":
    build_index()

