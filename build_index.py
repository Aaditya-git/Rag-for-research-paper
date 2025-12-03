from pathlib import Path
from typing import List, Dict

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MILVUS_DB_PATH = "./milvus_rag.db"
COLLECTION_NAME = "rag_chunks"


def setup_milvus() -> None:
    """Connect to Milvus Lite using a local file."""
    print(f"Connecting to Milvus Lite at {MILVUS_DB_PATH} ...")
    connections.connect(alias="default", uri=MILVUS_DB_PATH)
    print("Connected.")


def create_collection(collection_name: str = COLLECTION_NAME) -> Collection:
    """Create or reset the collection for RAG chunks."""

    if utility.has_collection(collection_name):
        print(f"Collection {collection_name} already exists, dropping it.")
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="pdf_name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="page_number", dtype=DataType.INT64),
        FieldSchema(name="char_count", dtype=DataType.INT64),
    ]

    schema = CollectionSchema(fields=fields, description="RAG chunks")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    print(f"Created collection: {collection_name}")
    return collection


def load_chunks_from_markdown(md_file: Path) -> List[Dict]:
    """Parse chunks from markdown file produced by chunk_markdown_file."""
    chunks = []

    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()

    sections = content.split("## Chunk ")[1:]  # skip the header

    for section in sections:
        lines = section.split("\n")

        # chunk id is the first line after "## Chunk "
        chunk_id = lines[0].strip()

        # find text block
        text_start = None
        for i, line in enumerate(lines):
            if line.strip() == "**Text:**":
                text_start = i + 2
                break

        if text_start is None:
            continue

        text_lines = []
        for line in lines[text_start:]:
            if line.strip() == "---":
                break
            text_lines.append(line)

        text = "\n".join(text_lines).strip()
        if not text:
            continue

        # page number
        page_num = None
        for line in lines:
            if "Page Number:" in line:
                try:
                    page_num = int(line.split(":")[-1].strip())
                except Exception:
                    page_num = None

        chunk = {
            "chunk_id": chunk_id,
            "text": text,
            "page_number": page_num if page_num is not None else -1,
            "char_count": len(text),
        }
        chunks.append(chunk)

    return chunks


def insert_chunks(
    chunks: List[Dict],
    collection: Collection,
    model: SentenceTransformer,
    pdf_name: str,
) -> None:
    """Embed and insert chunks for a single pdf into Milvus."""
    if not chunks:
        print(f"No chunks to insert for {pdf_name}")
        return

    texts = [c["text"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks from {pdf_name} ...")
    embeddings = model.encode(texts, show_progress_bar=True)

    chunk_ids = [f"{pdf_name}_chunk_{c['chunk_id']}" for c in chunks]
    pdf_names = [pdf_name] * len(chunks)
    page_numbers = [int(c["page_number"]) for c in chunks]
    char_counts = [int(c["char_count"]) for c in chunks]

    data = [
        chunk_ids,
        pdf_names,
        texts,
        embeddings.tolist(),
        page_numbers,
        char_counts,
    ]

    collection.insert(data)
    collection.flush()

    print(f"Inserted {len(chunks)} chunks from {pdf_name} into {collection.name}.")


def build_rag_index() -> None:
    """Build index from all available chunk markdown files."""
    print("\n" + "=" * 60)
    print("BUILDING RAG INDEX IN MILVUS")
    print("=" * 60)

    setup_milvus()
    collection = create_collection()
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Loaded embedding model: {EMBEDDING_MODEL}")

    base_dir = Path("outputs/pymupdf4llm")
    chunk_files = list(base_dir.glob("**/chunks/*_chunks_recursive.md"))

    if not chunk_files:
        print("No chunk files found.")
        print("Run: python run_extraction.py first to generate chunks.")
        return

    print(f"Found {len(chunk_files)} chunk file(s).")

    total_chunks = 0
    for chunk_file in chunk_files:
        pdf_name = chunk_file.stem.replace("_chunks_recursive", "")
        print(f"\nProcessing file: {chunk_file}")
        print(f"Detected pdf name: {pdf_name}")

        chunks = load_chunks_from_markdown(chunk_file)
        print(f"Loaded {len(chunks)} chunks from markdown.")
        total_chunks += len(chunks)

        insert_chunks(chunks, collection, model, pdf_name)

    print("\n" + "=" * 60)
    print(f"INDEX BUILT SUCCESSFULLY. Total chunks indexed: {total_chunks}")
    print(f"Collection name: {collection.name}")
    print("=" * 60 + "\n")


def main() -> None:
    build_rag_index()


if __name__ == "__main__":
    main()

