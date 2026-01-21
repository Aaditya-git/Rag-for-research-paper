from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CACHE_DIR = DATA_DIR / "cache"

MILVUS_DB_PATH = str(DATA_DIR / "milvus_rag.db")
COLLECTION_NAME = "rag_chunks"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

OPENAI_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
MAX_CONTEXT_CHUNKS = 3

