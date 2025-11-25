from pathlib import Path
from chunking.text_chunker import chunk_markdown_file
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import time


# ==================== CONFIGURATION ====================

# Choose your embedding model
EMBEDDING_MODELS = {
    "mini": ("all-MiniLM-L6-v2", 384),           # Fast, basic (demo only)
    "mpnet": ("all-mpnet-base-v2", 768),         # Good balance
    "e5": ("intfloat/e5-large-v2", 1024),        # Research quality
    "bge": ("BAAI/bge-large-en-v1.5", 1024),     # State-of-the-art
}

# SELECT YOUR MODEL HERE
MODEL_KEY = "bge"  # Change to "e5" or "bge" for research
MODEL_NAME, EMBEDDING_DIM = EMBEDDING_MODELS[MODEL_KEY]

# Milvus connection settings
USE_DOCKER = False  # Set to True if using Docker Milvus

if USE_DOCKER:
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"
else:
    MILVUS_URI = "./milvus_demo.db"  # Milvus Lite


# ==================== FUNCTIONS ====================

def setup_milvus():
    """Setup Milvus connection."""
    print(f"Connecting to Milvus...")
    
    if USE_DOCKER:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        print(f"Connected to Milvus Standalone at {MILVUS_HOST}:{MILVUS_PORT}")
    else:
        connections.connect(
            alias="default",
            uri=MILVUS_URI
        )
        print(f"Connected to Milvus Lite at {MILVUS_URI}")


def create_collection(collection_name, dim=EMBEDDING_DIM):
    """Create a Milvus collection with specified dimensions."""
    
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="pdf_name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="char_count", dtype=DataType.INT64),
        FieldSchema(name="word_count", dtype=DataType.INT64),
        FieldSchema(name="page_number", dtype=DataType.INT64),
        FieldSchema(name="section_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="chunking_method", dtype=DataType.VARCHAR, max_length=50),
    ]
    
    schema = CollectionSchema(fields=fields, description=f"Research chunks: {collection_name}")
    collection = Collection(name=collection_name, schema=schema)
    
    # Create index for vector search
    index_params = {
        "metric_type": "COSINE",  
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    print(f"Created collection: {collection_name} (dim={dim})")
    return collection


def load_embedding_model(model_name):
    """Load embedding model."""
    print(f"Loading embedding model: {model_name}")
    print("This may take a few minutes on first run...")
    model = SentenceTransformer(model_name)
    print(f"Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def insert_chunks_to_milvus(chunks, collection, model, pdf_name):
    """Insert chunks into Milvus with embeddings."""
    
    if not chunks:
        print(f"No chunks to insert for {pdf_name}")
        return
    
    texts = [chunk['text'] for chunk in chunks]
    
    print(f"Generating {EMBEDDING_DIM}D embeddings for {len(texts)} chunks from {pdf_name}...")
    
    # For E5 models, add instruction prefix
    if "e5" in MODEL_NAME.lower():
        texts_with_prefix = [f"passage: {text}" for text in texts]
        embeddings = model.encode(texts_with_prefix, show_progress_bar=True, normalize_embeddings=True)
    else:
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    data = [
        [f"{pdf_name}_chunk_{chunk['chunk_id']}" for chunk in chunks],
        [pdf_name] * len(chunks),
        texts,
        embeddings.tolist(),
        [chunk['char_count'] for chunk in chunks],
        [chunk['word_count'] for chunk in chunks],
        [chunk.get('page_number', 0) for chunk in chunks],
        [chunk.get('section_type', 'body') for chunk in chunks],
        [chunk['chunking_method'] for chunk in chunks],
    ]
    
    collection.insert(data)
    collection.flush()
    
    print(f"Inserted {len(chunks)} chunks from {pdf_name}")


def search_similar_chunks(query, collection, model, top_k=5):
    """Search for similar chunks."""
    
    # For E5 models, add query prefix
    if "e5" in MODEL_NAME.lower():
        query_with_prefix = f"query: {query}"
        query_embedding = model.encode([query_with_prefix], normalize_embeddings=True)
    else:
        query_embedding = model.encode([query], normalize_embeddings=True)
    
    collection.load()
    
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_id", "pdf_name", "text", "char_count", "page_number", "section_type", "chunking_method"]
    )
    
    return results


def find_all_temp_markdown_files(base_dir="outputs/pymupdf4llm"):
    """Find all temporary markdown files."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return []
    
    temp_files = list(base_path.glob("**/*_temp.md"))
    return temp_files


def compare_chunking_methods(chunk_size=1000, overlap=200, similarity_threshold=0.7):
    """
    Compare recursive and semantic chunking for research.
    """
    print("\n" + "=" * 80)
    print("RESEARCH-GRADE PDF CHUNKING COMPARISON")
    print("=" * 80)
    print(f"Embedding Model: {MODEL_NAME}")
    print(f"Embedding Dimensions: {EMBEDDING_DIM}")
    print(f"Chunk Size: {chunk_size} chars")
    print(f"Overlap: {overlap} chars")
    print(f"Similarity Threshold: {similarity_threshold}")
    print("=" * 80)
    
    # Find PDFs
    md_files = find_all_temp_markdown_files()
    
    if not md_files:
        print("\nNo markdown files found!")
        print("Please run extraction first: python run_extraction.py")
        return
    
    print(f"\nFound {len(md_files)} PDF(s):")
    for md_file in md_files:
        pdf_name = md_file.stem.replace('_temp', '')
        print(f"  - {pdf_name}")
    
    # Setup
    setup_milvus()
    model = load_embedding_model(MODEL_NAME)
    
    # Create collections
    recursive_collection = create_collection("recursive_chunks")
    semantic_collection = create_collection("semantic_chunks")
    
    # Statistics
    all_recursive_chunks = []
    all_semantic_chunks = []
    recursive_time_total = 0
    semantic_time_total = 0
    
    # Process each PDF
    for md_file in md_files:
        pdf_name = md_file.stem.replace('_temp', '')
        
        print(f"\n{'='*80}")
        print(f"Processing: {pdf_name}")
        print('='*80)
        
        # Recursive chunking
        print("\n[RECURSIVE CHUNKING]")
        start = time.time()
        recursive_chunks = chunk_markdown_file(
            str(md_file),
            chunk_size=chunk_size,
            overlap=overlap,
            use_semantic=False
        )
        recursive_time = time.time() - start
        recursive_time_total += recursive_time
        print(f"Time: {recursive_time:.2f}s")
        
        # Semantic chunking
        print("\n[SEMANTIC CHUNKING]")
        start = time.time()
        semantic_chunks = chunk_markdown_file(
            str(md_file),
            chunk_size=chunk_size,
            use_semantic=True,
            similarity_threshold=similarity_threshold
        )
        semantic_time = time.time() - start
        semantic_time_total += semantic_time
        print(f"Time: {semantic_time:.2f}s")
        
        # Insert to Milvus
        print("\n[STORING IN MILVUS]")
        insert_chunks_to_milvus(recursive_chunks, recursive_collection, model, pdf_name)
        insert_chunks_to_milvus(semantic_chunks, semantic_collection, model, pdf_name)
        
        all_recursive_chunks.extend(recursive_chunks)
        all_semantic_chunks.extend(semantic_chunks)
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\nRecursive Chunking:")
    print(f"  Total PDFs: {len(md_files)}")
    print(f"  Total chunks: {len(all_recursive_chunks)}")
    if all_recursive_chunks:
        avg_char = sum(c['char_count'] for c in all_recursive_chunks) / len(all_recursive_chunks)
        print(f"  Avg chunk size: {avg_char:.0f} chars")
    print(f"  Processing time: {recursive_time_total:.2f}s")
    
    print(f"\nSemantic Chunking:")
    print(f"  Total PDFs: {len(md_files)}")
    print(f"  Total chunks: {len(all_semantic_chunks)}")
    if all_semantic_chunks:
        avg_char = sum(c['char_count'] for c in all_semantic_chunks) / len(all_semantic_chunks)
        print(f"  Avg chunk size: {avg_char:.0f} chars")
    print(f"  Processing time: {semantic_time_total:.2f}s")
    
    # Test queries
    print("\n" + "=" * 80)
    print("SEMANTIC SEARCH TEST")
    print("=" * 80)
    
    test_queries = [
        "What is the main contribution of this research?",
        "What datasets were used in the experiments?",
        "What are the limitations of the proposed method?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)
        
        print("\n[RECURSIVE RESULTS]")
        rec_results = search_similar_chunks(query, recursive_collection, model, top_k=3)
        for hits in rec_results:
            for i, hit in enumerate(hits):
                print(f"\n  Rank {i+1}: (Score: {hit.distance:.4f})")
                print(f"    PDF: {hit.entity.get('pdf_name')}")
                print(f"    Page: {hit.entity.get('page_number')}")
                print(f"    Section: {hit.entity.get('section_type')}")
                print(f"    Text: {hit.entity.get('text')[:200]}...")
        
        print("\n[SEMANTIC RESULTS]")
        sem_results = search_similar_chunks(query, semantic_collection, model, top_k=3)
        for hits in sem_results:
            for i, hit in enumerate(hits):
                print(f"\n  Rank {i+1}: (Score: {hit.distance:.4f})")
                print(f"    PDF: {hit.entity.get('pdf_name')}")
                print(f"    Page: {hit.entity.get('page_number')}")
                print(f"    Section: {hit.entity.get('section_type')}")
                print(f"    Text: {hit.entity.get('text')[:200]}...")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nDatabase: {'Docker Milvus' if USE_DOCKER else './milvus_demo.db'}")
    print(f"Model: {MODEL_NAME} ({EMBEDDING_DIM}D)")
    print(f"Collections:")
    print(f"  - recursive_chunks: {len(all_recursive_chunks)} chunks")
    print(f"  - semantic_chunks: {len(all_semantic_chunks)} chunks")


if __name__ == "__main__":
    compare_chunking_methods(
        chunk_size=1000,
        overlap=200,
        similarity_threshold=0.7
    )