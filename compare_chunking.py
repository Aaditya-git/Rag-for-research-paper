from pathlib import Path
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import json


# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MILVUS_DB = "./milvus_rag.db"


def setup_milvus():
    """Connect to Milvus Lite."""
    print("Connecting to Milvus Lite...")
    connections.connect(
        alias="default",
        uri=MILVUS_DB
    )
    print(f"Connected to Milvus at {MILVUS_DB}")


def create_collection(collection_name="rag_chunks"):
    """Create collection for RAG chunks."""
    
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")
    
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
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    print(f"Created collection: {collection_name}")
    return collection


def load_chunks_from_markdown(md_file):
    """Parse chunks from markdown file."""
    chunks = []
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple parser - splits by "## Chunk"
    chunk_sections = content.split("## Chunk ")[1:]  # Skip header
    
    for section in chunk_sections:
        lines = section.split('\n')
        
        # Extract chunk_id
        chunk_id = lines[0].strip()
        
        # Find text section
        text_start = None
        for i, line in enumerate(lines):
            if line.strip() == "**Text:**":
                text_start = i + 2
                break
        
        if text_start:
            # Get text until next separator or end
            text_lines = []
            for line in lines[text_start:]:
                if line.strip() == "---":
                    break
                text_lines.append(line)
            
            text = '\n'.join(text_lines).strip()
            
            if text:
                # Extract page number
                page_num = 1
                for line in lines:
                    if "Page Number:" in line:
                        try:
                            page_num = int(line.split(":")[-1].strip())
                        except:
                            pass
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "page_number": page_num,
                    "char_count": len(text)
                })
    
    return chunks


def insert_chunks(chunks, collection, model, pdf_name):
    """Insert chunks into Milvus."""
    
    if not chunks:
        print(f"No chunks to insert for {pdf_name}")
        return
    
    texts = [chunk['text'] for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    data = [
        [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))],
        [pdf_name] * len(chunks),
        texts,
        embeddings.tolist(),
        [chunk['page_number'] for chunk in chunks],
        [chunk['char_count'] for chunk in chunks],
    ]
    
    collection.insert(data)
    collection.flush()
    
    print(f"Inserted {len(chunks)} chunks from {pdf_name}")


def search(query, collection, model, top_k=5):
    """Search for similar chunks."""
    
    query_embedding = model.encode([query])
    
    collection.load()
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_id", "pdf_name", "text", "page_number", "char_count"]
    )
    
    return results


def build_rag_index():
    """Build RAG index from all chunked files."""
    
    print("\n" + "=" * 60)
    print("BUILDING RAG INDEX")
    print("=" * 60)
    
    # Setup
    setup_milvus()
    collection = create_collection()
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Loaded model: {EMBEDDING_MODEL}\n")
    
    # Find all chunk files
    base_dir = Path("outputs/pymupdf4llm")
    chunk_files = list(base_dir.glob("**/chunks/*_recursive.md"))
    
    if not chunk_files:
        print("No chunk files found!")
        print("Run: python run_extraction.py first")
        return None
    
    print(f"Found {len(chunk_files)} chunk file(s):")
    
    # Process each chunk file
    for chunk_file in chunk_files:
        pdf_name = chunk_file.stem.replace('_chunks_recursive', '')
        print(f"\nProcessing: {pdf_name}")
        
        chunks = load_chunks_from_markdown(chunk_file)
        print(f"Loaded {len(chunks)} chunks from file")
        
        insert_chunks(chunks, collection, model, pdf_name)
    
    print("\n" + "=" * 60)
    print("INDEX BUILT SUCCESSFULLY")
    print("=" * 60)
    
    return collection, model


def interactive_search(collection, model):
    """Interactive RAG search."""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE RAG SEARCH")
    print("=" * 60)
    print("Type your questions (or 'quit' to exit)\n")
    
    while True:
        query = input("\nQuery: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print("\nSearching...")
        results = search(query, collection, model, top_k=3)
        
        print("\n" + "-" * 60)
        print("RESULTS:")
        print("-" * 60)
        
        for hits in results:
            for i, hit in enumerate(hits):
                print(f"\nRank {i+1}: (Distance: {hit.distance:.4f})")
                print(f"  PDF: {hit.entity.get('pdf_name')}")
                print(f"  Page: {hit.entity.get('page_number')}")
                print(f"  Text: {hit.entity.get('text')[:300]}...")
                print()


def main():
    """Main RAG workflow."""
    
    # Build index
    collection, model = build_rag_index()
    
    if collection and model:
        # Interactive search
        interactive_search(collection, model)


if __name__ == "__main__":
    main()