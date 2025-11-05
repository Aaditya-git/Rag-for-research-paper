import re
import json
from pathlib import Path
from datetime import datetime


def split_into_sentences(text):
    """Split text into sentences."""
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def calculate_chunk_relevance(chunk_text):
    """
    Calculate relevance score for a chunk based on content indicators.
    Higher score = more relevant/important content.
    
    Factors:
    - Length (longer chunks often have more context)
    - Presence of keywords (abstract, introduction, conclusion, table, figure)
    - Number of citations or references
    - Numerical data presence
    """
    score = 0.5  # Base score
    
    # Length factor (normalize to 0-0.2)
    length_factor = min(len(chunk_text) / 2000, 0.2)
    score += length_factor
    
    # Important section keywords
    important_keywords = [
        'abstract', 'introduction', 'conclusion', 'summary',
        'methodology', 'results', 'discussion', 'key findings'
    ]
    
    chunk_lower = chunk_text.lower()
    for keyword in important_keywords:
        if keyword in chunk_lower:
            score += 0.1
    
    # Contains tables or figures
    if 'table' in chunk_lower or 'figure' in chunk_lower:
        score += 0.15
    
    # Contains citations (e.g., [1], (2020), et al.)
    citation_patterns = [r'\[\d+\]', r'\(\d{4}\)', r'et al\.']
    for pattern in citation_patterns:
        if re.search(pattern, chunk_text):
            score += 0.05
    
    # Contains numerical data
    if re.search(r'\d+\.?\d*%|\d+\.?\d*', chunk_text):
        score += 0.1
    
    # Normalize score to 0-1 range
    return min(score, 1.0)


def chunk_text(text, chunk_size=1000, overlap=200, source_metadata=None):
    """
    Chunk text respecting sentence boundaries with metadata.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        source_metadata: Dictionary with source document metadata
    
    Returns:
        List of dictionaries with chunk text and metadata
    """
    if not text or not text.strip():
        return []
    
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""
    char_position = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if current_chunk:
            test_chunk = current_chunk + " " + sentence
        else:
            test_chunk = sentence
        
        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunk_metadata = create_chunk_metadata(
                    chunk_text=current_chunk,
                    chunk_id=len(chunks),
                    char_start=char_position,
                    char_end=char_position + len(current_chunk),
                    source_metadata=source_metadata
                )
                chunks.append(chunk_metadata)
                
                char_position += len(current_chunk)
                overlap_text = get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                if len(sentence) > chunk_size:
                    sub_chunks = split_long_sentence(sentence, chunk_size)
                    for sub in sub_chunks[:-1]:
                        chunk_metadata = create_chunk_metadata(
                            chunk_text=sub,
                            chunk_id=len(chunks),
                            char_start=char_position,
                            char_end=char_position + len(sub),
                            source_metadata=source_metadata
                        )
                        chunks.append(chunk_metadata)
                        char_position += len(sub)
                    current_chunk = sub_chunks[-1]
                else:
                    current_chunk = sentence
    
    if current_chunk:
        chunk_metadata = create_chunk_metadata(
            chunk_text=current_chunk,
            chunk_id=len(chunks),
            char_start=char_position,
            char_end=char_position + len(current_chunk),
            source_metadata=source_metadata
        )
        chunks.append(chunk_metadata)
    
    return chunks


def create_chunk_metadata(chunk_text, chunk_id, char_start, char_end, source_metadata):
    """
    Create metadata for a chunk.
    
    Returns dictionary with:
    - chunk_id: Unique identifier
    - text: The actual chunk text
    - char_count: Length of chunk
    - char_start: Starting position in source
    - char_end: Ending position in source
    - relevance_score: Calculated importance score
    - source_document: Original document name
    - source_path: Full path to source
    - extraction_method: How it was extracted
    - created_at: Timestamp
    """
    relevance_score = calculate_chunk_relevance(chunk_text)
    
    metadata = {
        "chunk_id": chunk_id,
        "text": chunk_text,
        "char_count": len(chunk_text),
        "char_start": char_start,
        "char_end": char_end,
        "relevance_score": round(relevance_score, 3),
        "created_at": datetime.now().isoformat()
    }
    
    if source_metadata:
        metadata.update({
            "source_document": source_metadata.get("document_name"),
            "source_path": source_metadata.get("document_path"),
            "source_page_range": source_metadata.get("page_range"),
            "extraction_method": source_metadata.get("extraction_method", "pymupdf"),
            "document_type": source_metadata.get("document_type", "pdf")
        })
    
    return metadata


def get_overlap_text(text, overlap_size):
    """Get the last N characters from text for overlap."""
    if len(text) <= overlap_size:
        return text
    
    overlap_text = text[-overlap_size:]
    sentence_start = max(
        overlap_text.find('. '),
        overlap_text.find('! '),
        overlap_text.find('? ')
    )
    
    if sentence_start > 0:
        return overlap_text[sentence_start + 2:]
    
    return overlap_text


def split_long_sentence(sentence, max_size):
    """Split a sentence that's longer than max_size."""
    words = sentence.split()
    chunks = []
    current = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1
        
        if current_length + word_length <= max_size:
            current.append(word)
            current_length += word_length
        else:
            if current:
                chunks.append(" ".join(current))
            current = [word]
            current_length = len(word)
    
    if current:
        chunks.append(" ".join(current))
    
    return chunks


def save_chunks_as_markdown(chunks, output_file):
    """
    Save chunks as a markdown file with metadata in frontmatter.
    
    Format:
    ---
    chunk_id: 0
    relevance_score: 0.75
    source_document: paper.pdf
    char_range: 0-1000
    ---
    
    [Chunk text here]
    
    ---
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Chunked Document\n\n")
        f.write(f"**Total Chunks:** {len(chunks)}\n\n")
        f.write(f"---\n\n")
        
        for chunk in chunks:
            f.write(f"## Chunk {chunk['chunk_id']}\n\n")
            
            # Metadata section
            f.write(f"**Metadata:**\n")
            f.write(f"- Relevance Score: {chunk['relevance_score']}\n")
            f.write(f"- Character Count: {chunk['char_count']}\n")
            f.write(f"- Position: {chunk['char_start']}-{chunk['char_end']}\n")
            
            if 'source_document' in chunk:
                f.write(f"- Source Document: {chunk['source_document']}\n")
            if 'source_path' in chunk:
                f.write(f"- Source Path: {chunk['source_path']}\n")
            if 'extraction_method' in chunk:
                f.write(f"- Extraction Method: {chunk['extraction_method']}\n")
            
            f.write(f"\n**Text:**\n\n")
            f.write(f"{chunk['text']}\n\n")
            f.write(f"---\n\n")
    
    print(f"Saved chunks as markdown: {output_path}")


def chunk_file(input_file, output_format="markdown", chunk_size=1000, overlap=200):
    """
    Read a markdown file and chunk its content.
    
    Args:
        input_file: Path to input markdown file
        output_format: "markdown" or "json"
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks
    
    Returns:
        List of chunks with metadata
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"File not found: {input_file}")
        return []
    
    print(f"Reading file: {input_path.name}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"File size: {len(text):,} characters")
    
    # Create source metadata
    source_metadata = {
        "document_name": input_path.name,
        "document_path": str(input_path.absolute()),
        "extraction_method": "pymupdf",
        "document_type": "pdf"
    }
    
    chunks = chunk_text(text, chunk_size, overlap, source_metadata)
    
    print(f"Created {len(chunks)} chunks")
    
    if chunks:
        avg_size = sum(c['char_count'] for c in chunks) / len(chunks)
        avg_relevance = sum(c['relevance_score'] for c in chunks) / len(chunks)
        print(f"Average chunk size: {avg_size:.0f} characters")
        print(f"Average relevance score: {avg_relevance:.3f}")
    
    # Save output
    output_dir = input_path.parent / "chunks"
    output_dir.mkdir(exist_ok=True)
    
    if output_format == "markdown":
        output_file = output_dir / f"{input_path.stem}_chunks.md"
        save_chunks_as_markdown(chunks, output_file)
    else:
        output_file = output_dir / f"{input_path.stem}_chunks.json"
        chunks_data = {
            "source_file": input_path.name,
            "source_path": str(input_path.absolute()),
            "total_chunks": len(chunks),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "created_at": datetime.now().isoformat(),
            "chunks": chunks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved chunks as JSON: {output_file}")
    
    return chunks


def chunk_all_extractions(base_dir="outputs/pymupdf4llm", output_format="markdown", 
                          chunk_size=1000, overlap=200):
    """
    Chunk all extracted markdown files in the output directory.
    
    Args:
        base_dir: Base directory containing extracted PDFs
        output_format: "markdown" or "json"
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    print(f"\nSearching for markdown files in: {base_dir}")
    print("-" * 60)
    
    md_files = list(base_path.glob("**/*_text.md"))
    
    if not md_files:
        print("No markdown files found")
        return
    
    print(f"Found {len(md_files)} markdown file(s)\n")
    
    for md_file in md_files:
        print(f"\nProcessing: {md_file.name}")
        print("-" * 40)
        
        chunks = chunk_file(
            str(md_file),
            output_format=output_format,
            chunk_size=chunk_size,
            overlap=overlap
        )
    
    print(f"\n{'=' * 60}")
    print("CHUNKING COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    chunk_all_extractions(
        base_dir="D:\study-code-repeat\coding\pdf-extraction-pipeline\outputs\pymupdf4llm",
        output_format="markdown",  # or "json"
        chunk_size=1000,
        overlap=200
    )