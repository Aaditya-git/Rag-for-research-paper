import re
import json
from pathlib import Path


def split_into_sentences(text):
    """Split text into sentences."""
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text, chunk_size=1000, overlap=200):
    if not text or not text.strip():
        return []
    
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check if adding this sentence exceeds chunk size
        if current_chunk:
            test_chunk = current_chunk + " " + sentence
        else:
            test_chunk = sentence
        
        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk)
                
                # Create overlap from end of previous chunk
                overlap_text = get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                # Sentence itself is longer than chunk_size
                if len(sentence) > chunk_size:
                    # Split long sentence by words
                    sub_chunks = split_long_sentence(sentence, chunk_size)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1]
                else:
                    current_chunk = sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def get_overlap_text(text, overlap_size):
    if len(text) <= overlap_size:
        return text
    
    # Try to find a sentence boundary within overlap
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
        word_length = len(word) + 1  # +1 for space
        
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


def chunk_file(input_file, output_file=None, chunk_size=1000, overlap=200):
    """
    Read a markdown file and chunk its content.
    
    Args:
        input_file: Path to input markdown file
        output_file: Path to save chunks JSON (optional)
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks
    
    Returns:
        List of chunks
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"File not found: {input_file}")
        return []
    
    print(f"Reading file: {input_path.name}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"File size: {len(text):,} characters")
    
    chunks = chunk_text(text, chunk_size, overlap)
    
    print(f"Created {len(chunks)} chunks")
    
    if chunks:
        avg_size = sum(len(c) for c in chunks) / len(chunks)
        print(f"Average chunk size: {avg_size:.0f} characters")
    
    # Save to JSON if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        chunks_data = {
            "source_file": input_path.name,
            "total_chunks": len(chunks),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunks": [
                {
                    "chunk_id": i,
                    "text": chunk,
                    "char_count": len(chunk)
                }
                for i, chunk in enumerate(chunks)
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved chunks to: {output_path}")
    
    return chunks


def chunk_all_extractions(base_dir="outputs/pymupdf4llm", chunk_size=1000, overlap=200):
    """
    Chunk all extracted markdown files in the output directory.
    
    Args:
        base_dir: Base directory containing extracted PDFs
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
        
        output_file = md_file.parent / "chunks" / f"{md_file.stem}_chunks.json"
        
        chunks = chunk_file(
            str(md_file),
            str(output_file),
            chunk_size,
            overlap
        )
    
    print(f"\n{'=' * 60}")
    print("CHUNKING COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # Example usage
    chunk_all_extractions(
        base_dir="D:\study-code-repeat\coding\pdf-extraction-pipeline\outputs\pymupdf4llm",
        chunk_size=1000,
        overlap=200
    )