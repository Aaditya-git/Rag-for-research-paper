import re
from pathlib import Path


# ==================== HELPER FUNCTIONS ====================

def split_into_sentences(text):
    """Split text into sentences."""
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def detect_section_type(text):
    """Detect what type of section this chunk belongs to."""
    text_lower = text.lower()

    if any(word in text_lower[:100] for word in ['abstract', 'overview', 'tldr']):
        return "abstract"
    elif any(word in text_lower[:100] for word in ['introduction', 'background', 'motivation']):
        return "introduction"
    elif any(word in text_lower[:100] for word in ['method', 'approach', 'algorithm', 'implementation']):
        return "methodology"
    elif any(word in text_lower[:100] for word in ['result', 'experiment', 'evaluation', 'performance']):
        return "results"
    elif any(word in text_lower[:100] for word in ['discussion', 'analysis', 'interpretation']):
        return "discussion"
    elif any(word in text_lower[:100] for word in ['conclusion', 'summary', 'future work']):
        return "conclusion"
    elif any(word in text_lower[:100] for word in ['reference', 'bibliography', 'citation']):
        return "references"
    else:
        return "body"


def create_text_chunk_metadata(chunk_text, chunk_id, char_start, char_end, source_metadata, page_number=None, chunk_method="recursive"):
    """Create rich metadata for a text chunk."""

    sentence_count = len([s for s in chunk_text.split('.') if s.strip()])
    word_count = len(chunk_text.split())

    has_code = bool(re.search(r'```|def |class |import |function', chunk_text))
    has_math = bool(re.search(r'\$.*?\$|\\frac|\\sum|equation', chunk_text))
    has_citation = bool(re.search(r'\[\d+\]|\(\d{4}\)|et al\.', chunk_text))
    has_list = bool(re.search(r'^\s*[-*•]\s', chunk_text, re.MULTILINE))

    section_type = detect_section_type(chunk_text)

    metadata = {
        "chunk_id": chunk_id,
        "type": "text",
        "text": chunk_text,
        "chunking_method": chunk_method,
        "char_count": len(chunk_text),
        "char_start": char_start,
        "char_end": char_end,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "contains_code": has_code,
        "contains_math": has_math,
        "contains_citation": has_citation,
        "contains_list": has_list,
        "section_type": section_type,
        "page_number": page_number,
    }

    if source_metadata:
        metadata.update(source_metadata)

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
    """Split a sentence that is longer than max_size."""
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


# ==================== RECURSIVE CHUNKING ONLY ====================

def chunk_text_recursive(text, chunk_size=1000, overlap=200, source_metadata=None):
    """
    Recursive chunking: Respects sentence boundaries.
    Traditional method - fast and reliable.
    """
    if not text or not text.strip():
        return []

    page_pattern = r'--- Page (\d+) ---'
    page_markers = [(m.start(), int(m.group(1))) for m in re.finditer(page_pattern, text)]

    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""
    char_position = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        test_chunk = current_chunk + " " + sentence if current_chunk else sentence

        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            if current_chunk:
                page_num = None
                for marker_pos, page in page_markers:
                    if marker_pos <= char_position:
                        page_num = page
                    else:
                        break

                chunk_data = create_text_chunk_metadata(
                    chunk_text=current_chunk,
                    chunk_id=len(chunks),
                    char_start=char_position,
                    char_end=char_position + len(current_chunk),
                    source_metadata=source_metadata,
                    page_number=page_num,
                    chunk_method="recursive"
                )
                chunks.append(chunk_data)

                char_position += len(current_chunk)
                overlap_text = get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                if len(sentence) > chunk_size:
                    sub_chunks = split_long_sentence(sentence, chunk_size)
                    for sub in sub_chunks[:-1]:
                        page_num = None
                        for marker_pos, page in page_markers:
                            if marker_pos <= char_position:
                                page_num = page

                        chunk_data = create_text_chunk_metadata(
                            chunk_text=sub,
                            chunk_id=len(chunks),
                            char_start=char_position,
                            char_end=char_position + len(sub),
                            source_metadata=source_metadata,
                            page_number=page_num,
                            chunk_method="recursive"
                        )
                        chunks.append(chunk_data)
                        char_position += len(sub)
                    current_chunk = sub_chunks[-1]
                else:
                    current_chunk = sentence

    if current_chunk:
        page_num = None
        for marker_pos, page in page_markers:
            if marker_pos <= char_position:
                page_num = page

        chunk_data = create_text_chunk_metadata(
            chunk_text=current_chunk,
            chunk_id=len(chunks),
            char_start=char_position,
            char_end=char_position + len(current_chunk),
            source_metadata=source_metadata,
            page_number=page_num,
            chunk_method="recursive"
        )
        chunks.append(chunk_data)

    return chunks


# ==================== SAVE CHUNKS ====================

def save_chunks_as_markdown(chunks, output_file, method_name):
    """Save chunks as markdown."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Chunked Document - {method_name}\n\n")
        f.write(f"**Total Chunks:** {len(chunks)}\n")
        f.write(f"**Chunking Method:** {method_name}\n\n")
        f.write("---\n\n")

        for chunk in chunks:
            f.write(f"## Chunk {chunk['chunk_id']}\n\n")
            f.write("**Metadata:**\n")
            f.write(f"- Type: {chunk['type']}\n")
            f.write(f"- Chunking Method: {chunk['chunking_method']}\n")
            f.write(f"- Page Number: {chunk.get('page_number', 'N/A')}\n")
            f.write(f"- Section Type: {chunk['section_type']}\n")
            f.write(f"- Character Count: {chunk['char_count']}\n")
            f.write(f"- Word Count: {chunk['word_count']}\n")
            f.write(f"- Sentence Count: {chunk['sentence_count']}\n")

            f.write("\n**Text:**\n\n")
            f.write(f"{chunk['text']}\n\n")
            f.write("---\n\n")

    print(f"Saved {len(chunks)} chunks to: {output_file}")


def chunk_markdown_file(md_file, chunk_size=1000, overlap=200):
    """
    Chunk a markdown file using recursive chunking only.

    Args:
        md_file: Path to markdown file
        chunk_size: Maximum chunk size
        overlap: Overlap for recursive chunking
    """
    md_path = Path(md_file)

    print(f"\nChunking: {md_path.name}")
    print("Method: Recursive")

    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"File size: {len(text):,} characters")

    source_metadata = {
        "source_document": md_path.name,
        "source_path": str(md_path.absolute()),
    }

    chunks = chunk_text_recursive(text, chunk_size, overlap, source_metadata)
    method_name = "Recursive"

    print(f"Created {len(chunks)} chunks")

    if chunks:
        avg_size = sum(c['char_count'] for c in chunks) / len(chunks)
        avg_words = sum(c['word_count'] for c in chunks) / len(chunks)
        print(f"Average chunk size: {avg_size:.0f} characters, {avg_words:.0f} words")

    base_name = md_path.stem.replace('_temp', '').replace('_text', '')
    output_dir = md_path.parent / "chunks"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{base_name}_chunks_recursive.md"

    save_chunks_as_markdown(chunks, output_file, method_name)

    return chunks

