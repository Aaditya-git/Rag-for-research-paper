import re
from pathlib import Path


def split_into_sentences(text):
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


def create_text_chunk_metadata(chunk_text, chunk_id, char_start, char_end, source_metadata, page_number=None):
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


def create_image_chunk_metadata(image_file, chunk_id, source_metadata):
    """Create metadata for an image chunk."""
    
    page_match = re.search(r'page(\d+)', image_file.name)
    page_num = int(page_match.group(1)) if page_match else None
    
    img_match = re.search(r'img(\d+)', image_file.name)
    img_index = int(img_match.group(1)) if img_match else None
    
    metadata = {
        "chunk_id": chunk_id,
        "type": "image",
        "image_path": str(image_file),
        "image_name": image_file.name,
        "image_format": image_file.suffix.lower().replace('.', ''),
        "file_size_bytes": image_file.stat().st_size,
        "page_number": page_num,
        "image_index_on_page": img_index,
        "description": f"Image {img_index} from page {page_num}",
    }
    
    if source_metadata:
        metadata.update(source_metadata)
    
    return metadata


def create_table_chunk_metadata(table_file, table_content, chunk_id, source_metadata):
    """Create metadata for a table chunk."""
    
    page_match = re.search(r'page(\d+)', table_file.name)
    page_num = int(page_match.group(1)) if page_match else None
    
    table_match = re.search(r'table(\d+)', table_file.name)
    table_index = int(table_match.group(1)) if table_match else None
    
    table_lines = [line for line in table_content.split('\n') if '|' in line]
    row_count = len(table_lines) - 2 if len(table_lines) > 2 else 0
    col_count = len(table_lines[0].split('|')) - 2 if table_lines else 0
    
    headers = []
    if len(table_lines) > 0:
        headers = [cell.strip() for cell in table_lines[0].split('|')[1:-1]]
    
    metadata = {
        "chunk_id": chunk_id,
        "type": "table",
        "text": table_content,
        "table_file": table_file.name,
        "row_count": row_count,
        "column_count": col_count,
        "headers": headers,
        "page_number": page_num,
        "table_index_on_page": table_index,
        "description": f"Table {table_index} from page {page_num} with {row_count} rows and {col_count} columns",
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


def chunk_text(text, chunk_size=1000, overlap=200, source_metadata=None):
    """Chunk text respecting sentence boundaries with rich metadata."""
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
                    page_number=page_num
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
                            page_number=page_num
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
            page_number=page_num
        )
        chunks.append(chunk_data)
    
    return chunks


def chunk_images(images_dir, source_metadata=None):
    """Create chunks for each image with metadata."""
    images_path = Path(images_dir)
    
    if not images_path.exists():
        return []
    
    image_chunks = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    
    for image_file in sorted(images_path.iterdir()):
        if image_file.suffix.lower() in image_extensions:
            chunk_data = create_image_chunk_metadata(
                image_file=image_file,
                chunk_id=f"image_{len(image_chunks)}",
                source_metadata=source_metadata
            )
            image_chunks.append(chunk_data)
    
    return image_chunks


def chunk_tables(tables_dir, source_metadata=None):
    """Read markdown table files and create chunks."""
    tables_path = Path(tables_dir)
    
    if not tables_path.exists():
        return []
    
    table_chunks = []
    
    for table_file in sorted(tables_path.glob("*.md")):
        with open(table_file, 'r', encoding='utf-8') as f:
            table_content = f.read()
        
        chunk_data = create_table_chunk_metadata(
            table_file=table_file,
            table_content=table_content,
            chunk_id=f"table_{len(table_chunks)}",
            source_metadata=source_metadata
        )
        table_chunks.append(chunk_data)
    
    return table_chunks


def save_all_chunks_as_markdown(text_chunks, image_chunks, table_chunks, output_file):
    """Save all chunk types as markdown with rich metadata."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_chunks = len(text_chunks) + len(image_chunks) + len(table_chunks)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Chunked Document (Text + Images + Tables)\n\n")
        f.write(f"**Total Chunks:** {total_chunks}\n")
        f.write(f"- Text Chunks: {len(text_chunks)}\n")
        f.write(f"- Image Chunks: {len(image_chunks)}\n")
        f.write(f"- Table Chunks: {len(table_chunks)}\n\n")
        f.write("---\n\n")
        
        # Text chunks
        if text_chunks:
            f.write("# Text Chunks\n\n")
            for chunk in text_chunks:
                f.write(f"## Chunk {chunk['chunk_id']} (Text)\n\n")
                f.write("**Metadata:**\n")
                f.write(f"- Type: {chunk['type']}\n")
                f.write(f"- Page Number: {chunk.get('page_number', 'N/A')}\n")
                f.write(f"- Section Type: {chunk['section_type']}\n")
                f.write(f"- Character Count: {chunk['char_count']}\n")
                f.write(f"- Word Count: {chunk['word_count']}\n")
                f.write(f"- Sentence Count: {chunk['sentence_count']}\n")
                f.write(f"- Position: {chunk['char_start']}-{chunk['char_end']}\n")
                f.write(f"- Contains Code: {chunk['contains_code']}\n")
                f.write(f"- Contains Math: {chunk['contains_math']}\n")
                f.write(f"- Contains Citation: {chunk['contains_citation']}\n")
                f.write(f"- Contains List: {chunk['contains_list']}\n")
                
                if 'source_document' in chunk:
                    f.write(f"- Source Document: {chunk['source_document']}\n")
                
                f.write("\n**Text:**\n\n")
                f.write(f"{chunk['text']}\n\n")
                f.write("---\n\n")
        
        # Image chunks
        if image_chunks:
            f.write("\n# Image Chunks\n\n")
            for chunk in image_chunks:
                f.write(f"## {chunk['chunk_id']} (Image)\n\n")
                f.write("**Metadata:**\n")
                f.write(f"- Type: {chunk['type']}\n")
                f.write(f"- Description: {chunk['description']}\n")
                f.write(f"- Image Name: {chunk['image_name']}\n")
                f.write(f"- Image Path: {chunk['image_path']}\n")
                f.write(f"- Image Format: {chunk['image_format']}\n")
                f.write(f"- Page Number: {chunk['page_number']}\n")
                f.write(f"- Image Index on Page: {chunk['image_index_on_page']}\n")
                f.write(f"- File Size: {chunk['file_size_bytes']} bytes\n")
                f.write("\n---\n\n")
        
        # Table chunks
        if table_chunks:
            f.write("\n# Table Chunks\n\n")
            for chunk in table_chunks:
                f.write(f"## {chunk['chunk_id']} (Table)\n\n")
                f.write("**Metadata:**\n")
                f.write(f"- Type: {chunk['type']}\n")
                f.write(f"- Description: {chunk['description']}\n")
                f.write(f"- Page Number: {chunk['page_number']}\n")
                f.write(f"- Table Index on Page: {chunk['table_index_on_page']}\n")
                f.write(f"- Row Count: {chunk['row_count']}\n")
                f.write(f"- Column Count: {chunk['column_count']}\n")
                f.write(f"- Headers: {', '.join(chunk['headers']) if chunk['headers'] else 'N/A'}\n")
                f.write("\n**Table Content:**\n\n")
                f.write(f"{chunk['text']}\n\n")
                f.write("---\n\n")
    
    print(f"Saved {total_chunks} chunks to: {output_file}")


def chunk_markdown_file(md_file, chunk_size=1000, overlap=200):
    """Chunk a markdown file along with its images and tables."""
    md_path = Path(md_file)
    
    print(f"Chunking: {md_path.name}")
    
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"File size: {len(text):,} characters")
    
    source_metadata = {
        "source_document": md_path.name,
        "source_path": str(md_path.absolute()),
    }
    
    text_chunks = chunk_text(text, chunk_size, overlap, source_metadata)
    print(f"Created {len(text_chunks)} text chunks")
    
    base_name = md_path.stem.replace('_temp', '').replace('_text', '')
    images_dir = md_path.parent / f"{base_name}_images"
    image_chunks = chunk_images(images_dir, source_metadata)       # NEED TO WORK ON !!
    print(f"Created {len(image_chunks)} image chunks")
    
    tables_dir = md_path.parent / f"{base_name}_tables"
    table_chunks = chunk_tables(tables_dir, source_metadata)
    print(f"Created {len(table_chunks)} table chunks")
    
    if text_chunks:
        avg_size = sum(c['char_count'] for c in text_chunks) / len(text_chunks)
        avg_words = sum(c['word_count'] for c in text_chunks) / len(text_chunks)
        print(f"Average text chunk size: {avg_size:.0f} characters, {avg_words:.0f} words")
    
    output_dir = md_path.parent / "chunks"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{base_name}_chunks.md"
    
    save_all_chunks_as_markdown(text_chunks, image_chunks, table_chunks, output_file)
    
    all_chunks = {
        "text_chunks": text_chunks,
        "image_chunks": image_chunks,
        "table_chunks": table_chunks,
        "total": len(text_chunks) + len(image_chunks) + len(table_chunks)
    }
    
    return all_chunks