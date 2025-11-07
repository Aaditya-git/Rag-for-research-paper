import re
import json
import fitz
from pathlib import Path
from datetime import datetime


# ============= PDF EXTRACTION =============

def extract_pdf_to_markdown(pdf_path, output_dir=r"D:\study-code-repeat\coding\pdf-extraction-pipeline\outputs\pymupdf4llm"):
    """Extract PDF to markdown using PyMuPDF."""
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem
    
    print(f"\nExtracting PDF: {pdf_name}")
    print("-" * 60)
    
    output_base = Path(output_dir) / pdf_name
    output_base.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    md_lines = []
    
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        if text.strip():
            md_lines.append(f"\n--- Page {page_num} ---\n")
            md_lines.append(text)
    print(f"Extracted {len(doc)} pages")
    doc.close()
    
    md_text = "\n".join(md_lines)
    md_file = output_base / f"{pdf_name}_text.md"
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_text)
    
    
    print(f"Saved to: {md_file}")
    
    return str(md_file)


# ============= TEXT CHUNKING =============

def split_into_sentences(text):
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def calculate_chunk_relevance(chunk_text):
    score = 0.5
    length_factor = min(len(chunk_text) / 2000, 0.2)
    score += length_factor
    
    important_keywords = [
        'abstract', 'introduction', 'conclusion', 'summary',
        'methodology', 'results', 'discussion', 'key findings'
    ]
    
    chunk_lower = chunk_text.lower()
    for keyword in important_keywords:
        if keyword in chunk_lower:
            score += 0.1
    
    if 'table' in chunk_lower or 'figure' in chunk_lower:
        score += 0.15
    
    citation_patterns = [r'\[\d+\]', r'\(\d{4}\)', r'et al\.']
    for pattern in citation_patterns:
        if re.search(pattern, chunk_text):
            score += 0.05
    
    if re.search(r'\d+\.?\d*%|\d+\.?\d*', chunk_text):
        score += 0.1
    
    return min(score, 1.0)


def get_overlap_text(text, overlap_size):
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
                relevance_score = calculate_chunk_relevance(current_chunk)
                chunk_data = {
                    "chunk_id": len(chunks),
                    "text": current_chunk,
                    "char_count": len(current_chunk),
                    "char_start": char_position,
                    "char_end": char_position + len(current_chunk),
                    "relevance_score": round(relevance_score, 3),
                }
                
                if source_metadata:
                    chunk_data.update(source_metadata)
                
                chunks.append(chunk_data)
                
                char_position += len(current_chunk)
                overlap_text = get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                if len(sentence) > chunk_size:
                    sub_chunks = split_long_sentence(sentence, chunk_size)
                    for sub in sub_chunks[:-1]:
                        relevance_score = calculate_chunk_relevance(sub)
                        chunk_data = {
                            "chunk_id": len(chunks),
                            "text": sub,
                            "char_count": len(sub),
                            "char_start": char_position,
                            "char_end": char_position + len(sub),
                            "relevance_score": round(relevance_score, 3),
                        }
                        if source_metadata:
                            chunk_data.update(source_metadata)
                        chunks.append(chunk_data)
                        char_position += len(sub)
                    current_chunk = sub_chunks[-1]
                else:
                    current_chunk = sentence
    
    if current_chunk:
        relevance_score = calculate_chunk_relevance(current_chunk)
        chunk_data = {
            "chunk_id": len(chunks),
            "text": current_chunk,
            "char_count": len(current_chunk),
            "char_start": char_position,
            "char_end": char_position + len(current_chunk),
            "relevance_score": round(relevance_score, 3),
        }
        if source_metadata:
            chunk_data.update(source_metadata)
        chunks.append(chunk_data)
    
    return chunks


def save_chunks_as_markdown(chunks, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Chunked Document\n\n")
        f.write(f"**Total Chunks:** {len(chunks)}\n\n")
        f.write("---\n\n")
        
        for chunk in chunks:
            f.write(f"## Chunk {chunk['chunk_id']}\n\n")
            f.write("**Metadata:**\n")
            f.write(f"- Relevance Score: {chunk['relevance_score']}\n")
            f.write(f"- Character Count: {chunk['char_count']}\n")
            f.write(f"- Position: {chunk['char_start']}-{chunk['char_end']}\n")
            
            if 'source_document' in chunk:
                f.write(f"- Source Document: {chunk['source_document']}\n")
            if 'source_path' in chunk:
                f.write(f"- Source Path: {chunk['source_path']}\n")
            
            f.write("\n**Text:**\n\n")
            f.write(f"{chunk['text']}\n\n")
            f.write("---\n\n")
    
    print(f"Saved markdown chunks: {output_file}")


def chunk_markdown_file(md_file, chunk_size=1000, overlap=200):
    """Chunk a markdown file and save as markdown."""
    md_path = Path(md_file)
    
    print(f"\nChunking: {md_path.name}")
    print("-" * 60)
    
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"File size: {len(text):,} characters")
    
    source_metadata = {
        "source_document": md_path.name,
        "source_path": str(md_path.absolute()),
    }
    
    chunks = chunk_text(text, chunk_size, overlap, source_metadata)
    
    print(f"Created {len(chunks)} chunks")
    
    if chunks:
        avg_size = sum(c['char_count'] for c in chunks) / len(chunks)
        avg_relevance = sum(c['relevance_score'] for c in chunks) / len(chunks)
        print(f"Average chunk size: {avg_size:.0f} characters")
        print(f"Average relevance score: {avg_relevance:.3f}")
    
    output_dir = md_path.parent / "chunks"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{md_path.stem}_chunks.md"
    
    save_chunks_as_markdown(chunks, output_file)
    
    return chunks


# ============= MAIN PIPELINE =============

def process_pdf(pdf_path, chunk_size=1000, overlap=200):
    """
    Complete pipeline: Extract PDF to markdown, then chunk it.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks
    """
    print("\n" + "=" * 60)
    print("PDF TO CHUNKED MARKDOWN PIPELINE")
    print("=" * 60)
    
    # Step 1: Extract PDF to markdown
    md_file = extract_pdf_to_markdown(pdf_path)
    
    # Step 2: Chunk the markdown
    chunks = chunk_markdown_file(md_file, chunk_size, overlap)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Output: {Path(md_file).parent / 'chunks'}")
    
    return chunks


if __name__ == "__main__":
    # Process your PDF
    pdf_file = r"D:\study-code-repeat\coding\pdf-extraction-pipeline\pdfs\2309.11998v4.pdf"
    
    if Path(pdf_file).exists():
        process_pdf(
            pdf_path=pdf_file,
            chunk_size=1000,
            overlap=200
        )
    else:
        print(f"PDF not found: {pdf_file}")
        print("Please place your PDF in the pdfs/ folder")