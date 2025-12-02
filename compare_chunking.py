from pathlib import Path


def load_chunks_from_markdown(md_file):
    """Parse chunks from markdown file created by chunk_markdown_file."""
    chunks = []

    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    sections = content.split("## Chunk ")[1:]

    for section in sections:
        lines = section.split('\n')

        chunk_id = lines[0].strip()

        text_start = None
        for i, line in enumerate(lines):
            if line.strip() == "**Text:**":
                text_start = i + 2
                break

        if text_start is not None:
            text_lines = []
            for line in lines[text_start:]:
                if line.strip() == "---":
                    break
                text_lines.append(line)

            text = "\n".join(text_lines).strip()

            page_num = None
            for line in lines:
                if "Page Number:" in line:
                    try:
                        page_num = int(line.split(":")[-1].strip())
                    except Exception:
                        page_num = None

            if text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": text,
                        "page_number": page_num,
                        "char_count": len(text),
                    }
                )

    return chunks


def main():
    base_dir = Path("outputs/pymupdf4llm")
    chunk_files = list(base_dir.glob("**/chunks/*_chunks_recursive.md"))

    if not chunk_files:
        print("No chunk files found. Run run_extraction.py first.")
        return

    print(f"Found {len(chunk_files)} chunk file(s):\n")

    total_chunks = 0
    for cf in chunk_files:
        chunks = load_chunks_from_markdown(cf)
        print(f"{cf}: {len(chunks)} chunks")
        total_chunks += len(chunks)

    print(f"\nTotal chunks across all files: {total_chunks}")


if __name__ == "__main__":
    main()

