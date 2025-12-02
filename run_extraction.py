import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pymupdf4llm.extractor import PyMuPDFExtractor
from chunking.text_chunker import chunk_markdown_file


def main():
    print("\nPDF Extraction and Recursive Chunking")
    print("-" * 60)
    print("test")
    pdf_dir = Path("pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in pdfs/ folder")
        return

    print(f"Found {len(pdf_files)} PDF file(s)\n")

    extractor = PyMuPDFExtractor()

    for pdf_file in pdf_files:
        print(f"\n{'=' * 60}")
        print(f"File: {pdf_file.name}")
        print('=' * 60)

        try:
            # We call extract with auto_chunk=False so we control chunking here
            report = extractor.extract(str(pdf_file), auto_chunk=False)
            print(f"Extraction complete: {report['total_pages']} pages")

            pdf_name = pdf_file.stem
            temp_md_file = f"outputs/pymupdf4llm/{pdf_name}/{pdf_name}_temp.md"

            if Path(temp_md_file).exists():
                print("\nCreating Recursive Chunks...")
                chunks = chunk_markdown_file(
                    temp_md_file,
                    chunk_size=1000,
                    overlap=200
                )
                print(f"Created {len(chunks)} chunks")
            else:
                print(f"Temp file not found: {temp_md_file}")

        except Exception as e:
            print(f"Failed: {str(e)}")

    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

