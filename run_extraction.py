import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pymupdf4llm.extractor import PyMuPDFExtractor


def main():
    print("\nPDF Extraction and Chunking Pipeline")
    print("-" * 60)
    
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
            report = extractor.extract(
                str(pdf_file),
                auto_chunk=True,    
                chunk_size=1000,
                overlap=200
            )
            
            print(f"\nReport: {report['chunks_created']} chunks created")
            
        except Exception as e:
            print(f"Failed: {str(e)}")
    
    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
# ```

# ---

# ## **How it works:**

# 1. **Run:** `python run_extraction.py`

# 2. **Output structure:**
# ```
# outputs/pymupdf4llm/2309.11998v4/
# ├── 2309.11998v4_text.md          ← Extracted markdown
# ├── 2309.11998v4_images/          ← Extracted images
# ├── 2309.11998v4_tables/          ← Extracted tables
# └── chunks/
#     └── 2309.11998v4_text_chunks.md  ← Chunked markdown