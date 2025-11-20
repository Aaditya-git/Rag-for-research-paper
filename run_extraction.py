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
            # Extract PDF (creates temp markdown file)
            report = extractor.extract(
                str(pdf_file),
                auto_chunk=False  # Don't auto-chunk yet
            )
            
            print(f"\nExtraction complete: {report['total_pages']} pages")
            
            # Now manually chunk with BOTH methods
            from chunking.text_chunker import chunk_markdown_file
            
            # Find the temp file that was created
            pdf_name = pdf_file.stem
            temp_md_file = f"outputs/pymupdf4llm/{pdf_name}/{pdf_name}_temp.md"
            
            if Path(temp_md_file).exists():
                print("\n--- Creating Recursive Chunks ---")
                recursive_chunks = chunk_markdown_file(
                    temp_md_file,
                    chunk_size=1000,
                    overlap=200,
                    use_semantic=False
                )
                
                print("\n--- Creating Semantic Chunks ---")
                semantic_chunks = chunk_markdown_file(
                    temp_md_file,
                    chunk_size=1000,
                    overlap=200,
                    use_semantic=True,
                    similarity_threshold=0.7
                )
                
                print(f"\nChunking complete:")
                print(f"  - Recursive: {len(recursive_chunks)} chunks")
                print(f"  - Semantic: {len(semantic_chunks)} chunks")
            else:
                print(f"Temp file not found: {temp_md_file}")
            
        except Exception as e:
            print(f"Failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print("EXTRACTION AND CHUNKING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total PDFs processed: {len(pdf_files)}")
    print("Check outputs/ folder for results\n")


if __name__ == "__main__":
    main()