import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from pymupdf4llm.extractor import PyMuPDFExtractor


def main():
    print("=" * 60)
    # Find PDFs
    pdf_dir = Path("pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("Please add PDF files to the pdfs/ folder and try again")
        return
    
    print(f"Found {len(pdf_files)} PDF(s) to process\n")

    extractors = {
        "pymupdf4llm": PyMuPDFExtractor(), 
    }

    all_reports = []
    
    for pdf_file in pdf_files:
        print(f"\n{'#' * 60}")
        print(f"Processing: {pdf_file.name}")
        print(f"{'#' * 60}")
        
        for name, extractor in extractors.items():
            try:
                report = extractor.extract(str(pdf_file))
                all_reports.append(report)
            except Exception as e:
                print(f"{name} failed: {e}")
    

    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Total extractions: {len(all_reports)}")


if __name__ == "__main__":
    main()