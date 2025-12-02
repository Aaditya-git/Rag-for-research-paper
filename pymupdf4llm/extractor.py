import time
from pathlib import Path
from typing import Dict, Any
import fitz
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from chunking.text_chunker import chunk_markdown_file


class PyMuPDFExtractor:

    def __init__(self, output_dir: str = "outputs/pymupdf4llm"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, pdf_path: str, auto_chunk: bool = True, chunk_size: int = 1000, overlap: int = 200) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem
        start_time = time.time()

        print(f"\nProcessing {pdf_name} with PyMuPDF...")

        output_base = self.output_dir / pdf_name
        images_dir = output_base / f"{pdf_name}_images"
        tables_dir = output_base / f"{pdf_name}_tables"

        output_base.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        tables_dir.mkdir(exist_ok=True)

        # Extract markdown text only
        md_text = self._extract_markdown(pdf_path)

        # Get page count
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        # Image and table extraction disabled for now
        # image_count = self._extract_images(pdf_path, images_dir)
        # table_count = self._extract_tables(pdf_path, tables_dir)
        image_count = 0
        table_count = 0

        # Save temporary markdown file
        temp_md_file = output_base / f"{pdf_name}_temp.md"
        with open(temp_md_file, 'w', encoding='utf-8') as f:
            f.write(md_text)

        elapsed = time.time() - start_time

        print(f"Completed in {elapsed:.2f}s - Pages: {total_pages}, Images: {image_count}, Tables: {table_count}")

        # Auto-chunk if requested
        chunk_count = 0
        if auto_chunk:
            print("\nStarting chunking...")
            chunks = chunk_markdown_file(str(temp_md_file), chunk_size, overlap)
            chunk_count = len(chunks) if isinstance(chunks, list) else 0

        report = {
            "extractor": "pymupdf4llm",
            "pdf_name": pdf_name,
            "processing_time_seconds": round(elapsed, 2),
            "total_pages": total_pages,
            "total_text_length": len(md_text),
            "images_extracted": image_count,
            "tables_extracted": table_count,
            "chunks_created": chunk_count
        }

        return report

    def _extract_markdown(self, pdf_path: Path) -> str:
        doc = fitz.open(pdf_path)
        md_lines = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                md_lines.append(f"\n--- Page {page_num} ---\n")
                md_lines.append(text)

        doc.close()
        return "\n".join(md_lines)

    def _extract_structured_data(self, pdf_path: Path) -> Dict[str, Any]:
        doc = fitz.open(pdf_path)

        pages_data = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            blocks = page.get_text("dict")["blocks"]
            rect = page.rect

            page_info = {
                "page_number": page_num,
                "text": text,
                "text_length": len(text),
                "blocks": len(blocks),
                "width": rect.width,
                "height": rect.height
            }
            pages_data.append(page_info)

        doc.close()

        return {
            "total_pages": len(pages_data),
            "pages": pages_data
        }

    # Image and table extraction functions are intentionally left commented out
    # to focus only on text for now.

    # def _extract_images(self, pdf_path: Path, output_dir: Path) -> int:
    #     ...

    # def _extract_tables(self, pdf_path: Path, output_dir: Path) -> int:
    #     ...

