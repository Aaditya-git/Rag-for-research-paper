import os
import json
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
        
        # Extract markdown text
        md_text = self._extract_markdown(pdf_path)
        
        # Get page count
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        # Extract images and tables
        image_count = self._extract_images(pdf_path, images_dir)
        table_count = self._extract_tables(pdf_path, tables_dir)
        
        # Save temporary markdown file (DON'T DELETE IT)
        temp_md_file = output_base / f"{pdf_name}_temp.md"
        with open(temp_md_file, 'w', encoding='utf-8') as f:
            f.write(md_text)
        
        elapsed = time.time() - start_time
        
        print(f"Completed in {elapsed:.2f}s - Pages: {total_pages}, Images: {image_count}, Tables: {table_count}")
        
        # Auto-chunk if requested (optional, now controlled from run_extraction.py)
        chunk_count = 0
        if auto_chunk:
            print("\nStarting chunking...")
            from chunking.text_chunker import chunk_markdown_file
            chunks = chunk_markdown_file(str(temp_md_file), chunk_size, overlap)
            chunk_count = len(chunks) if isinstance(chunks, list) else chunks.get('total', 0)
            # DON'T delete temp file anymore
            # temp_md_file.unlink()
        
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
    
    # def _extract_images(self, pdf_path: Path, output_dir: Path) -> int:
    #     doc = fitz.open(pdf_path)
    #     image_count = 0
        
    #     for page_num, page in enumerate(doc, 1):
    #         image_list = page.get_images(full=True)
            
    #         for img_index, img in enumerate(image_list):
    #             xref = img[0]
    #             try:
    #                 base_image = doc.extract_image(xref)
    #                 image_bytes = base_image["image"]
    #                 image_ext = base_image["ext"]
                    
    #                 image_filename = output_dir / f"page{page_num}_img{img_index + 1}.{image_ext}"
    #                 with open(image_filename, "wb") as img_file:
    #                     img_file.write(image_bytes)
                    
    #                 image_count += 1
    #             except:
    #                 continue
        
    #     doc.close()
    #     return image_count
    
    # def _extract_tables(self, pdf_path: Path, output_dir: Path) -> int:
    #     """Extract tables and save as MARKDOWN format."""
    #     doc = fitz.open(pdf_path)
    #     table_count = 0
        
    #     for page_num, page in enumerate(doc, 1):
    #         try:
    #             tables = page.find_tables()
                
    #             if tables:
    #                 for table_index, table in enumerate(tables):
    #                     try:
    #                         table_data = table.extract()
                            
    #                         # Save as markdown instead of JSON
    #                         table_file = output_dir / f"page{page_num}_table{table_index + 1}.md"
                            
    #                         with open(table_file, 'w', encoding='utf-8') as f:
    #                             f.write(f"# Table from Page {page_num}\n\n")
    #                             f.write(f"**Table Index:** {table_index + 1}\n\n")
    #                             f.write(f"**Rows:** {len(table_data) if table_data else 0}\n\n")
    #                             f.write("---\n\n")
                                
    #                             # Convert to markdown table
    #                             if table_data and len(table_data) > 0:
    #                                 # Header row
    #                                 if table_data[0]:
    #                                     f.write("| " + " | ".join(str(cell) if cell else "" for cell in table_data[0]) + " |\n")
    #                                     f.write("|" + "|".join([" --- " for _ in table_data[0]]) + "|\n")
                                    
    #                                 # Data rows
    #                                 for row in table_data[1:]:
    #                                     if row:
    #                                         f.write("| " + " | ".join(str(cell) if cell else "" for cell in row) + " |\n")
                            
    #                         table_count += 1
    #                     except:
    #                         continue
    #         except:
    #             continue
        
    #     doc.close()
    #     return table_count