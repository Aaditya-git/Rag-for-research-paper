import os
import json
import time
from pathlib import Path
from typing import Dict, Any
import fitz


class PyMuPDFExtractor:
    
    def __init__(self, output_dir: str = "outputs/pymupdf4llm"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
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
        
        md_text = self._extract_markdown(pdf_path)
        json_data = self._extract_structured_data(pdf_path)
        image_count = self._extract_images(pdf_path, images_dir)
        table_count = self._extract_tables(pdf_path, tables_dir)
        
        md_file = output_base / f"{pdf_name}_text.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_text)
        
        json_file = output_base / f"{pdf_name}_struct.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - start_time
        report = {
            "extractor": "pymupdf4llm",
            "pdf_name": pdf_name,
            "processing_time_seconds": round(elapsed, 2),
            "total_pages": json_data['total_pages'],
            "total_text_length": len(md_text),
            "images_extracted": image_count,
            "tables_extracted": table_count,
            "output_files": {
                "markdown": str(md_file.name),
                "json": str(json_file.name),
                "images_dir": str(images_dir.name),
                "tables_dir": str(tables_dir.name)
            }
        }
        
        report_file = output_base / f"{pdf_name}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"Completed in {elapsed:.2f}s - Pages: {json_data['total_pages']}, Images: {image_count}, Tables: {table_count}")
        
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
    
    def _extract_images(self, pdf_path: Path, output_dir: Path) -> int:
        doc = fitz.open(pdf_path)
        image_count = 0
        
        for page_num, page in enumerate(doc, 1):
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    image_filename = output_dir / f"page{page_num}_img{img_index + 1}.{image_ext}"
                    with open(image_filename, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    image_count += 1
                except Exception as e:
                    continue
        
        doc.close()
        return image_count
    
    def _extract_tables(self, pdf_path: Path, output_dir: Path) -> int:
        doc = fitz.open(pdf_path)
        table_count = 0
        
        for page_num, page in enumerate(doc, 1):
            try:
                tables = page.find_tables()
                
                if tables:
                    for table_index, table in enumerate(tables):
                        try:
                            table_data = table.extract()
                            table_file = output_dir / f"page{page_num}_table{table_index + 1}.json"
                            
                            with open(table_file, 'w', encoding='utf-8') as f:
                                json.dump({
                                    "page": page_num,
                                    "table_index": table_index + 1,
                                    "rows": len(table_data) if table_data else 0,
                                    "data": table_data
                                }, f, indent=2, ensure_ascii=False)
                            
                            table_count += 1
                        except Exception as e:
                            continue
            except Exception as e:
                continue
        
        doc.close()
        return table_count