import os
import json
import time
from pathlib import Path
from typing import Dict, Any
from docling.document_converter import DocumentConverter


class DoclingExtractor:
    
    def __init__(self, output_dir: str = "outputs/docling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.converter = DocumentConverter()
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem
        start_time = time.time()
        
        print(f"\nProcessing {pdf_name} with Docling...")
        
        output_base = self.output_dir / pdf_name
        images_dir = output_base / f"{pdf_name}_images"
        tables_dir = output_base / f"{pdf_name}_tables"
        
        output_base.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        tables_dir.mkdir(exist_ok=True)
        
        result = self.converter.convert(str(pdf_path))
        
        md_text = result.document.export_to_markdown()
        json_data = self._extract_structured_data(result)
        image_count = self._extract_images(result, images_dir)
        table_count = self._extract_tables(result, tables_dir)
        
        md_file = output_base / f"{pdf_name}_text.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_text)
        
        json_file = output_base / f"{pdf_name}_struct.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - start_time
        report = {
            "extractor": "docling",
            "pdf_name": pdf_name,
            "processing_time_seconds": round(elapsed, 2),
            "total_text_length": len(md_text),
            "images_extracted": image_count,
            "tables_extracted": table_count,
            "output_files": {
                "markdown": str(md_file.relative_to(self.output_dir.parent)),
                "json": str(json_file.relative_to(self.output_dir.parent)),
                "images_dir": str(images_dir.relative_to(self.output_dir.parent)),
                "tables_dir": str(tables_dir.relative_to(self.output_dir.parent))
            }
        }
        
        report_file = output_base / f"{pdf_name}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"Completed in {elapsed:.2f}s - Images: {image_count}, Tables: {table_count}")
        
        return report
    
    def _extract_structured_data(self, result) -> Dict[str, Any]:
        doc_dict = result.document.export_to_dict()
        return doc_dict
    
    def _extract_images(self, result, output_dir: Path) -> int:
        image_count = 0
        
        if hasattr(result.document, 'pictures'):
            for i, picture in enumerate(result.document.pictures):
                try:
                    image_file = output_dir / f"image_{i + 1}.png"
                    picture.get_image().save(image_file)
                    image_count += 1
                except:
                    pass
        
        return image_count
    
    def _extract_tables(self, result, output_dir: Path) -> int:
        table_count = 0
        
        if hasattr(result.document, 'tables'):
            for i, table in enumerate(result.document.tables):
                try:
                    table_file = output_dir / f"table_{i + 1}.json"
                    
                    table_data = {
                        "index": i,
                        "data": table.export_to_dataframe().to_dict() if hasattr(table, 'export_to_dataframe') else str(table)
                    }
                    
                    with open(table_file, 'w', encoding='utf-8') as f:
                        json.dump(table_data, f, indent=2, ensure_ascii=False)
                    
                    table_count += 1
                except:
                    pass
        
        return table_count