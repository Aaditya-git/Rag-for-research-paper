import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json


class UnstructuredExtractor:
    
    def __init__(self, output_dir: str = "outputs/unstructured"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem
        start_time = time.time()
        
        print(f"\nProcessing {pdf_name} with Unstructured...")
        
        output_base = self.output_dir / pdf_name
        images_dir = output_base / f"{pdf_name}_images"
        tables_dir = output_base / f"{pdf_name}_tables"
        
        output_base.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        tables_dir.mkdir(exist_ok=True)
        
        elements = partition_pdf(
            filename=str(pdf_path),
            extract_images_in_pdf=True,
            extract_image_block_output_dir=str(images_dir),
            infer_table_structure=True,
            strategy="hi_res"
        )
        
        md_text = self._elements_to_markdown(elements)
        json_data = self._extract_structured_data(elements)
        table_count = self._extract_tables(elements, tables_dir)
        
        md_file = output_base / f"{pdf_name}_text.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_text)
        
        json_file = output_base / f"{pdf_name}_struct.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        image_count = len(list(images_dir.glob("*")))
        
        elapsed = time.time() - start_time
        report = {
            "extractor": "unstructured",
            "pdf_name": pdf_name,
            "processing_time_seconds": round(elapsed, 2),
            "total_elements": len(elements),
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
        
        print(f"Completed in {elapsed:.2f}s - Elements: {len(elements)}, Images: {image_count}, Tables: {table_count}")
        
        return report
    
    def _elements_to_markdown(self, elements: List) -> str:
        md_lines = []
        
        for elem in elements:
            elem_type = type(elem).__name__
            text = str(elem)
            
            if elem_type == "Title":
                md_lines.append(f"\n# {text}\n")
            elif elem_type == "NarrativeText":
                md_lines.append(f"{text}\n")
            elif elem_type == "ListItem":
                md_lines.append(f"- {text}")
            elif elem_type == "Table":
                md_lines.append(f"\n{text}\n")
            else:
                md_lines.append(text)
        
        return "\n".join(md_lines)
    
    def _extract_structured_data(self, elements: List) -> Dict[str, Any]:
        structured = {
            "total_elements": len(elements),
            "elements": []
        }
        
        for i, elem in enumerate(elements):
            elem_data = {
                "index": i,
                "type": type(elem).__name__,
                "text": str(elem),
                "metadata": elem.metadata.to_dict() if hasattr(elem, 'metadata') else {}
            }
            structured["elements"].append(elem_data)
        
        return structured
    
    def _extract_tables(self, elements: List, output_dir: Path) -> int:
        table_count = 0
        
        for i, elem in enumerate(elements):
            if type(elem).__name__ == "Table":
                table_file = output_dir / f"table_{i + 1}.json"
                
                table_data = {
                    "index": i,
                    "text": str(elem),
                    "html": elem.metadata.text_as_html if hasattr(elem.metadata, 'text_as_html') else None
                }
                
                with open(table_file, 'w', encoding='utf-8') as f:
                    json.dump(table_data, f, indent=2, ensure_ascii=False)
                
                table_count += 1
        
        return table_count