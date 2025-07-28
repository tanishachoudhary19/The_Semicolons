"""
Document processing module for extracting text and structure from PDFs
"""

import fitz  # PyMuPDF
import re
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.font_size_threshold = 2.0  # Minimum font size difference for heading detection
        
    def process_pdf(self, pdf_path: str) -> Dict:
        """Extract text, structure, and metadata from PDF."""
        try:
            doc = fitz.open(pdf_path)
            
            # Extract page-wise content
            pages_content = []
            font_sizes = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text with font information
                blocks = page.get_text("dict")
                page_content = self._extract_page_content(blocks, page_num + 1)
                pages_content.append(page_content)
                
                # Collect font sizes for heading detection
                page_fonts = self._extract_font_sizes(blocks)
                font_sizes.extend(page_fonts)
                
            doc.close()
            
            # Determine heading thresholds
            heading_thresholds = self._calculate_heading_thresholds(font_sizes)
            
            return {
                'pages': pages_content,
                'heading_thresholds': heading_thresholds,
                'total_pages': len(pages_content)
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return {'pages': [], 'heading_thresholds': {}, 'total_pages': 0}
    
    def _extract_page_content(self, blocks: Dict, page_num: int) -> Dict:
        """Extract structured content from a page."""
        text_elements = []
        
        for block in blocks.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            text_elements.append({
                                'text': text,
                                'font_size': span.get("size", 12),
                                'font_flags': span.get("flags", 0),
                                'bbox': span.get("bbox", [0, 0, 0, 0])
                            })
        
        return {
            'page_number': page_num,
            'elements': text_elements,
            'full_text': ' '.join([elem['text'] for elem in text_elements])
        }
    
    def _extract_font_sizes(self, blocks: Dict) -> List[float]:
        """Extract all font sizes from page blocks."""
        font_sizes = []
        
        for block in blocks.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span.get("text", "").strip():
                            font_sizes.append(span.get("size", 12))
        
        return font_sizes
    
    def _calculate_heading_thresholds(self, font_sizes: List[float]) -> Dict[str, float]:
        """Calculate font size thresholds for different heading levels."""
        if not font_sizes:
            return {"h1": 16, "h2": 14, "h3": 12, "body": 11}
        
        # Sort font sizes and find common sizes
        sorted_sizes = sorted(set(font_sizes), reverse=True)
        
        # Assume largest fonts are headings
        thresholds = {"body": min(font_sizes)}
        
        if len(sorted_sizes) >= 4:
            thresholds.update({
                "h1": sorted_sizes[0],
                "h2": sorted_sizes[1], 
                "h3": sorted_sizes[2]
            })
        elif len(sorted_sizes) >= 2:
            thresholds.update({
                "h1": sorted_sizes[0],
                "h2": sorted_sizes[1],
                "h3": sorted_sizes[1]
            })
        else:
            # Fallback to standard sizes
            avg_size = sum(font_sizes) / len(font_sizes)
            thresholds.update({
                "h1": avg_size + 4,
                "h2": avg_size + 2,
                "h3": avg_size + 1
            })
        
        return thresholds
