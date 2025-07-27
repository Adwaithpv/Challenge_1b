#!/usr/bin/env python3
"""
Round 1A Solution: Adobe India Hackathon
Extracts structured outlines from PDF files.

Requirements:
- Extract document title and hierarchical headings (H1, H2, H3, etc.)
- Output structured JSON with text content and page numbers  
- Process 50 pages in <10 seconds, stay within 16GB RAM
- Work offline without internet access
"""

import sys
import json
import time
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any

# Force CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow logging

# Add src to path for imports
sys.path.append('src')

from pdf_layout_analysis.run_pdf_layout_analysis_fast import analyze_pdf_fast
from toc.extract_table_of_contents import extract_table_of_contents


class Round1ASolutionEngine:
    """Optimized engine for Round 1A document structure extraction."""
    
    def __init__(self):
        self.max_processing_time = 10.0  # seconds
        self.start_time = None
        
    def extract_document_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract document title and hierarchical headings from PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with title and hierarchical headings
        """
        self.start_time = time.time()
        
        # Step 1: Analyze PDF layout to get segments
        with open(pdf_path, 'rb') as file:
            pdf_content = file.read()
        
        # Use existing fast layout analysis 
        segment_boxes = analyze_pdf_fast(pdf_content, "temp_analysis", keep_pdf=False)
        
        # Step 2: Extract TOC using existing system
        toc_data = extract_table_of_contents(pdf_content, segment_boxes, skip_document_name=False)
        
        # Step 3: Convert to Round 1A format
        result = self._convert_to_round1a_format(toc_data)
        
        processing_time = time.time() - self.start_time
        
        if processing_time > self.max_processing_time:
            print(f"Warning: Processing took {processing_time:.2f}s (target: <{self.max_processing_time}s)")
        
        return result
    
    def _convert_to_round1a_format(self, toc_data: List[Dict]) -> Dict[str, Any]:
        """
        Convert TOC data to Round 1A required JSON format.
        
        Args:
            toc_data: TOC data from existing extraction system
            
        Returns:
            Structured JSON with title and hierarchical headings
        """
        if not toc_data:
            return {
                "title": "",
                "outline": []
            }
        
        # Find the document title (usually the first meaningful heading)
        title = ""
        outline = []
        
        for item in toc_data:
            indentation = item.get("indentation", 0)
            text = item.get("label", "").strip()
            page = int(item.get("bounding_box", {}).get("page", 1))
            
            if not text:
                continue
            
            # Skip very short text that might be artifacts
            if len(text) <= 2:
                continue
                
            # If we haven't found a title yet and this looks like a main title
            if not title and indentation <= 1 and len(text) > 5:
                # Check if this looks like a document title (not a section heading)
                if not any(keyword in text.upper() for keyword in ["PART", "CHAPTER", "SECTION"]):
                    title = text
                    continue
            
            # Convert indentation to heading level (H1, H2, H3, etc.)
            # Better mapping based on typical document structure:
            # - Indentation 0-1: Main sections (H1)
            # - Indentation 2: Subsections (H2) 
            # - Indentation 3+: Sub-subsections (H3+)
            if indentation <= 1:
                heading_level = 1  # H1 for main sections
            elif indentation == 2:
                heading_level = 2  # H2 for subsections
            else:
                heading_level = 3  # H3+ for deeper levels
                
            outline.append({
                "level": f"H{heading_level}",
                "text": text,
                "page": page
            })
        
        # If we still don't have a title, use the first meaningful heading
        if not title and outline:
            title = outline[0]["text"]
            outline = outline[1:]  # Remove it from outline since it's now the title
        
        return {
            "title": title,
            "outline": outline
        }


def main():
    """Main entry point for Round 1A solution."""
    parser = argparse.ArgumentParser(
        description="Round 1A: Extract structured outlines from PDF files"
    )
    parser.add_argument("pdf_file", help="Path to PDF file")
    parser.add_argument(
        "-o", "--output", 
        help="Output JSON file (default: print to stdout)"
    )
    parser.add_argument(
        "--pretty", 
        action="store_true",
        help="Pretty print JSON output"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
        
    if not pdf_path.suffix.lower() == '.pdf':
        print(f"Error: File must be a PDF: {pdf_path}")
        sys.exit(1)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Extract document structure
        engine = Round1ASolutionEngine()
        result = engine.extract_document_structure(str(pdf_path))
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add timing information to result
        if isinstance(result, dict):
            result["processing_time_seconds"] = round(processing_time, 2)
        
        # Post-process and format JSON output
        from post_process_output import JSONPostProcessor
        
        processor = JSONPostProcessor()
        processed_result = processor.validate_and_fix_structure(result)
        
        # Format JSON output
        json_kwargs = {"indent": 2} if args.pretty else {}
        json_output = processor.format_json(processed_result, json_kwargs.get("indent", 2))
        
        # Output result
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json_output, encoding='utf-8')
            print(f"Results saved to: {output_path}")
        else:
            print(json_output)
            
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 