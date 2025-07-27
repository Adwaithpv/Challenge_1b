#!/usr/bin/env python3
"""
Wrapper script that runs Round 1A solution and automatically post-processes the output.

This script combines the PDF analysis with automatic JSON formatting to ensure
the output matches the required schema with proper indentation.
"""

import sys
import json
import tempfile
import subprocess
from pathlib import Path
from post_process_output import JSONPostProcessor


def run_solution_with_post_processing(pdf_path: str, output_path: str = None, indent: int = 2):
    """
    Run the Round 1A solution and post-process the output.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path for the final formatted output (optional)
        indent: Number of spaces for JSON indentation
    """
    # Create a temporary file for the raw output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        temp_output_path = temp_file.name
    
    try:
        # Run the main solution
        print(f"Processing PDF: {pdf_path}")
        cmd = [sys.executable, "round1a_solution.py", pdf_path, "-o", temp_output_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running solution: {result.stderr}")
            return False
        
        print("PDF processing completed successfully")
        
        # Post-process the output
        print("Post-processing JSON output...")
        processor = JSONPostProcessor()
        
        formatted_json = processor.process_file(temp_output_path, output_path, indent)
        
        if not output_path:
            print("\n" + "="*50)
            print("FORMATTED OUTPUT:")
            print("="*50)
            print(formatted_json)
        
        print(f"\nProcessing completed successfully!")
        if output_path:
            print(f"Formatted output saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    finally:
        # Clean up temporary file
        try:
            Path(temp_output_path).unlink()
        except:
            pass


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_with_post_processing.py <pdf_file> [output_file]")
        print("Example: python run_with_post_processing.py document.pdf output.json")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Validate PDF file
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Run the solution with post-processing
    success = run_solution_with_post_processing(pdf_path, output_path)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 