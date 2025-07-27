#!/usr/bin/env python3
"""
Docker entrypoint script for the Adobe India Hackathon Round 1A solution.

This script automatically processes all PDF files from /app/input directory
and generates corresponding filename.json files in /app/output directory.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.append('/app/src')
sys.path.append('/app')

# Force CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from round1a_solution_optimized import OptimizedRound1ASolutionEngine
from post_process_output import JSONPostProcessor


def find_pdf_files(input_dir: Path) -> List[Path]:
    """Find all PDF files in the input directory."""
    pdf_files = []
    
    if not input_dir.exists():
        print(f"⚠️  Input directory does not exist: {input_dir}")
        return pdf_files
    
    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == '.pdf':
            pdf_files.append(file_path)
    
    return sorted(pdf_files)


def process_pdf_file(pdf_path: Path, output_dir: Path, engine: OptimizedRound1ASolutionEngine, 
                    processor: JSONPostProcessor) -> bool:
    """Process a single PDF file and save the JSON output."""
    try:
        # Generate output filename
        output_filename = pdf_path.stem + '.json'
        output_path = output_dir / output_filename
        
        print(f"📄 Processing: {pdf_path.name}")
        start_time = time.time()
        
        # Extract document structure
        result = engine.extract_document_structure(str(pdf_path))
        
        # Post-process and validate JSON
        processed_result = processor.validate_and_fix_structure(result)
        
        # Format JSON with proper indentation
        json_output = processor.format_json(processed_result, indent=2)
        
        # Save to output file
        output_path.write_text(json_output, encoding='utf-8')
        
        processing_time = time.time() - start_time
        print(f"✅ Completed: {pdf_path.name} → {output_filename} ({processing_time:.2f}s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {pdf_path.name}: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error output file
        error_output = {
            "title": "",
            "outline": [],
            "error": str(e),
            "file": pdf_path.name
        }
        
        try:
            output_filename = pdf_path.stem + '.json'
            output_path = output_dir / output_filename
            output_path.write_text(json.dumps(error_output, indent=2), encoding='utf-8')
        except:
            pass
        
        return False


def main():
    """Main entrypoint for Docker container."""
    print("🚀 Adobe India Hackathon - Round 1A PDF Processing Container")
    print("=" * 60)
    
    # Set up directories
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Find all PDF files
    pdf_files = find_pdf_files(input_dir)
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {input_dir}")
        print("💡 Make sure PDF files are mounted to /app/input")
        return
    
    print(f"📊 Found {len(pdf_files)} PDF file(s) to process:")
    for pdf_file in pdf_files:
        print(f"   • {pdf_file.name}")
    
    print("\n" + "=" * 60)
    print("🔄 Starting PDF processing...")
    print("=" * 60)
    
    # Initialize processing engine and post-processor
    try:
        engine = OptimizedRound1ASolutionEngine(
            max_workers=None,  # Auto-detect optimal workers
            enable_memory_optimization=True
        )
        processor = JSONPostProcessor()
        
        print("✅ Processing engine initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize processing engine: {e}")
        sys.exit(1)
    
    # Process each PDF file
    total_start_time = time.time()
    successful_count = 0
    failed_count = 0
    
    for pdf_file in pdf_files:
        success = process_pdf_file(pdf_file, output_dir, engine, processor)
        if success:
            successful_count += 1
        else:
            failed_count += 1
    
    # Print summary
    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("📋 PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files: {len(pdf_files)}")
    print(f"✅ Successful: {successful_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📊 Average time per file: {total_time / len(pdf_files):.2f}s")
    
    # List output files
    output_files = list(output_dir.glob("*.json"))
    if output_files:
        print(f"\n📄 Generated output files:")
        for output_file in sorted(output_files):
            print(f"   • {output_file.name}")
    
    print(f"\n🎯 Processing completed! Results saved to {output_dir}")
    
    # Exit with appropriate code
    if failed_count > 0:
        sys.exit(1)  # Some files failed
    else:
        sys.exit(0)  # All files processed successfully


if __name__ == "__main__":
    main() 