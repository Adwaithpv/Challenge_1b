#!/usr/bin/env python3
"""
Docker entrypoint script for the Adobe India Hackathon Round 1B solution.

This script automatically processes PDF files and their corresponding Round 1A JSON files
from /app/input directory using a specified persona and job-to-be-done, then generates
persona-driven document intelligence output in /app/output directory.

Expected input structure:
/app/input/
  - document1.pdf
  - document1.json (Round 1A output)
  - document2.pdf  
  - document2.json (Round 1A output)
  - ...
  - persona.txt (contains the user persona)
  - job.txt (contains the job-to-be-done)

Expected output:
/app/output/
  - round1b_output.json (ranked sections with refined text)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Tuple

# Add src to path for imports
sys.path.append('/app/src')
sys.path.append('/app')

# Force CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from round1b_solution import Round1BDocumentIntelligence


def find_pdf_json_pairs(input_dir: Path) -> List[Tuple[Path, Path]]:
    """Find all PDF files and their corresponding Round 1A JSON files."""
    pairs = []
    
    if not input_dir.exists():
        print(f"⚠️  Input directory does not exist: {input_dir}")
        return pairs
    
    # Find all PDF files
    pdf_files = []
    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == '.pdf':
            pdf_files.append(file_path)
    
    # Find corresponding JSON files
    for pdf_path in pdf_files:
        json_path = input_dir / f"{pdf_path.stem}.json"
        if json_path.exists():
            pairs.append((pdf_path, json_path))
        else:
            print(f"⚠️  Warning: No Round 1A JSON found for {pdf_path.name}")
    
    return sorted(pairs)


def read_persona_and_job(input_dir: Path) -> Tuple[str, str]:
    """Read persona and job-to-be-done from input files."""
    persona_file = input_dir / "persona.txt"
    job_file = input_dir / "job.txt"
    
    # Default values if files don't exist
    default_persona = "Research Analyst"
    default_job = "analyze key information and trends"
    
    # Read persona
    try:
        if persona_file.exists():
            persona = persona_file.read_text(encoding='utf-8').strip()
            if not persona:
                persona = default_persona
        else:
            print(f"⚠️  persona.txt not found, using default: '{default_persona}'")
            persona = default_persona
    except Exception as e:
        print(f"⚠️  Error reading persona.txt: {e}, using default: '{default_persona}'")
        persona = default_persona
    
    # Read job
    try:
        if job_file.exists():
            job = job_file.read_text(encoding='utf-8').strip()
            if not job:
                job = default_job
        else:
            print(f"⚠️  job.txt not found, using default: '{default_job}'")
            job = default_job
    except Exception as e:
        print(f"⚠️  Error reading job.txt: {e}, using default: '{default_job}'")
        job = default_job
    
    return persona, job


def process_round1b(input_dir: Path, output_dir: Path) -> bool:
    """Process Round 1B document intelligence pipeline."""
    try:
        print("🚀 Starting Round 1B: Persona-Driven Document Intelligence")
        print("=" * 60)
        
        start_time = time.time()
        
        # Check for input.json (REQUIRED FORMAT)
        input_json_path = input_dir / "input.json"
        
        if input_json_path.exists():
            print(f"📄 Found input.json - using required JSON format")
            
            # Initialize Round 1B system
            print("🧠 Initializing Document Intelligence System...")
            system = Round1BDocumentIntelligence(model_cache_dir='/app/models')
            
            # Process documents from JSON
            print("⚡ Processing documents...")
            result = system.process_documents_from_input_json(str(input_json_path))
            
        else:
            print("⚠️  No input.json found - falling back to legacy format")
            
            # Find PDF and JSON pairs (LEGACY SUPPORT)
            pdf_json_pairs = find_pdf_json_pairs(input_dir)
            
            if not pdf_json_pairs:
                print("❌ No PDF-JSON pairs found in input directory")
                return False
            
            print(f"📁 Found {len(pdf_json_pairs)} document pairs:")
            for pdf_path, json_path in pdf_json_pairs:
                print(f"   • {pdf_path.name} + {json_path.name}")
            
            # Read persona and job (LEGACY)
            persona, job = read_persona_and_job(input_dir)
            print(f"👤 Persona: {persona}")
            print(f"🎯 Job-to-be-Done: {job}")
            print()
            
            # Extract paths for processing
            pdf_paths = [str(pdf) for pdf, _ in pdf_json_pairs]
            json_paths = [str(json) for _, json in pdf_json_pairs]
            
            # Initialize Round 1B system
            print("🧠 Initializing Document Intelligence System...")
            system = Round1BDocumentIntelligence(model_cache_dir='/app/models')
            
            # Process documents
            print("⚡ Processing documents...")
            result = system.process_documents(
                pdf_paths=pdf_paths,
                json_paths=json_paths,
                persona=persona,
                job=job
            )
        
        # Save output
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'output.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Print summary
        processing_time = time.time() - start_time
        print()
        print("✅ Round 1B Processing Completed Successfully!")
        print("=" * 60)
        print(f"📊 Summary:")
        print(f"   • Documents processed: {len(result['metadata']['input_documents'])}")
        print(f"   • Total sections analyzed: {len(result['extracted_sections'])}")
        print(f"   • Sections with refined text: {len(result['subsection_analysis'])}")
        print(f"   • Processing time: {processing_time:.2f}s")
        print(f"   • Output saved to: {output_path}")
        
        # Performance check
        if processing_time > 60:
            print(f"⚠️  Warning: Processing time ({processing_time:.2f}s) exceeded 60s target")
        else:
            print(f"🎉 Performance target met: {processing_time:.2f}s < 60s")
        
        # Show top 3 most relevant sections
        if result.get('extracted_sections'):
            print(f"\n🏆 Top 3 Most Relevant Sections:")
            for i, section in enumerate(result['extracted_sections'][:3]):
                print(f"   {i+1}. '{section['section_title']}' (Rank: {section['importance_rank']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Round 1B processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entrypoint for Docker container."""
    print("🐳 Adobe India Hackathon - Round 1B Docker Container")
    print("🧠 Persona-Driven Document Intelligence System")
    print("=" * 60)
    
    # Define input and output directories
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Validate input directory
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        print("💡 Please mount your input directory to /app/input")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for basic input files
    pdf_files = list(input_dir.glob("*.pdf"))
    json_files = list(input_dir.glob("*.json"))
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📂 Found {len(pdf_files)} PDF files and {len(json_files)} JSON files")
    
    if not pdf_files:
        print("❌ No PDF files found in input directory")
        return 1
        
    if not json_files:
        print("❌ No JSON files found in input directory")
        print("💡 Make sure you have Round 1A JSON outputs for your PDFs")
        return 1
    
    # Process Round 1B
    success = process_round1b(input_dir, output_dir)
    
    if success:
        print("\n🎉 Docker container execution completed successfully!")
        return 0
    else:
        print("\n❌ Docker container execution failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 