#!/usr/bin/env python3
"""
Round 1B Solution: Adobe India Hackathon - Persona-Driven Document Intelligence
A sophisticated system that functions as an intelligent document analyst.

This system ingests:
- Collection of related PDF documents (from Round 1A processing)
- User Persona (e.g., "PhD Researcher", "Investment Analyst")  
- Job-to-be-Done (e.g., "Prepare a literature review", "Analyze revenue trends")

And outputs:
- Ranked list of most relevant sections that address the persona's task
- Concise summaries ("Refined Text") of key sections

Constraints:
- CPU-only linux/amd64 architecture, 8 CPUs, 16GB RAM
- Complete end-to-end in <60 seconds for 3-5 documents
- Model size ≤1GB total
- Fully offline execution (no internet access)

Architecture follows the "Retrieve and Re-Rank" paradigm with extractive summarization.
"""

import sys
import json
import time
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime
import threading
import logging

# Force CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Force offline mode for transformers and sentence-transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Add src to path for imports (matching existing structure)
sys.path.append('src')
sys.path.append('.')

# Core imports for Round 1B functionality
try:
    from sentence_transformers import SentenceTransformer
    from summarizer import Summarizer
    from scipy.spatial.distance import cosine
    import fitz  # PyMuPDF for PDF text extraction
except ImportError as e:
    print(f"Error importing Round 1B dependencies: {e}")
    print("Please ensure sentence-transformers, bert-extractive-summarizer, scipy, and PyMuPDF are installed")
    sys.exit(1)


class Round1BDocumentIntelligence:
    """
    Persona-Driven Document Intelligence Engine
    
    Implements the complete pipeline:
    1. Content Hydration: Extract full text for sections using Round 1A JSON + PDFs
    2. Semantic Vectorization: Use all-MiniLM-L6-v2 for embeddings
    3. Relevance Ranking: Cosine similarity between query and sections
    4. Text Summarization: Extractive summarization with model reuse
    """
    
    def __init__(self, model_cache_dir: str = "./models"):
        """Initialize the document intelligence system."""
        self.model_cache_dir = model_cache_dir
        self.max_processing_time = 60.0  # seconds for entire pipeline
        self.start_time = None
        
        # Model instances (will be loaded on first use)
        self.embedding_model = None
        self.summarizer = None
        
        # Configuration based on user's blueprint
        self.model_name = "all-MiniLM-L6-v2"  # Optimal choice: 80MB, 14k sentences/sec
        self.embedding_dim = 384
        self.max_token_length = 256  # Model's token limit
        self.chunk_size = 200  # Tokens per chunk (with overlap)
        self.chunk_overlap = 50  # Token overlap between chunks
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_models(self):
        """Load the embedding model and summarizer with model reuse optimization."""
        if self.embedding_model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            
            # Use local model path for offline execution
            local_model_path = os.path.join(
                self.model_cache_dir, 
                "models--sentence-transformers--all-MiniLM-L6-v2",
                "snapshots",
                "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
            )
            
            # Check if local model exists, fallback to model name if not
            if os.path.exists(local_model_path):
                self.logger.info(f"Using local model from: {local_model_path}")
                self.embedding_model = SentenceTransformer(local_model_path)
            else:
                self.logger.info(f"Local model not found, using model name: {self.model_name}")
                self.embedding_model = SentenceTransformer(
                    self.model_name, 
                    cache_folder=self.model_cache_dir
                )
            
        if self.summarizer is None:
            self.logger.info("Initializing extractive summarizer")
            # Initialize summarizer with default BERT model (optimized for CPU)
            self.summarizer = Summarizer(
                model='distilbert-base-uncased',  # Lightweight BERT model
                hidden=-2,  # Use second-to-last layer
                reduce_option='mean'  # Mean pooling for sentence representations
            )
            
    def hydrate_content_from_round1a(self, pdf_paths: List[str], json_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Step 1: Content Hydration Process
        
        Transform Round 1A structural outline into content-rich sections.
        For each PDF and its corresponding JSON:
        1. Load the Round 1A JSON structure (titles, headings, page numbers)
        2. Extract full PDF text with page mapping
        3. Map content to sections based on heading boundaries
        
        Args:
            pdf_paths: List of PDF file paths
            json_paths: List of corresponding Round 1A JSON file paths
            
        Returns:
            List of hydrated sections with full content
        """
        hydrated_sections = []
        
        for pdf_path, json_path in zip(pdf_paths, json_paths):
            self.logger.info(f"Hydrating content from {pdf_path}")
            
            # Load Round 1A structure
            with open(json_path, 'r', encoding='utf-8') as f:
                round1a_data = json.load(f)
            
            # Extract full PDF text with page mapping
            pdf_text_by_page = self._extract_pdf_text_by_page(pdf_path)
            
            # Process each section in the outline
            outline = round1a_data.get('outline', [])
            document_title = round1a_data.get('title', os.path.basename(pdf_path))
            
            for i, section in enumerate(outline):
                section_content = self._extract_section_content(
                    section, outline, i, pdf_text_by_page
                )
                
                if section_content.strip():  # Only include non-empty sections
                    hydrated_section = {
                        'source_document': os.path.basename(pdf_path),
                        'document_title': document_title,
                        'section_title': section['text'],
                        'section_level': section['level'],
                        'start_page': section['page'],
                        'content': section_content,
                        'content_length': len(section_content)
                    }
                    hydrated_sections.append(hydrated_section)
        
        self.logger.info(f"Hydrated {len(hydrated_sections)} sections from {len(pdf_paths)} documents")
        return hydrated_sections
    
    def _extract_pdf_text_by_page(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF with page-by-page mapping."""
        pdf_text_by_page = {}
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    pdf_text_by_page[page_num + 1] = text  # 1-indexed pages
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
            
        return pdf_text_by_page
    
    def _extract_section_content(self, current_section: Dict, outline: List[Dict], 
                                section_index: int, pdf_text_by_page: Dict[int, str]) -> str:
        """
        Extract content for a specific section based on heading boundaries.
        
        Content for section[i] = text from section[i].page to section[i+1].page (exclusive)
        For the last section, content extends to end of document.
        """
        start_page = current_section['page']
        current_level = current_section['level']
        
        # Find the end boundary (next section of same or higher level)
        end_page = None
        for j in range(section_index + 1, len(outline)):
            next_section = outline[j]
            next_level = next_section['level']
            
            # Stop at next section of same or higher importance
            if self._get_heading_hierarchy_level(next_level) <= self._get_heading_hierarchy_level(current_level):
                end_page = next_section['page']
                break
        
        # Extract text from the page range
        section_text = ""
        max_page = max(pdf_text_by_page.keys()) if pdf_text_by_page else start_page
        actual_end_page = end_page if end_page else max_page + 1
        
        for page_num in range(start_page, actual_end_page):
            if page_num in pdf_text_by_page:
                section_text += pdf_text_by_page[page_num] + "\n"
        
        return section_text.strip()
    
    def _get_heading_hierarchy_level(self, heading_level: str) -> int:
        """Convert heading level (H1, H2, H3) to numeric hierarchy."""
        level_map = {'H1': 1, 'H2': 2, 'H3': 3, 'H4': 4, 'H5': 5, 'H6': 6}
        return level_map.get(heading_level, 99)  # Unknown levels get lowest priority
    
    def vectorize_corpus_and_query(self, hydrated_sections: List[Dict], persona: str, job: str) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Step 2: Semantic Vector Representation
        
        Convert query (persona + job) and document sections into high-dimensional
        semantic vectors using all-MiniLM-L6-v2.
        
        Args:
            hydrated_sections: Content-rich sections from Step 1
            persona: User persona (e.g., "Investment Analyst")
            job: Job-to-be-done (e.g., "Analyze revenue trends")
            
        Returns:
            Tuple of (section_vectors, query_vector)
        """
        self._load_models()
        
        # Formulate descriptive query following blueprint strategy
        query_text = f"A {persona} needs to {job}. The relevant information is:"
        self.logger.info(f"Query formulated: {query_text}")
        
        # Generate query embedding
        query_vector = self.embedding_model.encode([query_text])[0]
        
        # Generate section embeddings with chunking for long content
        section_vectors = []
        for section in hydrated_sections:
            content = section['content']
            
            if len(content.split()) <= self.chunk_size:
                # Short content - encode directly
                vector = self.embedding_model.encode([content])[0]
            else:
                # Long content - chunk and aggregate
                vector = self._encode_long_content(content)
            
            section_vectors.append(vector)
        
        self.logger.info(f"Generated embeddings for {len(section_vectors)} sections and 1 query")
        return section_vectors, query_vector
    
    def _encode_long_content(self, content: str) -> np.ndarray:
        """
        Handle content longer than model's token limit using chunking and aggregation.
        
        Strategy:
        1. Split content into overlapping chunks of ~200 tokens
        2. Generate embedding for each chunk
        3. Use mean pooling to create single representative vector
        """
        words = content.split()
        chunks = []
        
        # Create overlapping chunks
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            
            # Move start position (with overlap)
            start += (self.chunk_size - self.chunk_overlap)
            if start >= len(words):
                break
        
        # Generate embeddings for all chunks
        if chunks:
            chunk_embeddings = self.embedding_model.encode(chunks)
            # Mean pooling across chunks
            aggregated_vector = np.mean(chunk_embeddings, axis=0)
        else:
            # Fallback for edge cases
            aggregated_vector = self.embedding_model.encode([content[:1000]])[0]
        
        return aggregated_vector
    
    def rank_sections_by_relevance(self, hydrated_sections: List[Dict], section_vectors: List[np.ndarray], 
                                 query_vector: np.ndarray) -> List[Dict]:
        """
        Step 3: Relevance Ranking using Cosine Similarity
        
        Calculate cosine similarity between query and each section to rank
        sections globally by relevance to the persona's job.
        
        Args:
            hydrated_sections: Original section data
            section_vectors: Section embedding vectors
            query_vector: Query embedding vector
            
        Returns:
            Ranked list of sections with similarity scores and importance ranks
        """
        ranked_sections = []
        
        # Calculate cosine similarity for each section
        for section, vector in zip(hydrated_sections, section_vectors):
            # Cosine similarity = 1 - cosine distance
            similarity_score = 1 - cosine(query_vector, vector)
            
            # Create ranked section object
            ranked_section = {
                'source_document': section['source_document'],
                'section_title': section['section_title'],
                'start_page': section['start_page'],
                'similarity_score': float(similarity_score),
                'content': section['content'],  # Keep for summarization
                'section_level': section['section_level']
            }
            ranked_sections.append(ranked_section)
        
        # Sort by similarity score (descending)
        ranked_sections.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Assign importance ranks
        for i, section in enumerate(ranked_sections):
            section['importance_rank'] = i + 1
        
        self.logger.info(f"Ranked {len(ranked_sections)} sections by relevance")
        if ranked_sections:
            self.logger.info(f"Top section: '{ranked_sections[0]['section_title']}' (score: {ranked_sections[0]['similarity_score']:.4f})")
        
        return ranked_sections
    
    def generate_refined_text(self, ranked_sections: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Step 4: Extractive Summarization for Refined Text
        
        Generate concise summaries for the most relevant sections using
        BERT-based extractive summarization with model reuse.
        
        Args:
            ranked_sections: Sections ranked by relevance
            top_k: Number of top sections to summarize
            
        Returns:
            List of sections with refined text summaries
        """
        self._load_models()
        
        # Process only top-k most relevant sections
        sections_to_summarize = ranked_sections[:top_k]
        
        for section in sections_to_summarize:
            content = section['content']
            
            try:
                # Generate summary using shared model (KEY OPTIMIZATION)
                summary = self.summarizer(
                    content, 
                    num_sentences=3,  # Concise 3-sentence summaries
                    use_first=False  # Don't bias toward first sentences
                )
                section['refined_text'] = summary.strip()
                
            except Exception as e:
                self.logger.warning(f"Summarization failed for section '{section['section_title']}': {e}")
                # Fallback: use first few sentences
                sentences = content.split('. ')[:3]
                section['refined_text'] = '. '.join(sentences) + '.' if sentences else content[:500]
        
        # For sections beyond top_k, don't include refined_text (per requirements)
        for section in ranked_sections[top_k:]:
            section['refined_text'] = ""
        
        self.logger.info(f"Generated refined text for top {min(top_k, len(ranked_sections))} sections")
        return ranked_sections
    
    def process_documents_from_input_json(self, input_json_path: str) -> Dict[str, Any]:
        """
        Main processing pipeline: Execute the complete workflow from input JSON.
        
        Args:
            input_json_path: Path to the input JSON file with required format
            
        Returns:
            Complete output JSON with ranked sections and metadata
        """
        # Load and parse input JSON
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Extract required fields
        documents = input_data.get('documents', [])
        persona_role = input_data.get('persona', {}).get('role', '')
        job_task = input_data.get('job_to_be_done', {}).get('task', '')
        
        # Get the directory of the input JSON file
        input_dir = os.path.dirname(os.path.abspath(input_json_path))
        
        # Build PDF and JSON paths with proper directory handling
        pdf_paths = []
        json_paths = []
        
        for doc in documents:
            filename = doc.get('filename', '')
            if filename.endswith('.pdf'):
                # Try multiple possible locations for PDFs
                possible_pdf_paths = [
                    filename,  # Current directory
                    os.path.join(input_dir, filename),  # Same dir as input JSON
                    os.path.join(input_dir, 'PDFs', filename),  # PDFs subdirectory
                    os.path.join('Collection 2', 'PDFs', filename)  # Explicit Collection 2/PDFs
                ]
                
                # Find the actual PDF path
                pdf_path = None
                for candidate_path in possible_pdf_paths:
                    if os.path.exists(candidate_path):
                        pdf_path = candidate_path
                        break
                
                # If PDF not found, use the PDFs subdirectory path for generation
                if pdf_path is None:
                    pdf_path = os.path.join(input_dir, 'PDFs', filename)
                
                # JSON path should be in the same directory as the input JSON
                json_filename = filename.replace('.pdf', '.json')
                json_path = os.path.join(input_dir, json_filename)
                
                pdf_paths.append(pdf_path)
                json_paths.append(json_path)
        
        # Check if Round 1A JSON files exist, if not generate them
        missing_jsons = self._check_and_generate_round1a_jsons(pdf_paths, json_paths)
        
        return self.process_documents(pdf_paths, json_paths, persona_role, job_task)

    def process_documents(self, pdf_paths: List[str], json_paths: List[str], 
                         persona: str, job: str) -> Dict[str, Any]:
        """
        Main processing pipeline: Execute the complete workflow.
        
        Args:
            pdf_paths: List of PDF file paths  
            json_paths: List of corresponding Round 1A JSON paths
            persona: User persona description
            job: Job-to-be-done description
            
        Returns:
            Complete output JSON with ranked sections and metadata
        """
        self.start_time = time.time()
        
        try:
            # Step 1: Content Hydration
            self.logger.info("Step 1: Hydrating content from Round 1A output...")
            hydrated_sections = self.hydrate_content_from_round1a(pdf_paths, json_paths)
            
            if not hydrated_sections:
                raise ValueError("No valid sections found after content hydration")
            
            # Step 2: Semantic Vectorization
            self.logger.info("Step 2: Generating semantic embeddings...")
            section_vectors, query_vector = self.vectorize_corpus_and_query(
                hydrated_sections, persona, job
            )
            
            # Step 3: Relevance Ranking
            self.logger.info("Step 3: Ranking sections by relevance...")
            ranked_sections = self.rank_sections_by_relevance(
                hydrated_sections, section_vectors, query_vector
            )
            
            # Step 4: Text Summarization
            self.logger.info("Step 4: Generating refined text summaries...")
            final_sections = self.generate_refined_text(ranked_sections)
            
            # Construct final output
            output = self._construct_final_output(final_sections, persona, job, pdf_paths)
            
            processing_time = time.time() - self.start_time
            self.logger.info(f"Total processing time: {processing_time:.2f}s")
            
            if processing_time > self.max_processing_time:
                self.logger.warning(f"Processing exceeded target time: {processing_time:.2f}s > {self.max_processing_time}s")
            
            return output
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise
    
    def _construct_final_output(self, ranked_sections: List[Dict], persona: str, 
                              job: str, pdf_paths: List[str]) -> Dict[str, Any]:
        """Construct the final JSON output according to Round 1B required format."""
        
        # Create extracted sections array (EXACT required format) - TOP 5 ONLY
        extracted_sections = []
        for section in ranked_sections[:5]:  # Only include top 5 most important sections
            extracted_section = {
                "document": section['source_document'],
                "section_title": section['section_title'], 
                "importance_rank": section['importance_rank'],
                "page_number": section['start_page']
            }
            extracted_sections.append(extracted_section)
        
        # Create subsection analysis for top sections with refined text (EXACT required format)
        subsection_analysis = []
        for section in ranked_sections:
            if section.get('refined_text', '').strip():
                analysis = {
                    "document": section['source_document'],
                    "refined_text": section['refined_text'],
                    "page_number": section['start_page']
                }
                subsection_analysis.append(analysis)
        
        # Construct complete output in EXACT required format
        output = {
            "metadata": {
                "input_documents": [os.path.basename(path) for path in pdf_paths],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        return output

    def _check_and_generate_round1a_jsons(self, pdf_paths: List[str], json_paths: List[str]) -> List[str]:
        """
        Check if Round 1A JSON files exist. If not, generate them using Round 1A solution.
        
        Args:
            pdf_paths: List of PDF file paths
            json_paths: List of expected JSON file paths
            
        Returns:
            List of JSON files that were missing and generated
        """
        missing_jsons = []
        
        # Check which JSON files are missing
        for pdf_path, json_path in zip(pdf_paths, json_paths):
            if not os.path.exists(json_path):
                missing_jsons.append((pdf_path, json_path))
        
        if missing_jsons:
            self.logger.info(f"Found {len(missing_jsons)} missing Round 1A JSON files. Generating them...")
            self._generate_round1a_outputs(missing_jsons)
        else:
            self.logger.info("All Round 1A JSON files exist. Proceeding with Round 1B processing.")
        
        return [json_path for _, json_path in missing_jsons]
    
    def _generate_round1a_outputs(self, missing_files: List[Tuple[str, str]]):
        """
        Generate Round 1A JSON outputs for missing files using the optimized Round 1A solution.
        
        Args:
            missing_files: List of (pdf_path, json_path) tuples for missing JSON files
        """
        self.logger.info("🔄 Generating Round 1A JSON files using optimized Round 1A solution...")
        
        try:
            # Import Round 1A solution
            from round1a_solution_optimized import OptimizedRound1ASolutionEngine
            
            # Initialize Round 1A engine
            round1a_engine = OptimizedRound1ASolutionEngine(
                max_workers=None,  # Auto-detect
                enable_memory_optimization=True
            )
            
            # Process each missing file
            for i, (pdf_path, json_path) in enumerate(missing_files, 1):
                self.logger.info(f"📄 Processing {i}/{len(missing_files)}: {os.path.basename(pdf_path)}")
                
                try:
                    # Check if PDF exists
                    if not os.path.exists(pdf_path):
                        self.logger.error(f"PDF file not found: {pdf_path}")
                        continue
                    
                    # Extract document structure using Round 1A
                    start_time = time.time()
                    structure = round1a_engine.extract_document_structure(pdf_path)
                    processing_time = time.time() - start_time
                    
                    # Add metadata
                    structure["source_pdf"] = os.path.basename(pdf_path)
                    structure["generated_by"] = "Round1A_OptimizedEngine"
                    structure["generation_time"] = datetime.now().isoformat()
                    structure["processing_time_seconds"] = round(processing_time, 2)
                    
                    # Save JSON file
                    os.makedirs(os.path.dirname(json_path), exist_ok=True)
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(structure, f, indent=2, ensure_ascii=False)
                    
                    self.logger.info(f"✅ Generated {os.path.basename(json_path)} in {processing_time:.2f}s")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing {pdf_path}: {e}")
                    
                    # Create a minimal fallback JSON structure
                    fallback_structure = {
                        "title": os.path.basename(pdf_path).replace('.pdf', ''),
                        "outline": [],
                        "source_pdf": os.path.basename(pdf_path),
                        "generated_by": "Round1A_Fallback",
                        "generation_time": datetime.now().isoformat(),
                        "error": str(e)
                    }
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(fallback_structure, f, indent=2, ensure_ascii=False)
                    
                    self.logger.warning(f"⚠️ Created fallback JSON for {os.path.basename(json_path)}")
            
            self.logger.info(f"✅ Round 1A JSON generation completed for {len(missing_files)} files")
            
        except ImportError as e:
            self.logger.error(f"❌ Could not import Round 1A solution: {e}")
            raise ValueError("Round 1A solution is required but not available")
        except Exception as e:
            self.logger.error(f"❌ Error during Round 1A generation: {e}")
            raise


def main():
    """Main entry point for Round 1B solution."""
    # Start timing the entire process
    main_start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Round 1B: Persona-Driven Document Intelligence')
    parser.add_argument('--input-json', type=str,
                       help='Input JSON file with required format (preferred method)')
    parser.add_argument('--input-dir', type=str,
                       help='Directory containing PDFs and Round 1A JSON files (legacy)')
    parser.add_argument('--output-file', type=str, default='output.json',
                       help='Output JSON file path')
    parser.add_argument('--persona', type=str,
                       help='User persona (legacy - use input JSON instead)')
    parser.add_argument('--job', type=str, 
                       help='Job-to-be-done (legacy - use input JSON instead)')
    parser.add_argument('--model-cache', type=str, default='./models',
                       help='Directory for model cache')
    
    args = parser.parse_args()
    
    print("🚀 Starting Round 1B: Persona-Driven Document Intelligence")
    print("=" * 60)
    
    # Initialize Round 1B system
    system = Round1BDocumentIntelligence(model_cache_dir=args.model_cache)
    
    try:
        # Method 1: Use input JSON (PREFERRED - matches required format)
        if args.input_json:
            result = system.process_documents_from_input_json(args.input_json)
            
        # Method 2: Legacy command-line arguments (for backwards compatibility)
        elif args.input_dir and args.persona and args.job:
            # Find PDF and JSON pairs
            input_path = Path(args.input_dir)
            pdf_paths = list(input_path.glob('*.pdf'))
            
            if not pdf_paths:
                print(f"No PDF files found in {input_path}")
                return 1
            
            # Find corresponding JSON files
            json_paths = []
            for pdf_path in pdf_paths:
                json_path = input_path / f"{pdf_path.stem}.json"
                if json_path.exists():
                    json_paths.append(str(json_path))
                else:
                    print(f"Warning: No Round 1A JSON found for {pdf_path.name}")
                    return 1
            
            if len(pdf_paths) != len(json_paths):
                print("Mismatch between PDF and JSON files")
                return 1
            
            result = system.process_documents(
                pdf_paths=[str(p) for p in pdf_paths],
                json_paths=json_paths,
                persona=args.persona,
                job=args.job
            )
        else:
            print("Error: Must provide either --input-json OR (--input-dir + --persona + --job)")
            print("Preferred method: --input-json input.json --output-file output.json")
            return 1
        
        # Save output
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Calculate total execution time
        total_time = time.time() - main_start_time
        
        print()
        print("✅ Round 1B Processing Completed Successfully!")
        print("=" * 60)
        print(f"📊 Final Results:")
        print(f"   • Documents processed: {len(result['metadata']['input_documents'])}")
        print(f"   • Total sections analyzed: {len(result['extracted_sections'])}")
        print(f"   • Sections with refined text: {len(result['subsection_analysis'])}")
        print(f"   • Output saved to: {output_path}")
        print()
        print(f"⏱️  TOTAL EXECUTION TIME: {total_time:.2f} seconds")
        print("=" * 60)
        
        # Performance evaluation
        if total_time <= 60:
            print(f"🎉 PERFORMANCE TARGET MET: {total_time:.2f}s ≤ 60s")
        else:
            print(f"⚠️  Performance target exceeded: {total_time:.2f}s > 60s")
        
        # Show top 3 most relevant sections
        if result.get('extracted_sections'):
            print(f"\n🏆 Top 3 Most Relevant Sections:")
            for i, section in enumerate(result['extracted_sections'][:3]):
                print(f"   {i+1}. '{section['section_title']}' from {section['document']} (Rank: {section['importance_rank']})")
        
        print()
        
    except Exception as e:
        total_time = time.time() - main_start_time
        print()
        print("❌ Round 1B Processing Failed!")
        print("=" * 60)
        print(f"Error: {e}")
        print(f"⏱️  Time elapsed before failure: {total_time:.2f} seconds")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 