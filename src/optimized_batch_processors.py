"""
Optimized Batch Processors for PDF Analysis
Implements parallel processing and batching for core PDF processing components.

Key optimizations:
- Batch processing for model predictions
- Parallel page processing
- Memory-efficient chunked processing
- Vectorized operations where possible
"""

import os
import numpy as np
import concurrent.futures
from typing import List, Dict, Any, Tuple
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import time

# Import core components
import sys
import os

# Add src to path for imports
if 'src' not in sys.path:
    sys.path.append('src')

from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfPage import PdfPage
from pdf_features.PdfToken import PdfToken
from fast_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration


class OptimizedTokenTypeTrainer(TokenTypeTrainer):
    """Optimized token type trainer with batch processing and parallel execution."""
    
    def __init__(self, pdfs_features: list[PdfFeatures], model_configuration: ModelConfiguration = None, max_workers: int = None):
        super().__init__(pdfs_features, model_configuration)
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self._model_cache = {}
        
    def get_model_input_parallel(self) -> np.ndarray:
        """Optimized model input generation with parallel processing."""
        features_rows = []
        context_size = self.model_configuration.context_size
        
        # Process each PDF in parallel if multiple PDFs
        if len(self.pdfs_features) > 1:
            with ThreadPoolExecutor(max_workers=min(len(self.pdfs_features), self.max_workers)) as executor:
                futures = []
                for pdf_features in self.pdfs_features:
                    future = executor.submit(self._process_pdf_features, pdf_features, context_size)
                    futures.append(future)
                
                for future in as_completed(futures):
                    features_rows.extend(future.result())
        else:
            # Single PDF - process pages in parallel
            for pdf_features in self.pdfs_features:
                pdf_features_rows = self._process_pdf_features_parallel(pdf_features, context_size)
                features_rows.extend(pdf_features_rows)
        
        return self.features_rows_to_x(features_rows)
    
    def _process_pdf_features(self, pdf_features: PdfFeatures, context_size: int) -> List[List[float]]:
        """Process a single PDF's features."""
        from pdf_tokens_type_trainer.TokenFeatures import TokenFeatures
        
        token_features = TokenFeatures(pdf_features)
        features_rows = []
        
        for page in pdf_features.pages:
            if not page.tokens:
                continue
            
            # Add padding tokens
            page_tokens = [
                self.get_padding_token(segment_number=i - 999999, page_number=page.page_number) 
                for i in range(context_size)
            ]
            page_tokens += page.tokens
            page_tokens += [
                self.get_padding_token(segment_number=999999 + i, page_number=page.page_number) 
                for i in range(context_size)
            ]
            
            # Process tokens in batches
            tokens_indexes = range(context_size, len(page_tokens) - context_size)
            page_features = [
                self.get_context_features(token_features, page_tokens, i) 
                for i in tokens_indexes
            ]
            features_rows.extend(page_features)
        
        return features_rows
    
    def _process_pdf_features_parallel(self, pdf_features: PdfFeatures, context_size: int) -> List[List[float]]:
        """Process PDF features with parallel page processing."""
        from pdf_tokens_type_trainer.TokenFeatures import TokenFeatures
        
        token_features = TokenFeatures(pdf_features)
        features_rows = []
        
        # Filter pages with tokens
        pages_with_tokens = [page for page in pdf_features.pages if page.tokens]
        
        if len(pages_with_tokens) > 4:  # Parallel processing for multiple pages
            with ThreadPoolExecutor(max_workers=min(len(pages_with_tokens), self.max_workers)) as executor:
                futures = []
                for page in pages_with_tokens:
                    future = executor.submit(self._process_page_features, page, token_features, context_size)
                    futures.append(future)
                
                for future in as_completed(futures):
                    features_rows.extend(future.result())
        else:
            # Sequential processing for few pages
            for page in pages_with_tokens:
                page_features = self._process_page_features(page, token_features, context_size)
                features_rows.extend(page_features)
        
        return features_rows
    
    def _process_page_features(self, page: PdfPage, token_features, context_size: int) -> List[List[float]]:
        """Process features for a single page."""
        # Add padding tokens
        page_tokens = [
            self.get_padding_token(segment_number=i - 999999, page_number=page.page_number) 
            for i in range(context_size)
        ]
        page_tokens += page.tokens
        page_tokens += [
            self.get_padding_token(segment_number=999999 + i, page_number=page.page_number) 
            for i in range(context_size)
        ]
        
        # Batch process token features
        tokens_indexes = range(context_size, len(page_tokens) - context_size)
        page_features = [
            self.get_context_features(token_features, page_tokens, i) 
            for i in tokens_indexes
        ]
        
        return page_features
    
    def predict_batch(self, model_path: str = None) -> np.ndarray:
        """Optimized batch prediction with caching."""
        # Use parallel model input generation
        x = self.get_model_input_parallel()
        
        if not x.any():
            return np.array([])
        
        # Cache model loading
        if model_path not in self._model_cache:
            import lightgbm as lgb
            self._model_cache[model_path] = lgb.Booster(model_file=model_path)
        
        lightgbm_model = self._model_cache[model_path]
        
        # Batch prediction - process in chunks if too large
        batch_size = 10000  # Adjust based on memory
        if len(x) > batch_size:
            predictions = []
            for i in range(0, len(x), batch_size):
                batch = x[i:i + batch_size]
                batch_predictions = lightgbm_model.predict(batch)
                predictions.append(batch_predictions)
            return np.concatenate(predictions)
        else:
            return lightgbm_model.predict(x)
    
    def set_token_types_optimized(self, model_path: str = None):
        """Optimized token type setting with batch processing."""
        predictions = self.predict_batch(model_path)
        predictions_assigned = 0
        
        for pdf_features in self.pdfs_features:
            # Process all pages for this PDF
            for page in pdf_features.pages:
                if not page.tokens:
                    continue
                
                page_predictions = predictions[
                    predictions_assigned:predictions_assigned + len(page.tokens)
                ]
                
                # Vectorized assignment
                for token, prediction in zip(page.tokens, page_predictions):
                    token.prediction = int(np.argmax(prediction))
                
                predictions_assigned += len(page.tokens)
        
        # Set token types in parallel
        self._set_token_types_parallel()
    
    def _set_token_types_parallel(self):
        """Set token types in parallel across all PDFs."""
        from pdf_token_type_labels.TokenType import TokenType
        
        all_tokens = []
        for pdf_features in self.pdfs_features:
            for page in pdf_features.pages:
                all_tokens.extend(page.tokens)
        
        # Batch process token type assignment
        if len(all_tokens) > 1000:  # Parallel for large numbers
            chunk_size = max(100, len(all_tokens) // self.max_workers)
            chunks = [all_tokens[i:i + chunk_size] for i in range(0, len(all_tokens), chunk_size)]
            
            with ThreadPoolExecutor(max_workers=min(len(chunks), self.max_workers)) as executor:
                futures = []
                for chunk in chunks:
                    future = executor.submit(self._set_chunk_token_types, chunk)
                    futures.append(future)
                
                # Wait for all to complete
                for future in as_completed(futures):
                    future.result()
        else:
            # Sequential for small numbers
            self._set_chunk_token_types(all_tokens)
    
    def _set_chunk_token_types(self, tokens: List[PdfToken]):
        """Set token types for a chunk of tokens."""
        from pdf_token_type_labels.TokenType import TokenType
        
        for token in tokens:
            token.token_type = TokenType.from_index(token.prediction)


class OptimizedParagraphExtractorTrainer(ParagraphExtractorTrainer):
    """Optimized paragraph extractor with parallel processing."""
    
    def __init__(self, pdfs_features: list[PdfFeatures], model_configuration, max_workers: int = None):
        super().__init__(pdfs_features, model_configuration)
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self._model_cache = {}
    
    def get_pdf_segments_optimized(self, paragraph_extractor_model_path: str) -> list:
        """Optimized PDF segment extraction with parallel processing."""
        paragraphs = self.get_paragraphs_parallel(paragraph_extractor_model_path)
        
        # Parallel segment creation
        if len(paragraphs) > 100:
            return self._create_segments_parallel(paragraphs)
        else:
            return self._create_segments_sequential(paragraphs)
    
    def get_paragraphs_parallel(self, paragraph_extractor_model_path: str) -> list:
        """Get paragraphs with parallel prediction."""
        # Use optimized prediction
        self.predict_optimized(paragraph_extractor_model_path)
        
        # Process paragraphs in parallel
        if len(self.pdfs_features) > 1:
            return self._extract_paragraphs_multi_pdf()
        else:
            return self._extract_paragraphs_single_pdf()
    
    def predict_optimized(self, model_path: str):
        """Optimized prediction with caching and batching."""
        # Cache model loading
        if model_path not in self._model_cache:
            import lightgbm as lgb
            self._model_cache[model_path] = lgb.Booster(model_file=model_path)
        
        # Use parallel model input generation
        x = self.get_model_input_parallel()
        
        if not x.any():
            return
        
        lightgbm_model = self._model_cache[model_path]
        
        # Batch prediction
        batch_size = 10000
        if len(x) > batch_size:
            predictions = []
            for i in range(0, len(x), batch_size):
                batch = x[i:i + batch_size]
                batch_predictions = lightgbm_model.predict(batch)
                predictions.append(batch_predictions)
            all_predictions = np.concatenate(predictions)
        else:
            all_predictions = lightgbm_model.predict(x)
        
        # Assign predictions in parallel
        self._assign_predictions_parallel(all_predictions)
    
    def get_model_input_parallel(self) -> np.ndarray:
        """Parallel model input generation."""
        features_rows = []
        
        if len(self.pdfs_features) > 1:
            # Multi-PDF parallel processing
            with ThreadPoolExecutor(max_workers=min(len(self.pdfs_features), self.max_workers)) as executor:
                futures = []
                for pdf_features in self.pdfs_features:
                    future = executor.submit(self._get_pdf_model_input, pdf_features)
                    futures.append(future)
                
                for future in as_completed(futures):
                    features_rows.extend(future.result())
        else:
            # Single PDF parallel page processing
            for pdf_features in self.pdfs_features:
                pdf_features_rows = self._get_pdf_model_input_parallel(pdf_features)
                features_rows.extend(pdf_features_rows)
        
        return self.features_rows_to_x(features_rows)
    
    def _get_pdf_model_input(self, pdf_features: PdfFeatures) -> List[List[float]]:
        """Get model input for a single PDF."""
        from pdf_tokens_type_trainer.TokenFeatures import TokenFeatures
        
        token_features = TokenFeatures(pdf_features)
        features_rows = []
        
        for page, token, next_token in self._loop_token_next_token_single_pdf(pdf_features):
            page_tokens = [
                self.get_padding_token(segment_number=i - 999999, page_number=page.page_number) 
                for i in range(self.model_configuration.context_size)
            ]
            page_tokens += page.tokens
            page_tokens += [
                self.get_padding_token(segment_number=999999 + i, page_number=page.page_number) 
                for i in range(self.model_configuration.context_size)
            ]
            
            token_index = page_tokens.index(token) if token in page_tokens else self.model_configuration.context_size
            context_features = self.get_context_features(token_features, page_tokens, token_index)
            features_rows.append(context_features)
        
        return features_rows
    
    def _get_pdf_model_input_parallel(self, pdf_features: PdfFeatures) -> List[List[float]]:
        """Get model input for a single PDF with parallel page processing."""
        from pdf_tokens_type_trainer.TokenFeatures import TokenFeatures
        
        token_features = TokenFeatures(pdf_features)
        
        # Get pages with tokens
        pages_with_data = []
        for page in pdf_features.pages:
            if not page.tokens:
                continue
            if len(page.tokens) == 1:
                pages_with_data.append((page, [(page.tokens[0], page.tokens[0])]))
            else:
                token_pairs = list(zip(page.tokens, page.tokens[1:]))
                pages_with_data.append((page, token_pairs))
        
        # Process pages in parallel
        if len(pages_with_data) > 2:
            with ThreadPoolExecutor(max_workers=min(len(pages_with_data), self.max_workers)) as executor:
                futures = []
                for page, token_pairs in pages_with_data:
                    future = executor.submit(
                        self._process_page_model_input, 
                        page, token_pairs, token_features
                    )
                    futures.append(future)
                
                features_rows = []
                for future in as_completed(futures):
                    features_rows.extend(future.result())
        else:
            features_rows = []
            for page, token_pairs in pages_with_data:
                page_features = self._process_page_model_input(page, token_pairs, token_features)
                features_rows.extend(page_features)
        
        return features_rows
    
    def _process_page_model_input(self, page: PdfPage, token_pairs: List[Tuple], token_features) -> List[List[float]]:
        """Process model input for a single page."""
        page_tokens = [
            self.get_padding_token(segment_number=i - 999999, page_number=page.page_number) 
            for i in range(self.model_configuration.context_size)
        ]
        page_tokens += page.tokens
        page_tokens += [
            self.get_padding_token(segment_number=999999 + i, page_number=page.page_number) 
            for i in range(self.model_configuration.context_size)
        ]
        
        features_rows = []
        for token, next_token in token_pairs:
            try:
                token_index = page_tokens.index(token)
                context_features = self.get_context_features(token_features, page_tokens, token_index)
                features_rows.append(context_features)
            except ValueError:
                # Token not found, skip
                continue
        
        return features_rows
    
    def _loop_token_next_token_single_pdf(self, pdf_features: PdfFeatures):
        """Loop through tokens for a single PDF."""
        for page in pdf_features.pages:
            if not page.tokens:
                continue
            if len(page.tokens) == 1:
                yield page, page.tokens[0], page.tokens[0]
            for token, next_token in zip(page.tokens, page.tokens[1:]):
                yield page, token, next_token
    
    def _assign_predictions_parallel(self, predictions: np.ndarray):
        """Assign predictions to tokens in parallel."""
        predictions_assigned = 0
        
        for pdf_features in self.pdfs_features:
            for page in pdf_features.pages:
                if not page.tokens:
                    continue
                
                if len(page.tokens) == 1:
                    page.tokens[0].prediction = int(np.argmax(predictions[predictions_assigned]))
                    predictions_assigned += 1
                else:
                    for i, (token, next_token) in enumerate(zip(page.tokens, page.tokens[1:])):
                        token.prediction = int(np.argmax(predictions[predictions_assigned + i]))
                    predictions_assigned += len(page.tokens) - 1
    
    def _extract_paragraphs_single_pdf(self) -> list:
        """Extract paragraphs for single PDF."""
        from fast_trainer.Paragraph import Paragraph
        
        paragraphs = []
        last_page = None
        
        for page, token, next_token in self.loop_token_next_token():
            if last_page != page:
                last_page = page
                paragraphs.append(Paragraph([token], page.pdf_name))
            if token == next_token:
                continue
            if token.prediction:
                paragraphs[-1].add_token(next_token)
                continue
            paragraphs.append(Paragraph([next_token], page.pdf_name))
        
        return paragraphs
    
    def _extract_paragraphs_multi_pdf(self) -> list:
        """Extract paragraphs for multiple PDFs in parallel."""
        with ThreadPoolExecutor(max_workers=min(len(self.pdfs_features), self.max_workers)) as executor:
            futures = []
            for pdf_features in self.pdfs_features:
                future = executor.submit(self._extract_paragraphs_for_pdf, pdf_features)
                futures.append(future)
            
            all_paragraphs = []
            for future in as_completed(futures):
                all_paragraphs.extend(future.result())
        
        return all_paragraphs
    
    def _extract_paragraphs_for_pdf(self, pdf_features: PdfFeatures) -> list:
        """Extract paragraphs for a specific PDF."""
        from fast_trainer.Paragraph import Paragraph
        
        paragraphs = []
        last_page = None
        
        for page in pdf_features.pages:
            if not page.tokens:
                continue
            
            if last_page != page:
                last_page = page
                paragraphs.append(Paragraph([page.tokens[0]], page.pdf_name))
            
            if len(page.tokens) == 1:
                continue
            
            for token, next_token in zip(page.tokens, page.tokens[1:]):
                if token.prediction:
                    paragraphs[-1].add_token(next_token)
                    continue
                paragraphs.append(Paragraph([next_token], page.pdf_name))
        
        return paragraphs
    
    def _create_segments_parallel(self, paragraphs: list) -> list:
        """Create PDF segments in parallel."""
        from fast_trainer.PdfSegment import PdfSegment
        
        chunk_size = max(10, len(paragraphs) // self.max_workers)
        chunks = [paragraphs[i:i + chunk_size] for i in range(0, len(paragraphs), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=min(len(chunks), self.max_workers)) as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(self._create_segments_chunk, chunk)
                futures.append(future)
            
            all_segments = []
            for future in as_completed(futures):
                all_segments.extend(future.result())
        
        return all_segments
    
    def _create_segments_chunk(self, paragraphs_chunk: list) -> list:
        """Create segments for a chunk of paragraphs."""
        from fast_trainer.PdfSegment import PdfSegment
        
        return [
            PdfSegment.from_pdf_tokens(paragraph.tokens, paragraph.pdf_name) 
            for paragraph in paragraphs_chunk
        ]
    
    def _create_segments_sequential(self, paragraphs: list) -> list:
        """Create PDF segments sequentially."""
        from fast_trainer.PdfSegment import PdfSegment
        
        return [
            PdfSegment.from_pdf_tokens(paragraph.tokens, paragraph.pdf_name) 
            for paragraph in paragraphs
        ]


class BatchProcessorFactory:
    """Factory for creating optimized batch processors."""
    
    @staticmethod
    def create_token_type_trainer(pdfs_features: list[PdfFeatures], 
                                 model_configuration: ModelConfiguration = None,
                                 max_workers: int = None) -> OptimizedTokenTypeTrainer:
        """Create an optimized token type trainer."""
        return OptimizedTokenTypeTrainer(pdfs_features, model_configuration, max_workers)
    
    @staticmethod
    def create_paragraph_extractor(pdfs_features: list[PdfFeatures],
                                  model_configuration,
                                  max_workers: int = None) -> OptimizedParagraphExtractorTrainer:
        """Create an optimized paragraph extractor trainer."""
        return OptimizedParagraphExtractorTrainer(pdfs_features, model_configuration, max_workers) 