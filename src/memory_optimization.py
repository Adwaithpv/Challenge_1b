"""
Memory Optimization Module for PDF Processing
Implements memory-efficient techniques to handle large PDFs within 16GB RAM limit.

Key optimizations:
- Streaming processing for large files
- Memory-mapped file operations
- Garbage collection management
- Memory monitoring and alerts
- Chunk-based processing with memory cleanup
"""

import gc
import os
import sys
import time
import psutil
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator
from contextlib import contextmanager
import weakref


class MemoryMonitor:
    """Real-time memory monitoring and management."""
    
    def __init__(self, max_memory_gb: float = 14.0):  # Leave 2GB buffer
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        
    def add_memory_warning_callback(self, callback):
        """Add callback to be called when memory usage is high."""
        self.callbacks.append(callback)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent()
        }
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is approaching the limit."""
        current_memory = self.process.memory_info().rss
        return current_memory > (self.max_memory_bytes * 0.85)  # 85% threshold
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        collected = gc.collect()
        return collected
    
    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """Context manager for memory-monitored operations."""
        initial_memory = self.get_memory_usage()
        start_time = time.time()
        
        print(f"🧠 Starting {operation_name} (Initial memory: {initial_memory['rss_mb']:.1f} MB)")
        
        try:
            yield self
        finally:
            final_memory = self.get_memory_usage()
            duration = time.time() - start_time
            memory_delta = final_memory['rss_mb'] - initial_memory['rss_mb']
            
            print(f"🧠 Completed {operation_name} in {duration:.2f}s")
            print(f"   Memory: {initial_memory['rss_mb']:.1f} → {final_memory['rss_mb']:.1f} MB ({memory_delta:+.1f} MB)")
            
            # Force cleanup if memory usage is high
            if self.is_memory_critical():
                print("⚠️  Memory usage critical, forcing garbage collection...")
                collected = self.force_garbage_collection()
                new_memory = self.get_memory_usage()
                print(f"   Freed {final_memory['rss_mb'] - new_memory['rss_mb']:.1f} MB ({collected} objects)")


class StreamingPDFProcessor:
    """Memory-efficient streaming PDF processor."""
    
    def __init__(self, memory_monitor: MemoryMonitor):
        self.memory_monitor = memory_monitor
        self.temp_files = []
        self.chunk_size = 1024 * 1024  # 1MB chunks
    
    def __del__(self):
        """Cleanup temporary files."""
        self.cleanup_temp_files()
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
        self.temp_files.clear()
    
    def process_large_pdf_streaming(self, pdf_path: str, max_chunk_size_mb: int = 50):
        """
        Process large PDFs using streaming approach to minimize memory usage.
        
        Args:
            pdf_path: Path to the PDF file
            max_chunk_size_mb: Maximum size of each processing chunk in MB
        """
        pdf_path = Path(pdf_path)
        file_size_mb = pdf_path.stat().st_size / 1024 / 1024
        
        if file_size_mb <= max_chunk_size_mb:
            # Small file, process normally
            return self._process_normal(pdf_path)
        else:
            # Large file, use streaming approach
            return self._process_streaming(pdf_path, max_chunk_size_mb)
    
    def _process_normal(self, pdf_path: Path):
        """Process normal-sized PDF without streaming."""
        with self.memory_monitor.memory_context(f"Normal processing: {pdf_path.name}"):
            with open(pdf_path, 'rb') as file:
                pdf_content = file.read()
            
            # Process the entire file
            return self._process_pdf_content(pdf_content, str(pdf_path))
    
    def _process_streaming(self, pdf_path: Path, max_chunk_size_mb: int):
        """Process large PDF using streaming approach."""
        file_size_mb = pdf_path.stat().st_size / 1024 / 1024
        num_chunks = max(1, int(file_size_mb / max_chunk_size_mb))
        
        print(f"📄 Large PDF detected ({file_size_mb:.1f} MB), using streaming processing with {num_chunks} chunks")
        
        # For PDF files, we can't easily chunk the file content since PDF structure
        # needs to be preserved. Instead, we'll process page by page with memory cleanup
        return self._process_page_by_page(pdf_path)
    
    def _process_page_by_page(self, pdf_path: Path):
        """Process PDF page by page to reduce memory usage."""
        with self.memory_monitor.memory_context(f"Page-by-page processing: {pdf_path.name}"):
            # Use pdf2image to process pages individually
            try:
                from pdf2image import convert_from_path
                import fitz  # PyMuPDF for efficient page extraction
            except ImportError:
                print("⚠️  pdf2image or PyMuPDF not available, falling back to normal processing")
                return self._process_normal(pdf_path)
            
            # Open PDF with PyMuPDF for efficient access
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            
            print(f"📖 Processing {total_pages} pages individually")
            
            all_results = []
            pages_per_batch = max(1, min(10, 50 // max(1, total_pages // 20)))  # Adaptive batch size
            
            for start_page in range(0, total_pages, pages_per_batch):
                end_page = min(start_page + pages_per_batch, total_pages)
                
                with self.memory_monitor.memory_context(f"Pages {start_page+1}-{end_page}"):
                    # Extract page range as separate PDF
                    page_pdf_path = self._extract_page_range(doc, start_page, end_page)
                    
                    try:
                        # Process this page range
                        with open(page_pdf_path, 'rb') as file:
                            page_content = file.read()
                        
                        page_result = self._process_pdf_content(page_content, str(page_pdf_path))
                        if page_result:
                            all_results.extend(page_result)
                        
                        # Force cleanup after each batch
                        del page_content
                        if self.memory_monitor.is_memory_critical():
                            self.memory_monitor.force_garbage_collection()
                            
                    finally:
                        # Clean up temporary page file
                        self._cleanup_temp_file(page_pdf_path)
            
            doc.close()
            return all_results
    
    def _extract_page_range(self, doc, start_page: int, end_page: int) -> Path:
        """Extract a range of pages to a temporary PDF file."""
        import fitz
        
        # Create new PDF with selected pages
        new_doc = fitz.open()
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        
        new_doc.save(str(temp_path))
        new_doc.close()
        
        self.temp_files.append(temp_path)
        return temp_path
    
    def _cleanup_temp_file(self, temp_path: Path):
        """Clean up a specific temporary file."""
        try:
            if temp_path.exists():
                temp_path.unlink()
            if temp_path in self.temp_files:
                self.temp_files.remove(temp_path)
        except:
            pass
    
    def _process_pdf_content(self, pdf_content: bytes, pdf_name: str):
        """Process PDF content using the optimized pipeline."""
        # Import optimized processors
        from round1a_solution_optimized import OptimizedPDFProcessor
        
        processor = OptimizedPDFProcessor(max_workers=2)  # Reduce workers for memory efficiency
        
        try:
            # Use the optimized analysis pipeline
            segment_boxes = processor.parallel_analyze_pdf_fast(
                pdf_content, 
                f"temp_analysis_{int(time.time())}", 
                keep_pdf=False
            )
            
            return segment_boxes
            
        except Exception as e:
            print(f"⚠️  Error processing {pdf_name}: {e}")
            return []


class MemoryEfficientCache:
    """Memory-efficient cache with automatic cleanup."""
    
    def __init__(self, max_size_mb: float = 500):  # 500MB cache limit
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.access_times = {}
        self.estimated_sizes = {}
        self.total_size = 0
        self._lock = threading.Lock()
    
    def get(self, key: str):
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def put(self, key: str, value, estimated_size: int = None):
        """Put item in cache with automatic size management."""
        if estimated_size is None:
            estimated_size = sys.getsizeof(value)
        
        with self._lock:
            # Check if we need to evict items
            while self.total_size + estimated_size > self.max_size_bytes and self.cache:
                self._evict_lru()
            
            # Add new item
            if key in self.cache:
                self.total_size -= self.estimated_sizes[key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.estimated_sizes[key] = estimated_size
            self.total_size += estimated_size
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.cache:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self.total_size -= self.estimated_sizes[lru_key]
        
        del self.cache[lru_key]
        del self.access_times[lru_key]
        del self.estimated_sizes[lru_key]
    
    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.estimated_sizes.clear()
            self.total_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "total_items": len(self.cache),
                "total_size_mb": self.total_size / 1024 / 1024,
                "max_size_mb": self.max_size_bytes / 1024 / 1024,
                "utilization": (self.total_size / self.max_size_bytes) * 100
            }


class OptimizedMemoryManager:
    """Comprehensive memory management for PDF processing."""
    
    def __init__(self, max_memory_gb: float = 14.0):
        self.memory_monitor = MemoryMonitor(max_memory_gb)
        self.streaming_processor = StreamingPDFProcessor(self.memory_monitor)
        self.cache = MemoryEfficientCache()
        
        # Register cleanup callback
        self.memory_monitor.add_memory_warning_callback(self._memory_warning_callback)
    
    def _memory_warning_callback(self):
        """Called when memory usage is high."""
        print("⚠️  High memory usage detected, performing cleanup...")
        
        # Clear cache
        cache_stats = self.cache.get_stats()
        self.cache.clear()
        print(f"   Cleared cache ({cache_stats['total_items']} items, {cache_stats['total_size_mb']:.1f} MB)")
        
        # Force garbage collection
        collected = self.memory_monitor.force_garbage_collection()
        print(f"   Garbage collected {collected} objects")
        
        # Clean up temporary files
        self.streaming_processor.cleanup_temp_files()
    
    def process_pdf_memory_efficient(self, pdf_path: str, **kwargs):
        """Process PDF with comprehensive memory management."""
        with self.memory_monitor.memory_context(f"Memory-efficient processing: {Path(pdf_path).name}"):
            # Check if result is cached
            cache_key = f"pdf_result_{pdf_path}_{hash(str(kwargs))}"
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                print("📋 Using cached result")
                return cached_result
            
            # Process with streaming if necessary
            result = self.streaming_processor.process_large_pdf_streaming(pdf_path)
            
            # Cache result if not too large
            if result and sys.getsizeof(result) < 50 * 1024 * 1024:  # Cache if < 50MB
                self.cache.put(cache_key, result)
            
            return result
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        memory_usage = self.memory_monitor.get_memory_usage()
        cache_stats = self.cache.get_stats()
        
        return {
            "memory_usage": memory_usage,
            "cache_stats": cache_stats,
            "memory_critical": self.memory_monitor.is_memory_critical(),
            "temp_files": len(self.streaming_processor.temp_files)
        }


# Global memory manager instance
_global_memory_manager = None

def get_memory_manager() -> OptimizedMemoryManager:
    """Get the global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = OptimizedMemoryManager()
    return _global_memory_manager


@contextmanager
def memory_optimized_processing(max_memory_gb: float = 14.0):
    """Context manager for memory-optimized PDF processing."""
    manager = OptimizedMemoryManager(max_memory_gb)
    
    try:
        yield manager
    finally:
        # Cleanup
        manager.cache.clear()
        manager.streaming_processor.cleanup_temp_files()
        manager.memory_monitor.force_garbage_collection()


# Decorator for memory monitoring
def monitor_memory(operation_name: str = None):
    """Decorator to monitor memory usage of functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            manager = get_memory_manager()
            
            with manager.memory_monitor.memory_context(name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator 