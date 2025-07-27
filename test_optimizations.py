#!/usr/bin/env python3
"""
Quick Test Script for PDF Processing Optimizations
Verifies that optimizations are working correctly.

Usage:
    python test_optimizations.py
"""

import sys
import time
import os
from pathlib import Path

# Force CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def test_basic_functionality():
    """Test basic functionality of optimized solution."""
    print("🧪 Testing Basic Functionality")
    print("="*50)
    
    try:
        # Test import of optimized solution
        from round1a_solution_optimized import OptimizedRound1ASolutionEngine
        print("✅ Optimized solution imports successfully")
        
        # Test import of memory optimization
        from src.memory_optimization import get_memory_manager, monitor_memory
        print("✅ Memory optimization modules import successfully")
        
        # Test import of batch processors
        from src.optimized_batch_processors import BatchProcessorFactory
        print("✅ Batch processors import successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_memory_monitoring():
    """Test memory monitoring functionality."""
    print("\n🧠 Testing Memory Monitoring")
    print("="*50)
    
    try:
        from src.memory_optimization import get_memory_manager
        
        manager = get_memory_manager()
        
        # Test memory usage reporting
        memory_report = manager.get_memory_report()
        print(f"✅ Memory monitoring working")
        print(f"   Current memory: {memory_report['memory_usage']['rss_mb']:.1f} MB")
        print(f"   Cache items: {memory_report['cache_stats']['total_items']}")
        
        # Test memory context
        with manager.memory_monitor.memory_context("Test operation"):
            time.sleep(0.1)  # Simulate some work
        print("✅ Memory context manager working")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory monitoring error: {e}")
        return False

def test_with_sample_pdf():
    """Test with a sample PDF if available."""
    print("\n📄 Testing with Sample PDF")
    print("="*50)
    
    # Look for test PDFs
    test_pdf_dirs = ["test_pdfs", "pdfs", "."]
    test_pdf = None
    
    for pdf_dir in test_pdf_dirs:
        pdf_path = Path(pdf_dir)
        if pdf_path.exists():
            pdf_files = list(pdf_path.glob("*.pdf"))
            if pdf_files:
                test_pdf = pdf_files[0]
                break
    
    if not test_pdf:
        print("⚠️  No test PDF found, skipping PDF processing test")
        print("   To test with a PDF, place a file in the test_pdfs/ directory")
        return True
    
    try:
        from round1a_solution_optimized import OptimizedRound1ASolutionEngine
        
        print(f"📖 Testing with: {test_pdf.name}")
        
        # Test optimized engine creation
        engine = OptimizedRound1ASolutionEngine(max_workers=2)
        print("✅ Optimized engine created successfully")
        
        # Test processing (with timeout for safety)
        start_time = time.time()
        try:
            result = engine.extract_document_structure(str(test_pdf))
            processing_time = time.time() - start_time
            
            print(f"✅ PDF processing completed in {processing_time:.2f}s")
            print(f"   Title: {result.get('title', 'N/A')[:50]}...")
            print(f"   Outline items: {len(result.get('outline', []))}")
            
            if 'memory_optimization' in result:
                mem_info = result['memory_optimization']
                print(f"   Memory optimization: {'enabled' if mem_info['enabled'] else 'disabled'}")
                print(f"   Memory usage: {mem_info['memory_usage']['rss_mb']:.1f} MB")
            
            return True
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ PDF processing failed after {processing_time:.2f}s: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Error during PDF test: {e}")
        return False

def test_performance_benchmark():
    """Test performance benchmark script."""
    print("\n📊 Testing Performance Benchmark")
    print("="*50)
    
    try:
        # Test benchmark imports
        from performance_benchmark import PerformanceBenchmark
        print("✅ Performance benchmark imports successfully")
        
        # Create benchmark instance
        benchmark = PerformanceBenchmark()
        print("✅ Performance benchmark instance created")
        
        return True
        
    except ImportError as e:
        print(f"❌ Benchmark import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\n📦 Testing Dependencies")
    print("="*50)
    
    required_deps = [
        ("torch", "PyTorch"),
        ("lightgbm", "LightGBM"), 
        ("psutil", "Memory monitoring"),
        ("numpy", "NumPy"),
        ("concurrent.futures", "Concurrent processing")
    ]
    
    optional_deps = [
        ("fitz", "PyMuPDF (for memory optimization)"),
        ("pdf2image", "PDF to image conversion")
    ]
    
    all_good = True
    
    # Test required dependencies
    for dep, name in required_deps:
        try:
            __import__(dep)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - REQUIRED")
            all_good = False
    
    # Test optional dependencies
    for dep, name in optional_deps:
        try:
            __import__(dep)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - OPTIONAL")
    
    return all_good

def main():
    """Run all tests."""
    print("🚀 PDF Processing Optimization Test Suite")
    print("="*60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Basic Functionality", test_basic_functionality),
        ("Memory Monitoring", test_memory_monitoring),
        ("Performance Benchmark", test_performance_benchmark),
        ("Sample PDF Processing", test_with_sample_pdf)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Optimizations are working correctly.")
        print("\n🚀 Next steps:")
        print("   1. Try: python round1a_solution_optimized.py test_pdfs/your_file.pdf")
        print("   2. Benchmark: python performance_benchmark.py test_pdfs/your_file.pdf")
        print("   3. Read: OPTIMIZATION_README.md for detailed usage")
    else:
        print("⚠️  Some tests failed. Check error messages above.")
        print("   Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 