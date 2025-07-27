#!/usr/bin/env python3
"""
Performance Benchmark Script for PDF Processing Optimization
Compares original solution vs optimized solution performance.

Usage:
    python performance_benchmark.py test_pdfs/NEP.pdf
    python performance_benchmark.py test_pdfs/
"""

import sys
import time
import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any
import traceback
import psutil
import multiprocessing as mp

# Force CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add src to path for imports
sys.path.append('src')


class PerformanceBenchmark:
    """Performance benchmark comparing original vs optimized solutions."""
    
    def __init__(self):
        self.results = []
        
    def run_original_solution(self, pdf_path: str) -> Dict[str, Any]:
        """Run the original solution and measure performance."""
        from round1a_solution import Round1ASolutionEngine
        
        print(f"📊 Running ORIGINAL solution on {pdf_path}")
        
        # Memory tracking
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        try:
            engine = Round1ASolutionEngine()
            result = engine.extract_document_structure(pdf_path)
            
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                "success": True,
                "processing_time": end_time - start_time,
                "memory_used": final_memory - initial_memory,
                "peak_memory": final_memory,
                "result": result,
                "error": None
            }
            
        except Exception as e:
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                "success": False,
                "processing_time": end_time - start_time,
                "memory_used": final_memory - initial_memory,
                "peak_memory": final_memory,
                "result": None,
                "error": str(e)
            }
    
    def run_optimized_solution(self, pdf_path: str, max_workers: int = None) -> Dict[str, Any]:
        """Run the optimized solution and measure performance."""
        from round1a_solution_optimized import OptimizedRound1ASolutionEngine
        
        workers = max_workers or min(mp.cpu_count(), 8)
        print(f"🚀 Running OPTIMIZED solution on {pdf_path} (workers: {workers})")
        
        # Memory tracking
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        try:
            engine = OptimizedRound1ASolutionEngine(max_workers=max_workers)
            result = engine.extract_document_structure(pdf_path)
            
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                "success": True,
                "processing_time": end_time - start_time,
                "memory_used": final_memory - initial_memory,
                "peak_memory": final_memory,
                "result": result,
                "error": None,
                "max_workers": workers
            }
            
        except Exception as e:
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                "success": False,
                "processing_time": end_time - start_time,
                "memory_used": final_memory - initial_memory,
                "peak_memory": final_memory,
                "result": None,
                "error": str(e),
                "max_workers": workers
            }
    
    def compare_accuracy(self, original_result: Dict, optimized_result: Dict) -> Dict[str, Any]:
        """Compare accuracy between original and optimized results."""
        if not original_result.get("success") or not optimized_result.get("success"):
            return {"accuracy_comparison": "N/A - One or both solutions failed"}
        
        orig_data = original_result["result"]
        opt_data = optimized_result["result"]
        
        # Compare titles
        title_match = orig_data.get("title", "") == opt_data.get("title", "")
        
        # Compare outline counts
        orig_outline = orig_data.get("outline", [])
        opt_outline = opt_data.get("outline", [])
        outline_count_match = len(orig_outline) == len(opt_outline)
        
        # Compare outline text similarity
        text_matches = 0
        if orig_outline and opt_outline:
            min_len = min(len(orig_outline), len(opt_outline))
            for i in range(min_len):
                if orig_outline[i].get("text", "") == opt_outline[i].get("text", ""):
                    text_matches += 1
        
        text_similarity = text_matches / max(len(orig_outline), len(opt_outline)) if (orig_outline or opt_outline) else 1.0
        
        return {
            "title_match": title_match,
            "outline_count_match": outline_count_match,
            "outline_count_original": len(orig_outline),
            "outline_count_optimized": len(opt_outline),
            "text_similarity": text_similarity,
            "overall_accuracy": "GOOD" if title_match and outline_count_match and text_similarity > 0.9 else "PARTIAL"
        }
    
    def benchmark_file(self, pdf_path: str, max_workers: int = None) -> Dict[str, Any]:
        """Benchmark a single PDF file."""
        pdf_path = Path(pdf_path)
        file_size = pdf_path.stat().st_size / 1024 / 1024  # MB
        
        print(f"\n{'='*60}")
        print(f"📄 BENCHMARKING: {pdf_path.name}")
        print(f"📏 File size: {file_size:.2f} MB")
        print(f"{'='*60}")
        
        # Run original solution
        original_result = self.run_original_solution(str(pdf_path))
        
        # Wait a moment for memory to stabilize
        time.sleep(2)
        
        # Run optimized solution
        optimized_result = self.run_optimized_solution(str(pdf_path), max_workers)
        
        # Compare accuracy
        accuracy_comparison = self.compare_accuracy(original_result, optimized_result)
        
        # Calculate performance improvements
        performance_improvement = self._calculate_improvements(original_result, optimized_result)
        
        benchmark_result = {
            "file_name": pdf_path.name,
            "file_size_mb": file_size,
            "original": original_result,
            "optimized": optimized_result,
            "accuracy": accuracy_comparison,
            "performance": performance_improvement
        }
        
        self._print_benchmark_results(benchmark_result)
        
        return benchmark_result
    
    def _calculate_improvements(self, original: Dict, optimized: Dict) -> Dict[str, Any]:
        """Calculate performance improvements."""
        if not original.get("success") or not optimized.get("success"):
            return {"status": "Cannot compare - one or both solutions failed"}
        
        orig_time = original["processing_time"]
        opt_time = optimized["processing_time"]
        
        time_improvement = ((orig_time - opt_time) / orig_time) * 100 if orig_time > 0 else 0
        speedup_factor = orig_time / opt_time if opt_time > 0 else 0
        
        orig_memory = original["memory_used"]
        opt_memory = optimized["memory_used"]
        memory_improvement = ((orig_memory - opt_memory) / orig_memory) * 100 if orig_memory > 0 else 0
        
        target_time = 10.0  # seconds
        orig_meets_target = orig_time <= target_time
        opt_meets_target = opt_time <= target_time
        
        return {
            "time_improvement_percent": time_improvement,
            "speedup_factor": speedup_factor,
            "memory_improvement_percent": memory_improvement,
            "original_meets_target": orig_meets_target,
            "optimized_meets_target": opt_meets_target,
            "original_time": orig_time,
            "optimized_time": opt_time,
            "original_memory": orig_memory,
            "optimized_memory": opt_memory
        }
    
    def _print_benchmark_results(self, result: Dict[str, Any]):
        """Print formatted benchmark results."""
        print(f"\n📊 BENCHMARK RESULTS:")
        print(f"{'─'*50}")
        
        # Performance comparison
        perf = result["performance"]
        if "status" in perf:
            print(f"❌ {perf['status']}")
            return
        
        print(f"⏱️  TIMING:")
        print(f"   Original:   {perf['original_time']:.2f}s")
        print(f"   Optimized:  {perf['optimized_time']:.2f}s")
        print(f"   Improvement: {perf['time_improvement_percent']:+.1f}% ({perf['speedup_factor']:.2f}x faster)")
        
        print(f"\n💾 MEMORY:")
        print(f"   Original:   {perf['original_memory']:.1f} MB")
        print(f"   Optimized:  {perf['optimized_memory']:.1f} MB")
        print(f"   Improvement: {perf['memory_improvement_percent']:+.1f}%")
        
        print(f"\n🎯 TARGET COMPLIANCE (<10s):")
        print(f"   Original:   {'✅' if perf['original_meets_target'] else '❌'}")
        print(f"   Optimized:  {'✅' if perf['optimized_meets_target'] else '❌'}")
        
        # Accuracy comparison
        acc = result["accuracy"]
        print(f"\n🎯 ACCURACY:")
        print(f"   Title match:        {'✅' if acc.get('title_match') else '❌'}")
        print(f"   Outline count:      {acc.get('outline_count_original')} → {acc.get('outline_count_optimized')} {'✅' if acc.get('outline_count_match') else '❌'}")
        print(f"   Text similarity:    {acc.get('text_similarity', 0):.1%}")
        print(f"   Overall accuracy:   {acc.get('overall_accuracy', 'N/A')}")
        
        # Workers info
        if result["optimized"].get("max_workers"):
            print(f"\n⚙️  CONFIGURATION:")
            print(f"   Workers used: {result['optimized']['max_workers']}")
            print(f"   CPU cores:    {mp.cpu_count()}")
    
    def benchmark_directory(self, pdf_dir: str, max_workers: int = None) -> List[Dict[str, Any]]:
        """Benchmark all PDF files in a directory."""
        pdf_dir = Path(pdf_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return []
        
        print(f"🔍 Found {len(pdf_files)} PDF files in {pdf_dir}")
        
        all_results = []
        for pdf_file in pdf_files:
            try:
                result = self.benchmark_file(str(pdf_file), max_workers)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error benchmarking {pdf_file}: {e}")
                traceback.print_exc()
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of all benchmark results."""
        if not results:
            return
        
        print(f"\n{'='*70}")
        print(f"📈 BENCHMARK SUMMARY ({len(results)} files)")
        print(f"{'='*70}")
        
        successful_results = [r for r in results if r["performance"].get("speedup_factor")]
        if not successful_results:
            print("❌ No successful comparisons")
            return
        
        # Calculate averages
        avg_speedup = sum(r["performance"]["speedup_factor"] for r in successful_results) / len(successful_results)
        avg_time_improvement = sum(r["performance"]["time_improvement_percent"] for r in successful_results) / len(successful_results)
        avg_memory_improvement = sum(r["performance"]["memory_improvement_percent"] for r in successful_results) / len(successful_results)
        
        # Target compliance
        orig_target_met = sum(1 for r in successful_results if r["performance"]["original_meets_target"])
        opt_target_met = sum(1 for r in successful_results if r["performance"]["optimized_meets_target"])
        
        print(f"⚡ AVERAGE IMPROVEMENTS:")
        print(f"   Speed:      {avg_speedup:.2f}x faster ({avg_time_improvement:+.1f}%)")
        print(f"   Memory:     {avg_memory_improvement:+.1f}%")
        
        print(f"\n🎯 TARGET COMPLIANCE (<10s):")
        print(f"   Original:   {orig_target_met}/{len(successful_results)} files")
        print(f"   Optimized:  {opt_target_met}/{len(successful_results)} files")
        print(f"   Improvement: +{opt_target_met - orig_target_met} files")
        
        # Best/worst performance
        best_speedup = max(successful_results, key=lambda r: r["performance"]["speedup_factor"])
        print(f"\n🏆 BEST SPEEDUP: {best_speedup['file_name']} ({best_speedup['performance']['speedup_factor']:.2f}x)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark PDF processing performance: original vs optimized"
    )
    parser.add_argument("path", help="Path to PDF file or directory containing PDFs")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker threads for optimized version"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file to save detailed results"
    )
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    
    path = Path(args.path)
    if not path.exists():
        print(f"❌ Path does not exist: {path}")
        sys.exit(1)
    
    print(f"🎯 PDF Processing Performance Benchmark")
    print(f"CPU cores available: {mp.cpu_count()}")
    print(f"Workers for optimized version: {args.max_workers or 'auto-detect'}")
    
    if path.is_file():
        results = [benchmark.benchmark_file(str(path), args.max_workers)]
    else:
        results = benchmark.benchmark_directory(str(path), args.max_workers)
    
    # Save detailed results if requested
    if args.output and results:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main() 