#!/usr/bin/env python3
import os
import sys

# Force CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add src to path
sys.path.append('src')

try:
    from pdf_layout_analysis.run_pdf_layout_analysis_fast import analyze_pdf_fast
    from toc.extract_table_of_contents import extract_table_of_contents
    print("All imports successful!")
    print("Setup is complete and ready to use!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run the setup again or check the installation.")
