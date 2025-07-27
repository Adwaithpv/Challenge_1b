#!/usr/bin/env python3
"""
Setup script for Round 1A Solution
Helps users install dependencies and verify the setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from setuptools import setup, find_packages

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"Python version: {sys.version}")
    return True

def check_poppler():
    """Check if poppler-utils is installed."""
    try:
        result = subprocess.run(['pdftohtml', '-v'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("Poppler-utils is installed")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("Poppler-utils not found")
    print("\nInstall Poppler-utils:")
    
    system = platform.system().lower()
    if system == "windows":
        print("  winget install oschwartz10612.Poppler_Microsoft.Winget.Source")
        print("  Or download from: https://github.com/oschwartz10612/poppler-windows/releases")
    elif system == "darwin":  # macOS
        print("  brew install poppler")
    else:  # Linux
        print("  sudo apt-get update && sudo apt-get install poppler-utils")
    
    return False

def install_python_dependencies():
    """Install Python dependencies."""
    print("\nInstalling Python dependencies...")
    
    try:
        # Install PyTorch CPU version first
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "torch", "torchvision", "--index-url", 
                       "https://download.pytorch.org/whl/cpu"], 
                      check=True)
        
        # Install other dependencies
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        
        # Install detectron2 separately
        print("Installing detectron2...")
        subprocess.run([sys.executable, "-m", "pip", "install", "detectron2", "-f", 
                       "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html"], 
                      check=True)
        
        print("Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Python dependencies: {e}")
        return False

def verify_installation():
    """Verify that all components are working."""
    print("\nVerifying installation...")
    
    try:
        # Test imports
        import torch
        import lightgbm
        import pydantic
        import lxml
        from PIL import Image
        import numpy as np
        
        print("All Python packages imported successfully")
        
        # Test CPU-only configuration
        if not torch.cuda.is_available():
            print("CUDA not available - CPU-only mode confirmed")
        else:
            print("CUDA is available but will be disabled by environment variables")
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def create_test_script():
    """Create a simple test script."""
    test_script = """#!/usr/bin/env python3
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
"""
    
    with open("test_setup.py", "w", encoding='utf-8') as f:
        f.write(test_script)
    
    print("Created test_setup.py - run 'python test_setup.py' to verify")

# Package configuration for setuptools
setup(
    name="round1a-solution",
    version="1.0.0",
    description="Adobe India Hackathon Round 1A PDF Processing Solution",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "lightgbm>=3.3.0",
        "numpy>=1.21.0",
        "opencv-python-headless>=4.5.0",
        "pdf2image>=1.17.0",
        "pydantic>=1.8.0",
        "lxml>=4.6.0",
        "pillow>=8.3.0",
        "pypandoc>=1.8.0",
        "fvcore>=0.1.5",
        "omegaconf>=2.1.0",
        "pycocotools>=2.0.4",
        "psutil>=5.8.0",
        "PyMuPDF>=1.20.0",
        "wheel>=0.37.0",
        "setuptools>=58.0.0",
    ],
    extras_require={
        "torch": ["torch", "torchvision"],
        "detectron2": ["detectron2"],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.pickle", "*.model"],
    },
    zip_safe=False,
)

def main():
    """Main setup function."""
    print("Round 1A Solution Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check poppler
    poppler_ok = check_poppler()
    
    # Install Python dependencies
    deps_ok = install_python_dependencies()
    
    # Verify installation
    verify_ok = verify_installation()
    
    # Create test script
    create_test_script()
    
    print("\n" + "=" * 50)
    if poppler_ok and deps_ok and verify_ok:
        print("Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python test_setup.py")
        print("2. Test with: python round1a_solution.py test_pdfs/NEP.pdf --pretty")
        print("3. Read README.md for detailed usage instructions")
    else:
        print("Setup completed with warnings. Please check the issues above.")
        print("\nTroubleshooting:")
        print("1. Install poppler-utils if not already installed")
        print("2. Check Python version (3.8+ required)")
        print("3. Try running: python test_setup.py")
        print("4. Read README.md for detailed troubleshooting")

if __name__ == "__main__":
    main() 