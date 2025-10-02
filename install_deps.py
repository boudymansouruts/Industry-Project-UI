#!/usr/bin/env python3
"""
Install dependencies from requirements.txt for SageMaker environment
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install dependencies from requirements.txt"""
    print("Installing dependencies from requirements.txt...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return False
    
    try:
        # Install from requirements.txt
        print("Installing from requirements.txt...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully from requirements.txt")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install from requirements.txt: {e}")
        
        # Fallback: install core packages individually
        print("Trying individual package installation...")
        core_packages = [
            "soundfile>=0.12.0",
            "librosa>=0.10.0", 
            "scipy>=1.10.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "torch>=2.2.0,<2.8.0",
            "transformers>=4.30.0,<4.50.0",
            "torchaudio>=2.2.0",
            "accelerate>=0.20.0",
            "sentencepiece>=0.1.99",
            "protobuf>=3.20.0",
            "huggingface-hub>=0.15.0",
            "datasets>=2.12.0,<4.2.0"
        ]
        
        for package in core_packages:
            try:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
        
        return False

if __name__ == "__main__":
    install_dependencies()
