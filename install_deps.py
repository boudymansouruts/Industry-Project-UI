#!/usr/bin/env python3
"""
Install missing dependencies for SageMaker environment
"""

import subprocess
import sys

def install_dependencies():
    """Install missing dependencies"""
    print("Installing missing dependencies...")
    
    # Core audio processing dependencies
    packages = [
        "soundfile>=0.12.0",
        "librosa>=0.10.0", 
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "accelerate>=0.20.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "huggingface-hub>=0.15.0",
        "datasets>=2.12.0,<4.2.0"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    print("\nDependencies installation complete!")

if __name__ == "__main__":
    install_dependencies()
