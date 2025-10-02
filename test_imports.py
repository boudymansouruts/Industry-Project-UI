#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed
"""

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import soundfile as sf
        print("✅ soundfile imported successfully")
    except ImportError as e:
        print(f"❌ soundfile import failed: {e}")
        return False
    
    try:
        import librosa
        print("✅ librosa imported successfully")
    except ImportError as e:
        print(f"❌ librosa import failed: {e}")
        return False
    
    try:
        import scipy
        print("✅ scipy imported successfully")
    except ImportError as e:
        print(f"❌ scipy import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ torch imported successfully")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    try:
        import transformers
        print("✅ transformers imported successfully")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        return False
    
    print("\n✅ All imports successful!")
    return True

if __name__ == "__main__":
    test_imports()
