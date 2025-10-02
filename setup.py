#!/usr/bin/env python
"""
Setup script for BioBERT Health Risk Detection
Helps with initial setup and dependency checking
"""

import sys
import os
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (slower training)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False


def create_directories():
    """Create necessary project directories"""
    directories = [
        "data",
        "models",
        "models/best_model",
        "models/checkpoints",
        "results",
        "logs",
        "cache"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created")


def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    # Check if in virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️  WARNING: Not in a virtual environment!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def download_sample_data():
    """Create sample data for testing"""
    print("\n📊 Creating sample dataset...")
    
    sample_data = """text,emotion
"I feel so hopeless and empty inside",sadness
"I'm constantly worried about everything",fear
"Work pressure is overwhelming me",stressed
"I'm so angry at this situation",angry
"I feel completely alone",lonely
"I'm confused about what to do",confused
"My body aches and I'm exhausted",pain
"I feel so ashamed of myself",ashamed
"Today was amazing and I'm so happy",happy
"I love spending time with family",love
"So excited about the new opportunity",excited
"Feeling calm and peaceful today",calm
"""
    
    data_path = Path("data/dailytalk.csv")
    if not data_path.exists():
        with open(data_path, 'w') as f:
            f.write(sample_data)
        print("✅ Sample data created at data/dailytalk.csv")
    else:
        print("✅ Data file already exists")


def test_import():
    """Test if all modules can be imported"""
    print("\n🔍 Testing imports...")
    
    modules = [
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "seaborn",
        "tqdm"
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            failed.append(module)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    
    print("\n✅ All imports successful")
    return True


def main():
    """Main setup function"""
    print("="*60)
    print("BioBERT Health Risk Detection - Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    response = input("\nInstall Python requirements? (y/n): ")
    if response.lower() == 'y':
        if not install_requirements():
            print("\n⚠️  Setup incomplete - please install requirements manually")
            sys.exit(1)
    
    # Test imports
    if test_import():
        # Check CUDA
        check_cuda()
    
    # Create sample data
    download_sample_data()
    
    print("\n"+"="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print("\n📚 Next steps:")
    print("1. Review config.py to adjust parameters")
    print("2. Place your data in data/dailytalk.csv (or use sample data)")
    print("3. Run: python main.py --mode full")
    print("\nFor interactive inference: python main.py --mode inference")
    print("\n📖 See README.md for detailed documentation")


if __name__ == "__main__":
    main()
