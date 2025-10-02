#!/usr/bin/env python3
"""
SageMaker Installation Script
Handles dependency conflicts with AutoGluon and MLflow
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and log the result"""
    logger.info(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} - Failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def install_sagemaker_compatible():
    """Install packages compatible with SageMaker's pre-installed packages"""
    
    logger.info("=== SageMaker Compatible Installation ===")
    
    # Step 1: Install core packages individually to avoid conflicts
    packages = [
        ("torch>=2.2.0,<2.8.0", "PyTorch (compatible with AutoGluon)"),
        ("transformers>=4.30.0,<4.50.0", "Transformers (compatible with AutoGluon)"),
        ("torchaudio>=2.2.0", "TorchAudio"),
        ("librosa>=0.10.0", "Librosa for audio processing"),
        ("soundfile>=0.12.0", "SoundFile for audio I/O"),
        ("flask>=2.3.0", "Flask web framework"),
        ("accelerate>=0.20.0", "Accelerate for Whisper"),
        ("sentencepiece>=0.1.99", "SentencePiece for tokenization"),
        ("huggingface-hub>=0.15.0", "Hugging Face Hub"),
    ]
    
    for package, description in packages:
        if not run_command(f"pip install '{package}'", f"Installing {description}"):
            logger.warning(f"Failed to install {package}, continuing...")
    
    # Step 2: Install datasets with specific version to avoid conflicts
    if not run_command("pip install 'datasets>=2.12.0,<4.2.0'", "Installing datasets (compatible version)"):
        logger.warning("Failed to install datasets, trying alternative...")
        run_command("pip install datasets", "Installing datasets (latest)")
    
    # Step 3: Test critical imports
    logger.info("🧪 Testing critical imports...")
    
    test_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("librosa", "Librosa"),
        ("flask", "Flask"),
    ]
    
    all_imports_ok = True
    for module, name in test_imports:
        try:
            __import__(module)
            logger.info(f"✅ {name} import successful")
        except ImportError as e:
            logger.error(f"❌ {name} import failed: {e}")
            all_imports_ok = False
    
    # Step 4: Test Whisper specifically
    logger.info("🎤 Testing Whisper imports...")
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        logger.info("✅ Whisper imports successful!")
    except ImportError as e:
        logger.error(f"❌ Whisper imports failed: {e}")
        all_imports_ok = False
    
    if all_imports_ok:
        logger.info("🎉 All installations successful!")
        logger.info("🚀 Ready to start the server with: python start_sagemaker_server.py")
    else:
        logger.error("❌ Some installations failed. Check the errors above.")
        return False
    
    return True

def main():
    """Main installation function"""
    try:
        success = install_sagemaker_compatible()
        if success:
            logger.info("✅ Installation completed successfully!")
        else:
            logger.error("❌ Installation had issues. Please check the logs.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Installation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
