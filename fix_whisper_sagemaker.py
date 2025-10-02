#!/usr/bin/env python3
"""
SageMaker Whisper Installation Fix
Handles WhisperProcessor import issues
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
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} - Failed: {e}")
        if e.stderr:
            logger.error(f"Error: {e.stderr.strip()}")
        return False

def check_transformers_version():
    """Check the current transformers version"""
    try:
        import transformers
        version = transformers.__version__
        logger.info(f"📦 Current transformers version: {version}")
        return version
    except ImportError:
        logger.error("❌ Transformers not installed")
        return None

def install_whisper_compatible_transformers():
    """Install a transformers version that supports Whisper"""
    
    logger.info("=== Whisper-Compatible Transformers Installation ===")
    
    # Step 1: Check current version
    current_version = check_transformers_version()
    
    # Step 2: Uninstall current transformers if needed
    if current_version:
        logger.info("🔄 Uninstalling current transformers...")
        run_command("pip uninstall transformers -y", "Uninstall current transformers")
    
    # Step 3: Install specific transformers version that supports Whisper
    whisper_versions = [
        "transformers>=4.30.0,<4.35.0",  # Known to support Whisper
        "transformers>=4.25.0,<4.30.0",  # Alternative version
        "transformers>=4.20.0,<4.25.0",  # Fallback version
    ]
    
    for version in whisper_versions:
        logger.info(f"🔄 Trying transformers version: {version}")
        if run_command(f"pip install '{version}'", f"Install {version}"):
            # Test Whisper import
            if test_whisper_import():
                logger.info(f"✅ Successfully installed {version} with Whisper support!")
                return True
            else:
                logger.warning(f"❌ {version} installed but Whisper not working")
                continue
    
    # Step 4: Try installing from source if all else fails
    logger.info("🔄 Trying to install transformers from source...")
    if run_command("pip install git+https://github.com/huggingface/transformers.git", "Install from source"):
        if test_whisper_import():
            logger.info("✅ Successfully installed transformers from source!")
            return True
    
    return False

def test_whisper_import():
    """Test if Whisper imports work"""
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        logger.info("✅ WhisperProcessor and WhisperForConditionalGeneration imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Whisper import failed: {e}")
        return False

def install_other_dependencies():
    """Install other required dependencies"""
    logger.info("🔄 Installing other dependencies...")
    
    # Try to install from unified requirements.txt first
    if run_command("pip install -r requirements.txt", "Install from requirements.txt"):
        logger.info("✅ Requirements installed from requirements.txt")
    else:
        logger.warning("Failed to install from requirements.txt, installing individually...")
        
        packages = [
            "torch>=2.2.0,<2.8.0",
            "torchaudio>=2.2.0", 
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "flask>=2.3.0",
            "accelerate>=0.20.0",
            "sentencepiece>=0.1.99",
            "huggingface-hub>=0.15.0",
        ]
        
        for package in packages:
            run_command(f"pip install '{package}'", f"Install {package}")

def main():
    """Main installation function"""
    logger.info("=== SageMaker Whisper Installation Fix ===")
    
    # Step 1: Install other dependencies first
    install_other_dependencies()
    
    # Step 2: Install Whisper-compatible transformers
    if install_whisper_compatible_transformers():
        logger.info("🎉 Whisper installation successful!")
        
        # Step 3: Final test
        logger.info("🧪 Running final tests...")
        if test_whisper_import():
            logger.info("✅ All tests passed!")
            logger.info("🚀 Ready to start server: python start_sagemaker_server.py")
        else:
            logger.error("❌ Final test failed")
            return False
    else:
        logger.error("❌ Failed to install Whisper-compatible transformers")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            logger.error("❌ Installation failed. Please check the logs.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Installation failed with error: {e}")
        sys.exit(1)
