#!/usr/bin/env python3
"""
SageMaker-compatible startup script for Risk Audio Analysis Pipeline
Handles import issues and environment setup
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_install_dependencies():
    """Check and install missing dependencies"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'torch',
        'transformers', 
        'librosa',
        'soundfile',
        'flask',
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} is missing")
    
    if missing_packages:
        logger.info(f"Installing missing packages: {missing_packages}")
        import subprocess
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
                return False
    
    return True

def test_whisper_imports():
    """Test Whisper-specific imports"""
    logger.info("Testing Whisper imports...")
    
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        logger.info("✅ WhisperProcessor and WhisperForConditionalGeneration imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Whisper import failed: {e}")
        logger.info("Trying to install transformers with specific version...")
        
        try:
            import subprocess
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                'transformers>=4.30.0', 'torch>=2.0.0', 'torchaudio>=2.0.0'
            ])
            
            # Try import again
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            logger.info("✅ Whisper imports successful after installation")
            return True
            
        except Exception as e2:
            logger.error(f"❌ Still failed after installation: {e2}")
            return False

def setup_environment():
    """Setup environment variables and paths"""
    logger.info("Setting up environment...")
    
    # Set environment variables
    os.environ['PYTHONPATH'] = os.getcwd()
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
    os.environ['HF_HOME'] = '/tmp/huggingface'
    
    # Create cache directories
    Path('/tmp/transformers_cache').mkdir(exist_ok=True)
    Path('/tmp/huggingface').mkdir(exist_ok=True)
    
    logger.info("✅ Environment setup complete")

def start_server():
    """Start the Flask server"""
    logger.info("Starting Flask server...")
    
    try:
        # Import and start the server
        from risk_web_app import app
        
        # Get port from environment (SageMaker uses 8080)
        port = int(os.environ.get('PORT', 8080))
        host = os.environ.get('HOST', '0.0.0.0')
        
        logger.info(f"Starting server on {host}:{port}")
        
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        logger.error("Trying alternative startup method...")
        
        # Alternative: direct import
        try:
            import risk_web_app
            risk_web_app.app.run(host='0.0.0.0', port=8080, debug=False)
        except Exception as e2:
            logger.error(f"❌ Alternative startup also failed: {e2}")
            sys.exit(1)

def main():
    """Main function"""
    logger.info("=== SageMaker Risk Audio Analysis Pipeline Startup ===")
    
    # Check dependencies
    if not check_and_install_dependencies():
        logger.error("❌ Dependency check failed")
        sys.exit(1)
    
    # Test Whisper imports
    if not test_whisper_imports():
        logger.error("❌ Whisper import test failed")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Start server
    logger.info("🚀 All checks passed, starting server...")
    start_server()

if __name__ == "__main__":
    main()
