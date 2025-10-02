#!/usr/bin/env python3
"""
Test Whisper functionality in SageMaker
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_whisper_imports():
    """Test all Whisper-related imports"""
    logger.info("🧪 Testing Whisper imports...")
    
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        logger.info("✅ WhisperProcessor imported successfully")
        logger.info("✅ WhisperForConditionalGeneration imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Whisper import failed: {e}")
        return False

def test_whisper_model_loading():
    """Test loading a Whisper model"""
    logger.info("🧪 Testing Whisper model loading...")
    
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        # Try to load a small model
        model_name = "openai/whisper-tiny"  # Smallest model for testing
        
        logger.info(f"🔄 Loading model: {model_name}")
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"✅ Processor type: {type(processor)}")
        logger.info(f"✅ Model type: {type(model)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_other_imports():
    """Test other required imports"""
    logger.info("🧪 Testing other imports...")
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("flask", "Flask"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("scikit_learn", "Scikit-learn"),
    ]
    
    all_success = True
    
    for module, name in imports_to_test:
        try:
            __import__(module)
            logger.info(f"✅ {name} imported successfully")
        except ImportError as e:
            logger.error(f"❌ {name} import failed: {e}")
            all_success = False
    
    return all_success

def main():
    """Main test function"""
    logger.info("=== SageMaker Whisper Functionality Test ===")
    
    # Test 1: Basic imports
    if not test_other_imports():
        logger.error("❌ Basic imports failed")
        return False
    
    # Test 2: Whisper imports
    if not test_whisper_imports():
        logger.error("❌ Whisper imports failed")
        return False
    
    # Test 3: Model loading
    if not test_whisper_model_loading():
        logger.error("❌ Whisper model loading failed")
        return False
    
    logger.info("🎉 All tests passed! Whisper is ready to use.")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("✅ Whisper functionality test completed successfully!")
        else:
            logger.error("❌ Whisper functionality test failed!")
            exit(1)
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        exit(1)
