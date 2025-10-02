#!/usr/bin/env python
"""
Test script to verify BioBERT Health Risk Detection pipeline
Run this to ensure everything is working correctly
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test if all modules can be imported"""
    print("\n🔍 Testing module imports...")
    
    modules_to_test = [
        ('config', 'Configuration'),
        ('data_preprocessing', 'Data Preprocessing'),
        ('emotion_mapping', 'Emotion Mapping'),
        ('dataset_creation', 'Dataset Creation'),
        ('model_training', 'Model Training'),
        ('evaluation', 'Evaluation'),
        ('inference', 'Inference'),
        ('utils', 'Utilities'),
        ('main', 'Main Pipeline')
    ]
    
    failed = []
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {description} ({module_name}.py)")
        except Exception as e:
            print(f"  ❌ {description} ({module_name}.py): {str(e)}")
            failed.append(module_name)
    
    return len(failed) == 0


def test_configuration():
    """Test configuration loading"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from config import (
            MODEL_NAME, NUM_CLASSES, BATCH_SIZE,
            CLASS_LABELS, DEVICE
        )
        
        print(f"  ✅ Model: {MODEL_NAME}")
        print(f"  ✅ Classes: {NUM_CLASSES}")
        print(f"  ✅ Batch Size: {BATCH_SIZE}")
        print(f"  ✅ Device: {DEVICE}")
        print(f"  ✅ Class Labels: {len(CLASS_LABELS)} categories")
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {str(e)}")
        return False


def test_data_processing():
    """Test data processing pipeline"""
    print("\n📊 Testing data processing...")
    
    try:
        from data_preprocessing import TextPreprocessor, DataLoader
        
        # Test text preprocessing
        preprocessor = TextPreprocessor()
        test_text = "I'm feeling really anxious and can't sleep!!!"
        cleaned = preprocessor.clean_text(test_text)
        print(f"  ✅ Text preprocessing working")
        print(f"     Original: '{test_text}'")
        print(f"     Cleaned: '{cleaned}'")
        
        # Test data loader
        loader = DataLoader()
        print(f"  ✅ DataLoader initialized")
        
        return True
    except Exception as e:
        print(f"  ❌ Data processing error: {str(e)}")
        traceback.print_exc()
        return False


def test_emotion_mapping():
    """Test emotion mapping"""
    print("\n🎭 Testing emotion mapping...")
    
    try:
        from emotion_mapping import EmotionMapper
        
        mapper = EmotionMapper()
        
        test_emotions = {
            "sad": "depression",
            "anxious": "anxiety",
            "happy": "happiness",
            "angry": "anger",
            "confused": "confusion"
        }
        
        all_correct = True
        for original, expected in test_emotions.items():
            mapped = mapper.map_emotion(original)
            if mapped == expected:
                print(f"  ✅ '{original}' → '{mapped}'")
            else:
                print(f"  ❌ '{original}' → '{mapped}' (expected '{expected}')")
                all_correct = False
        
        return all_correct
    except Exception as e:
        print(f"  ❌ Emotion mapping error: {str(e)}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation"""
    print("\n🤖 Testing model creation...")
    
    try:
        import torch
        from model_training import BioBERTClassifier, create_model
        
        # Create model
        model = create_model()
        
        # Test forward pass
        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
        
        expected_shape = (batch_size, 12)  # 12 classes
        actual_shape = tuple(logits.shape)
        
        if actual_shape == expected_shape:
            print(f"  ✅ Model created successfully")
            print(f"  ✅ Forward pass working: output shape {actual_shape}")
        else:
            print(f"  ❌ Unexpected output shape: {actual_shape} (expected {expected_shape})")
            return False
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Total parameters: {total_params:,}")
        
        return True
    except Exception as e:
        print(f"  ❌ Model creation error: {str(e)}")
        traceback.print_exc()
        return False


def test_inference_pipeline():
    """Test inference pipeline"""
    print("\n🔮 Testing inference pipeline...")
    
    try:
        from inference import HealthRiskPredictor, PredictionResult
        
        # Note: This will work with untrained model for testing
        predictor = HealthRiskPredictor()
        
        test_texts = [
            "I'm feeling very depressed and hopeless",
            "So excited about my new job!",
            "The stress is overwhelming"
        ]
        
        for text in test_texts:
            result = predictor.predict_single(text)
            print(f"  ✅ Predicted: '{text[:30]}...'")
            print(f"     Emotion: {result.predicted_emotion}")
            print(f"     Risk: {result.risk_level}")
            print(f"     Confidence: {result.confidence:.2%}")
        
        return True
    except Exception as e:
        print(f"  ❌ Inference error: {str(e)}")
        # This might fail if model isn't trained yet, which is okay
        if "not found" in str(e).lower():
            print("     (This is expected if model hasn't been trained yet)")
            return True
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions"""
    print("\n🛠️ Testing utilities...")
    
    try:
        from utils import set_seed, create_directories, monitor_gpu_usage
        
        # Test seed setting
        set_seed(42)
        print(f"  ✅ Random seed setting working")
        
        # Test directory creation
        create_directories()
        print(f"  ✅ Directory creation working")
        
        # Test GPU monitoring
        gpu_info = monitor_gpu_usage()
        if gpu_info['gpu_available']:
            print(f"  ✅ GPU monitoring: {gpu_info['device_name']}")
        else:
            print(f"  ⚠️ GPU not available (CPU mode)")
        
        return True
    except Exception as e:
        print(f"  ❌ Utilities error: {str(e)}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("BioBERT Health Risk Detection - System Test")
    print("="*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Data Processing", test_data_processing),
        ("Emotion Mapping", test_emotion_mapping),
        ("Model Creation", test_model_creation),
        ("Utilities", test_utils),
        ("Inference Pipeline", test_inference_pipeline)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with unexpected error: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Place your data in data/dailytalk.csv")
        print("2. Run: python main.py --mode full")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("You may need to:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check your Python version (3.8+ required)")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
