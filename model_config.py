#!/usr/bin/env python3
"""
Model configuration utility for Risk Audio Analysis Pipeline
Allows easy switching between different Whisper models
"""

import os
import json
from pathlib import Path

class ModelConfig:
    """Manage model configurations"""
    
    MODELS = {
        "whisper-large": {
            "name": "Whisper Large (Pre-trained)",
            "path": "openai/whisper-large-v2",
            "description": "OpenAI's Whisper Large model - best accuracy, slower processing",
            "recommended_for": "Production, high accuracy requirements"
        },
        "whisper-base": {
            "name": "Whisper Base (Pre-trained)", 
            "path": "openai/whisper-base",
            "description": "OpenAI's Whisper Base model - good accuracy, faster processing",
            "recommended_for": "Development, faster processing"
        },
        "whisper-enhanced": {
            "name": "Whisper Enhanced (Fine-tuned)",
            "path": "whisper-enhanced",
            "description": "Fine-tuned Whisper model on DailyTalk dataset",
            "recommended_for": "Domain-specific audio, if available"
        }
    }
    
    @classmethod
    def list_models(cls):
        """List available models"""
        print("Available Models:")
        print("=" * 50)
        for key, model in cls.MODELS.items():
            status = "✅ Available" if cls.is_model_available(key) else "❌ Not Available"
            print(f"{key}:")
            print(f"  Name: {model['name']}")
            print(f"  Path: {model['path']}")
            print(f"  Description: {model['description']}")
            print(f"  Recommended for: {model['recommended_for']}")
            print(f"  Status: {status}")
            print()
    
    @classmethod
    def is_model_available(cls, model_key):
        """Check if model is available"""
        model_path = cls.MODELS[model_key]["path"]
        
        if model_path.startswith("openai/"):
            # Pre-trained models are always available
            return True
        else:
            # Check if local model exists
            return os.path.exists(model_path)
    
    @classmethod
    def get_current_model(cls):
        """Get current model configuration"""
        config_file = Path("model_config.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get("current_model", "whisper-large")
        return "whisper-large"  # Default
    
    @classmethod
    def set_model(cls, model_key):
        """Set the current model"""
        if model_key not in cls.MODELS:
            print(f"❌ Error: Model '{model_key}' not found")
            return False
        
        if not cls.is_model_available(model_key):
            print(f"❌ Error: Model '{model_key}' is not available")
            return False
        
        config = {
            "current_model": model_key,
            "model_info": cls.MODELS[model_key]
        }
        
        with open("model_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Model set to: {cls.MODELS[model_key]['name']}")
        print(f"   Path: {cls.MODELS[model_key]['path']}")
        return True
    
    @classmethod
    def get_model_path(cls, model_key=None):
        """Get the path for a model"""
        if model_key is None:
            model_key = cls.get_current_model()
        
        return cls.MODELS[model_key]["path"]

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Configuration Utility")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--set", type=str, help="Set current model")
    parser.add_argument("--current", action="store_true", help="Show current model")
    parser.add_argument("--path", type=str, help="Get path for specific model")
    
    args = parser.parse_args()
    
    if args.list:
        ModelConfig.list_models()
    elif args.set:
        ModelConfig.set_model(args.set)
    elif args.current:
        current = ModelConfig.get_current_model()
        model_info = ModelConfig.MODELS[current]
        print(f"Current model: {model_info['name']}")
        print(f"Path: {model_info['path']}")
        print(f"Status: {'✅ Available' if ModelConfig.is_model_available(current) else '❌ Not Available'}")
    elif args.path:
        path = ModelConfig.get_model_path(args.path)
        print(path)
    else:
        # Interactive mode
        print("=== Model Configuration Utility ===")
        print()
        
        current = ModelConfig.get_current_model()
        print(f"Current model: {ModelConfig.MODELS[current]['name']}")
        print()
        
        ModelConfig.list_models()
        
        print("Commands:")
        print("  python model_config.py --list     # List all models")
        print("  python model_config.py --set whisper-large  # Set model")
        print("  python model_config.py --current   # Show current model")
        print("  python model_config.py --path whisper-base  # Get model path")

if __name__ == "__main__":
    main()
