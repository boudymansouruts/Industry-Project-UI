#!/usr/bin/env python3
"""
Script to restore or recreate the whisper-enhanced model
"""

import os
import shutil
from pathlib import Path

def check_whisper_enhanced():
    """Check if whisper-enhanced model exists"""
    enhanced_path = Path("whisper-enhanced")
    if enhanced_path.exists():
        print("✅ whisper-enhanced model found")
        return True
    else:
        print("❌ whisper-enhanced model not found")
        return False

def restore_from_backup():
    """Try to restore from backup locations"""
    backup_locations = [
        "backup/whisper-enhanced",
        "../whisper-enhanced", 
        "models/whisper-enhanced",
        "whisper-enhanced-backup"
    ]
    
    for backup_path in backup_locations:
        if Path(backup_path).exists():
            print(f"Found backup at: {backup_path}")
            shutil.copytree(backup_path, "whisper-enhanced")
            print("✅ whisper-enhanced model restored from backup")
            return True
    
    print("❌ No backup found")
    return False

def recreate_enhanced_model():
    """Recreate the whisper-enhanced model by fine-tuning"""
    print("🔄 Recreating whisper-enhanced model...")
    print("This will require:")
    print("1. DailyTalk dataset")
    print("2. Training script")
    print("3. Significant compute time")
    print()
    
    response = input("Do you want to proceed with recreation? (y/N): ").strip().lower()
    if response == 'y':
        print("⚠️  Recreation not implemented yet")
        print("You would need to:")
        print("1. Download DailyTalk dataset")
        print("2. Run training script")
        print("3. Save model to whisper-enhanced/")
        return False
    else:
        print("❌ Recreation cancelled")
        return False

def main():
    """Main function"""
    print("=== Whisper Enhanced Model Manager ===")
    print()
    
    if check_whisper_enhanced():
        print("Model is available and ready to use")
        return
    
    print("Model not found. Attempting to restore...")
    
    if restore_from_backup():
        return
    
    print("No backup found. Options:")
    print("1. Recreate the model (requires training)")
    print("2. Use pre-trained models instead")
    print()
    
    choice = input("Choose option (1/2): ").strip()
    
    if choice == "1":
        recreate_enhanced_model()
    elif choice == "2":
        print("✅ Using pre-trained models:")
        print("   - whisper-large: Best accuracy")
        print("   - whisper-base: Faster processing")
        print("Run: python model_config.py --list")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
