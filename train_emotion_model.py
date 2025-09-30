#!/usr/bin/env python3
"""
Train Emotion Recognition Model with Detailed Debugging
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("EMOTION RECOGNITION MODEL TRAINING")
print("=" * 80)
print()

# Debug: Check imports
print("Step 1: Importing modules...")

# Change to Emotion_Recognition directory for imports
emotion_dir = Path("Emotion_Recognition")
os.chdir(emotion_dir)

try:
    from config import *
    print("✅ Config imported")
    print(f"   - Device: {DEVICE}")
    print(f"   - Num classes: {NUM_CLASSES}")
    print(f"   - Model: {MODEL_NAME}")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
except Exception as e:
    print(f"❌ Error importing config: {e}")
    sys.exit(1)

try:
    from main import BioBERTHealthRiskPipeline
    print("✅ Main pipeline imported")
except Exception as e:
    print(f"❌ Error importing main: {e}")
    sys.exit(1)

print()

# Debug: Check data
print("Step 2: Checking data...")
data_path = Path("data/dailytalk.csv")
if data_path.exists():
    print(f"✅ Data file found: {data_path}")
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"   - Total samples: {len(df)}")
    print(f"   - Columns: {list(df.columns)}")
    if 'emotion' in df.columns:
        print(f"   - Emotion distribution:")
        emotion_counts = df['emotion'].value_counts()
        for emotion, count in emotion_counts.head(10).items():
            print(f"     • {emotion}: {count}")
else:
    print(f"❌ Data file not found: {data_path}")
    sys.exit(1)

print()

# Debug: Check directories
print("Step 3: Checking directories...")
dirs_to_create = ['models', 'models/best_model', 'models/checkpoints', 'results', 'logs', 'data']
for dir_name in dirs_to_create:
    dir_path = Path(dir_name)
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directory ready: {dir_name}")

print()

# Start training
print("=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print()

try:
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = BioBERTHealthRiskPipeline()
    print("✅ Pipeline initialized")
    print()
    
    # Prepare data
    print("Preparing data (splitting train/val/test)...")
    pipeline.prepare_data(data_path)
    print("✅ Data prepared")
    print(f"   - Train samples: {len(pipeline.train_df)}")
    print(f"   - Val samples: {len(pipeline.val_df)}")
    print(f"   - Test samples: {len(pipeline.test_df)}")
    print()
    
    # Create datasets
    print("Creating PyTorch datasets...")
    pipeline.create_datasets()
    print("✅ Datasets created")
    print(f"   - Train batches: {len(pipeline.train_loader)}")
    print(f"   - Val batches: {len(pipeline.val_loader)}")
    print(f"   - Test batches: {len(pipeline.test_loader)}")
    print()
    
    # Train model
    print("=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    print()
    print("This will take 30-60 minutes...")
    print("Progress will be shown below:")
    print()
    
    pipeline.train_model()
    print("\n✅ Training completed!")
    print()
    
    # Evaluate model
    print("=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)
    print()
    
    pipeline.evaluate_model()
    print("✅ Evaluation completed!")
    print()
    
    # Summary
    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print()
    print("✅ Model trained and saved successfully!")
    print(f"📁 Model location: Emotion_Recognition/models/best_model/best_model.pt")
    print(f"📊 Results location: Emotion_Recognition/results/")
    print()
    print("🚀 Next steps:")
    print("   1. Restart the risk-focused server:")
    print("      python start_risk_server.py")
    print("   2. The server will now use the trained model")
    print("   3. Upload audio files for accurate emotion detection")
    print()
    
except KeyboardInterrupt:
    print("\n\n⚠️ Training interrupted by user")
    print("You can resume training later using the checkpoints in models/checkpoints/")
    
except Exception as e:
    print(f"\n\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
print("DONE!")
print("=" * 80)
