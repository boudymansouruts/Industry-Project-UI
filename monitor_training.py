#!/usr/bin/env python3
"""
Training Monitor Script
Monitors the emotion recognition model training progress
"""

import os
import time
import glob
from datetime import datetime

def monitor_training():
    """Monitor training progress by checking logs and model files"""
    
    print("🔍 Monitoring Emotion Recognition Training...")
    print("=" * 60)
    
    # Check for log files
    log_files = glob.glob("logs/*.log")
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        print(f"📋 Latest log: {latest_log}")
        
        # Read last few lines
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                print(f"📊 Last 5 log entries:")
                for line in lines[-5:]:
                    print(f"   {line.strip()}")
        except Exception as e:
            print(f"❌ Error reading log: {e}")
    else:
        print("📋 No log files found yet")
    
    # Check for model checkpoints
    model_files = glob.glob("models/*.pth")
    if model_files:
        print(f"\n💾 Model checkpoints found: {len(model_files)}")
        for model_file in sorted(model_files):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))
            print(f"   {model_file} ({size_mb:.1f} MB) - {mod_time.strftime('%H:%M:%S')}")
    else:
        print("\n💾 No model checkpoints found yet")
    
    # Check for results
    result_files = glob.glob("results/*.png")
    if result_files:
        print(f"\n📈 Results generated: {len(result_files)}")
        for result_file in sorted(result_files):
            mod_time = datetime.fromtimestamp(os.path.getmtime(result_file))
            print(f"   {result_file} - {mod_time.strftime('%H:%M:%S')}")
    
    print("\n" + "=" * 60)
    print("⏰ Training is running in background...")
    print("💡 Run this script again to check progress")

if __name__ == "__main__":
    monitor_training()