#!/usr/bin/env python3
"""
Main script for Speaker Identification with Whisper
"""

import os
import sys
import argparse
from transcribe_audio import identify_speakers_from_audio

def main():
    """Main function for speaker identification"""
    
    parser = argparse.ArgumentParser(description="Speaker Identification with Whisper")
    parser.add_argument("audio_file", help="Path to audio file to transcribe")
    parser.add_argument("--model", default="whisper-generic-speakers", help="Path to model directory")
    parser.add_argument("--output", help="Output file path (optional)")
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        return 1
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model directory not found: {args.model}")
        print("Please run 'python train_model.py' first to train the model")
        return 1
    
    print("🎤 Speaker Identification with Whisper")
    print("="*50)
    print(f"📁 Audio file: {args.audio_file}")
    print(f"🤖 Model: {args.model}")
    
    # Transcribe audio with speaker identification
    result = identify_speakers_from_audio(args.audio_file, args.model)
    
    if result:
        # Save to file if output path specified
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"💾 Result saved to: {args.output}")
        else:
            # Save to default file
            with open("speaker_transcription.txt", "w", encoding="utf-8") as f:
                f.write(result)
            print(f"💾 Result saved to: speaker_transcription.txt")
        
        return 0
    else:
        print("❌ Transcription failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())

