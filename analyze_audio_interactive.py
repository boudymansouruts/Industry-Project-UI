#!/usr/bin/env python3
"""
Interactive Audio Analysis Script for Jupyter/SageMaker
No web server required - runs directly in Python/Jupyter
"""

import sys
from pathlib import Path
import pandas as pd

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from transcription_chunk_risk_pipeline import TranscriptionChunkRiskPipeline

def list_audio_files():
    """List available audio files"""
    uploads_dir = Path("uploads")
    audio_files = []
    
    if uploads_dir.exists():
        for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
            audio_files.extend(list(uploads_dir.glob(ext)))
    
    return sorted(audio_files)

def display_results(result):
    """Display analysis results in a formatted way"""
    
    print("\n" + "=" * 70)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"📁 File: {result.audio_file}")
    print(f"⏱️  Duration: {result.audio_duration:.1f}s")
    print(f"⚙️  Processing Time: {result.processing_time:.1f}s")
    print(f"📝 Total Chunks: {result.total_chunks}")
    print(f"🚨 High Risk Chunks: {len(result.high_risk_chunks)}")
    print(f"⚠️  Moderate Risk Chunks: {len(result.moderate_risk_chunks)}")
    print(f"✅ Low Risk Chunks: {len(result.low_risk_chunks)}")
    print(f"📊 Overall Risk Level: {result.risk_summary['overall_risk_level']}")
    
    # Display transcript
    if hasattr(result, 'raw_transcription') and result.raw_transcription:
        print("\n" + "=" * 70)
        print("📝 FULL TRANSCRIPT")
        print("=" * 70)
        print(result.raw_transcription)
    
    # Display risk analysis
    risk_chunks = result.high_risk_chunks + result.moderate_risk_chunks
    
    if risk_chunks:
        print("\n" + "=" * 70)
        print("⚠️  RISK ANALYSIS DETAILS")
        print("=" * 70 + "\n")
        
        risk_data = []
        for i, chunk in enumerate(risk_chunks, 1):
            print(f"\n[{i}] Risk Level: {chunk.get('risk_level', 'Unknown')}")
            print(f"    Speaker: {chunk.get('speaker', 'Unknown')}")
            print(f"    Emotion: {chunk.get('emotion', 'Unknown')}")
            print(f"    Confidence: {chunk.get('confidence', 0):.1%}")
            print(f"    Text: {chunk.get('text', 'N/A')}")
            
            risk_data.append({
                'Speaker': chunk.get('speaker', 'Unknown'),
                'Emotion': chunk.get('emotion', 'Unknown'),
                'Risk Level': chunk.get('risk_level', 'Unknown'),
                'Confidence': f"{chunk.get('confidence', 0):.1%}",
                'Text': chunk.get('text', '')[:60] + ('...' if len(chunk.get('text', '')) > 60 else '')
            })
        
        # Create DataFrame for better visualization
        print("\n" + "=" * 70)
        print("📊 RISK SUMMARY TABLE")
        print("=" * 70)
        df = pd.DataFrame(risk_data)
        print(df.to_string(index=False))
    else:
        print("\n✅ No significant risk detected")
    
    print("\n" + "=" * 70)

def analyze_file(file_path):
    """Analyze a specific audio file"""
    print(f"\n🎤 Processing: {file_path}")
    print("=" * 70)
    print("⏳ This may take a few minutes...")
    print("   - Loading models...")
    print("   - Transcribing audio...")
    print("   - Analyzing emotions...")
    print("   - Detecting risk levels...")
    print("=" * 70)
    
    try:
        # Initialize pipeline
        pipeline = TranscriptionChunkRiskPipeline()
        
        # Process the audio
        result = pipeline.process_audio(file_path)
        
        if result:
            print("\n✅ Processing completed successfully!")
            display_results(result)
            return result
        else:
            print("\n❌ Processing failed!")
            return None
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def interactive_mode():
    """Interactive file selection mode"""
    print("\n" + "=" * 70)
    print("🎤 AUDIO ANALYSIS & RISK DETECTION - INTERACTIVE MODE")
    print("=" * 70)
    
    # List available files
    audio_files = list_audio_files()
    
    if not audio_files:
        print("\n⚠️  No audio files found in uploads/ directory")
        print("💡 Please add audio files to the 'uploads/' folder first")
        return
    
    print(f"\n📁 Found {len(audio_files)} audio file(s):\n")
    for i, file in enumerate(audio_files, 1):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   {i}. {file.name} ({size_mb:.2f} MB)")
    
    print("\n" + "=" * 70)
    
    # Get user selection
    while True:
        try:
            selection = input(f"\nSelect file number (1-{len(audio_files)}) or 'q' to quit: ").strip()
            
            if selection.lower() == 'q':
                print("👋 Goodbye!")
                return
            
            file_index = int(selection) - 1
            
            if 0 <= file_index < len(audio_files):
                selected_file = audio_files[file_index]
                analyze_file(str(selected_file))
                
                # Ask if user wants to analyze another file
                another = input("\n\nAnalyze another file? (y/n): ").strip().lower()
                if another != 'y':
                    print("👋 Goodbye!")
                    return
            else:
                print(f"❌ Invalid selection. Please enter a number between 1 and {len(audio_files)}")
                
        except ValueError:
            print("❌ Invalid input. Please enter a number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Analysis & Risk Detection')
    parser.add_argument('--file', '-f', type=str, help='Path to audio file to analyze')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--list', '-l', action='store_true', help='List available files')
    
    args = parser.parse_args()
    
    if args.list:
        audio_files = list_audio_files()
        if audio_files:
            print(f"\n📁 Available audio files ({len(audio_files)}):\n")
            for i, file in enumerate(audio_files, 1):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   {i}. {file.name} ({size_mb:.2f} MB)")
        else:
            print("\n⚠️  No audio files found in uploads/ directory")
    elif args.file:
        analyze_file(args.file)
    elif args.interactive:
        interactive_mode()
    else:
        # Default: interactive mode
        interactive_mode()

if __name__ == '__main__':
    main()

