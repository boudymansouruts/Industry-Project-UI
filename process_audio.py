#!/usr/bin/env python3
"""
Simple CLI function to process the specific audio file
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def process_specific_file():
    """Process the specific audio file mentioned by the user"""
    
    # The specific file path
    audio_file = "uploads/20251002_133634_093407-i-837-61455560954.wav"
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file not found: {audio_file}")
        print("Available files in uploads:")
        uploads_dir = Path("uploads")
        if uploads_dir.exists():
            for file in uploads_dir.glob("*.wav"):
                print(f"   • {file}")
        return False
    
    print(f"🎤 Processing: {audio_file}")
    print("=" * 60)
    
    try:
        # Import and run the pipeline
        from transcription_chunk_risk_pipeline import TranscriptionChunkRiskPipeline
        
        # Initialize pipeline
        pipeline = TranscriptionChunkRiskPipeline()
        
        # Process the file
        result = pipeline.process_audio(audio_file)
        
        if result:
            print("\n✅ Processing completed successfully!")
            
            # Save results to JSON
            output_file = "results/audio_analysis_results.json"
            os.makedirs("results", exist_ok=True)
            
            import json
            with open(output_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            print(f"📊 Results saved to: {output_file}")
            
            # Print summary
            print("\n" + "="*60)
            print("RISK-FOCUSED ANALYSIS RESULTS")
            print("="*60)
            print(f"Audio file: {result.audio_file}")
            print(f"Duration: {result.audio_duration:.1f}s")
            print(f"Processing time: {result.processing_time:.1f}s")
            print(f"Total chunks: {result.total_chunks}")
            print(f"High risk chunks: {len(result.high_risk_chunks)}")
            print(f"Moderate risk chunks: {len(result.moderate_risk_chunks)}")
            print(f"Overall risk level: {result.risk_summary['overall_risk_level']}")
            
            # Print raw transcript and overall sentiment
            if getattr(result, 'raw_transcription', None):
                print("\nRAW TRANSCRIPT (no speakers):")
                print("-"*30)
                raw_text = result.raw_transcription
                display_text = raw_text if len(raw_text) < 4000 else raw_text[:4000] + "..."
                print(display_text)
                if getattr(result, 'overall_raw_sentiment', None):
                    ors = result.overall_raw_sentiment
                    print("\nOverall transcript sentiment:")
                    print(f"  {ors.get('predicted_emotion','unknown')} ({ors.get('confidence',0):.1%})")
            
            return True
        else:
            print("❌ Processing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = process_specific_file()
    sys.exit(0 if success else 1)
