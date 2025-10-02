#!/usr/bin/env python3
"""
CLI Audio Processing Script
Processes audio files using the transcription and risk analysis pipeline
"""

import os
import sys
import argparse
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import the main pipeline
from transcription_chunk_risk_pipeline import process_audio_file

def setup_logging():
    """Setup logging for CLI"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def process_audio_cli(audio_path, output_dir=None):
    """
    Process audio file using CLI
    
    Args:
        audio_path (str): Path to audio file
        output_dir (str, optional): Output directory for results
    """
    
    # Validate input file
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found: {audio_path}")
        return False
    
    # Set default output directory
    if output_dir is None:
        output_dir = Path(audio_path).parent / "results"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎤 Processing audio file: {audio_path}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    try:
        # Process the audio file
        result = process_audio_file(
            audio_path=audio_path,
            output_dir=str(output_dir),
            save_results=True
        )
        
        if result:
            print("\n✅ Processing completed successfully!")
            print(f"📊 Results saved to: {output_dir}")
            
            # Print summary
            if hasattr(result, 'summary'):
                print("\n📋 Summary:")
                print(f"   • Total duration: {result.summary.get('duration', 'N/A')}")
                print(f"   • Speakers detected: {result.summary.get('speakers', 'N/A')}")
                print(f"   • Risk level: {result.summary.get('risk_level', 'N/A')}")
                print(f"   • Chunks processed: {result.summary.get('chunks', 'N/A')}")
            
            return True
        else:
            print("❌ Processing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        logging.exception("Processing error details:")
        return False

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Process audio files for transcription and risk analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_audio_processor.py uploads/audio.wav
  python cli_audio_processor.py uploads/audio.wav --output results/
  python cli_audio_processor.py uploads/audio.wav --output results/ --verbose
        """
    )
    
    parser.add_argument(
        'audio_file',
        help='Path to the audio file to process'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory for results (default: same as input file)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        setup_logging()
    
    # Process the audio file
    success = process_audio_cli(args.audio_file, args.output)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
