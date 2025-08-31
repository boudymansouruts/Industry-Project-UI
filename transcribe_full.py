#!/usr/bin/env python3
"""
Get full transcription of audio file using fine-tuned Whisper model
"""

import os
import torch
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def transcribe_full_audio(audio_path: str, model_dir: str = "whisper-dailytalk-3k"):
    """
    Transcribe full audio file and return complete text
    """
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return None
    
    if not os.path.exists(model_dir):
        print(f"⚠️ Fine-tuned model not found: {model_dir}")
        print("🔄 Using original Whisper model instead...")
        model_dir = "openai/whisper-base"
    
    print(f"🤖 Loading model from: {model_dir}")
    
    try:
        # Load model and processor
        processor = WhisperProcessor.from_pretrained(model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        model.eval()
        
        print(f"🎤 Loading audio: {audio_path}")
        
        # Load and preprocess audio
        audio, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        # Resample to 16kHz if needed
        if sr != 16000:
            print(f"🔄 Resampling from {sr}Hz to 16000Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        print(f"📊 Audio duration: {len(audio)/16000:.1f} seconds")
        print("🔄 Transcribing...")
        
        # Process audio with Whisper
        input_features = processor.feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=processor.get_decoder_prompt_ids(
                    language="en", 
                    task="transcribe"
                ),
                max_length=500,  # Allow longer transcriptions
                num_beams=5,     # Use beam search for better quality
                do_sample=False  # Deterministic output
            )
        
        # Decode to text
        transcription = processor.tokenizer.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        # Clean up transcription
        transcription = transcription.strip()
        
        return transcription
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return None

def main():
    """Main function"""
    audio_file = "091452-i-834-836.wav"
    
    print("🎵 Full Audio Transcription")
    print("="*50)
    
    # Get transcription
    transcription = transcribe_full_audio(audio_file)
    
    if transcription:
        print("\n" + "="*60)
        print("📝 FULL TRANSCRIPTION:")
        print("="*60)
        print(transcription)
        print("="*60)
        
        # Save transcription to file
        output_file = f"full_transcription_{Path(audio_file).stem}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Audio file: {audio_file}\n")
            f.write(f"Full transcription:\n\n{transcription}\n")
        
        print(f"💾 Transcription saved to: {output_file}")
        
        # Also return just the text for easy copying
        print(f"\n📋 TEXT ONLY (for copying):")
        print("-" * 40)
        print(transcription)
        
    else:
        print("❌ Transcription failed!")

if __name__ == '__main__':
    from pathlib import Path
    main()
