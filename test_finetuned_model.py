#!/usr/bin/env python3
"""
Test the fine-tuned Whisper model
"""

import os
import torch
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path

def test_finetuned_model(model_dir: str = "whisper-dailytalk-3k"):
    """Test the fine-tuned Whisper model"""
    
    if not os.path.exists(model_dir):
        print(f"❌ Fine-tuned model not found: {model_dir}")
        print("   Make sure fine-tuning has completed first!")
        return
    
    print(f"🤖 Loading fine-tuned model from: {model_dir}")
    
    try:
        # Load fine-tuned model and processor
        processor = WhisperProcessor.from_pretrained(model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        model.eval()
        
        print("✅ Fine-tuned model loaded successfully!")
        
        # Test on available audio files
        test_files = ["091419-o-832-853.wav", "091452-i-834-836.wav"]
        
        for audio_file in test_files:
            if os.path.exists(audio_file):
                print(f"\n🎤 Testing on: {audio_file}")
                print("-" * 40)
                
                # Load and preprocess audio
                audio, sr = sf.read(audio_file)
                if len(audio.shape) > 1:
                    audio = audio[:, 0]  # Convert to mono
                
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                
                # Normalize
                audio = librosa.util.normalize(audio)
                
                print(f"📊 Audio duration: {len(audio)/16000:.1f}s")
                
                # Transcribe with fine-tuned model
                input_features = processor.feature_extractor(
                    audio, sampling_rate=16000, return_tensors="pt"
                ).input_features
                
                with torch.no_grad():
                    predicted_ids = model.generate(
                        input_features,
                        forced_decoder_ids=processor.get_decoder_prompt_ids(
                            language="en", task="transcribe"
                        ),
                        max_length=200
                    )
                
                transcription = processor.tokenizer.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]
                
                print(f"📝 Fine-tuned Transcription:")
                print(f"   {transcription}")
                
                # Save transcription
                output_file = f"finetuned_transcript_{Path(audio_file).stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Audio file: {audio_file}\n")
                    f.write(f"Model: {model_dir}\n")
                    f.write(f"Transcription: {transcription}\n")
                
                print(f"💾 Saved to: {output_file}")
        
        print(f"\n🎉 Testing completed!")
        print(f"📁 Fine-tuned model ready for use: {model_dir}")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()

def compare_models():
    """Compare original vs fine-tuned model performance"""
    print("🆚 Comparing Original vs Fine-tuned Whisper")
    print("="*50)
    
    model_dir = "whisper-dailytalk-3k"
    test_file = "091452-i-834-836.wav"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
    
    # Load audio
    audio, sr = sf.read(test_file)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = librosa.util.normalize(audio)
    
    print(f"🎤 Testing on: {test_file}")
    
    # Test original model
    print("\n📍 Original Whisper Model:")
    try:
        processor_orig = WhisperProcessor.from_pretrained("openai/whisper-base")
        model_orig = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        
        input_features = processor_orig.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features
        
        with torch.no_grad():
            predicted_ids = model_orig.generate(
                input_features,
                forced_decoder_ids=processor_orig.get_decoder_prompt_ids(
                    language="en", task="transcribe"
                ),
                max_length=200
            )
        
        transcription_orig = processor_orig.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        
        print(f"   {transcription_orig}")
        
    except Exception as e:
        print(f"❌ Error with original model: {e}")
    
    # Test fine-tuned model
    print("\n🎯 Fine-tuned Whisper Model:")
    if os.path.exists(model_dir):
        try:
            processor_ft = WhisperProcessor.from_pretrained(model_dir)
            model_ft = WhisperForConditionalGeneration.from_pretrained(model_dir)
            
            input_features = processor_ft.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features
            
            with torch.no_grad():
                predicted_ids = model_ft.generate(
                    input_features,
                    forced_decoder_ids=processor_ft.get_decoder_prompt_ids(
                        language="en", task="transcribe"
                    ),
                    max_length=200
                )
            
            transcription_ft = processor_ft.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            print(f"   {transcription_ft}")
            
        except Exception as e:
            print(f"❌ Error with fine-tuned model: {e}")
    else:
        print("   Model not ready yet - still training...")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_models()
    else:
        test_finetuned_model()

if __name__ == '__main__':
    main()
