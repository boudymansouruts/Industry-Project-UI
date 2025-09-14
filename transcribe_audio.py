#!/usr/bin/env python3
"""
Post-processor to add speaker identification to Whisper transcriptions
"""

import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from preprocess import load_audio, split_audio_into_chunks, analyze_audio_features, identify_speakers_by_features, create_speaker_transcription

def identify_speakers_from_audio(audio_file_path, model_dir="whisper-generic-speakers"):
    """Identify speakers and transcribe with speaker labels"""
    
    # Check if audio file exists
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        return None
    
    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        return None
    
    try:
        # Load the fine-tuned model
        processor = WhisperProcessor.from_pretrained(model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Load and preprocess audio
        audio, sr = load_audio(audio_file_path)
        
        # Split audio into smaller chunks for speaker analysis
        chunks = split_audio_into_chunks(audio, chunk_length_seconds=6)  # Increased chunk size for better context
        
        # Process each chunk
        chunk_transcriptions = []
        chunk_features = []
        
        for i, chunk in enumerate(chunks):
            # Transcribe chunk with better parameters
            input_features = processor.feature_extractor(
                chunk, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            
            with torch.no_grad():
                # Try to detect language automatically, fallback to English
                predicted_ids = model.generate(
                    input_features,
                    max_length=200,  # Increased max length
                    num_beams=2,     # Use beam search for better accuracy
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            transcription = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
            
            # Clean up transcription
            transcription = clean_transcription(transcription)
            
            if transcription and len(transcription) > 3:
                chunk_transcriptions.append(transcription)
                chunk_features.append(analyze_audio_features(chunk))
            else:
                chunk_transcriptions.append("")
                chunk_features.append(None)
        
        # Identify speakers based on audio features
        speaker_assignments = identify_speakers_by_features(chunk_features)
        
        # Create final transcription with speaker labels
        final_transcription = create_speaker_transcription(chunk_transcriptions, speaker_assignments)
        
        return final_transcription
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return None

def clean_transcription(text):
    """Clean up transcription text"""
    # Remove repeated words/phrases
    words = text.split()
    cleaned_words = []
    
    for i, word in enumerate(words):
        # Skip if it's a repetition of the previous word
        if i > 0 and word.lower() == words[i-1].lower():
            continue
        # Skip very short repeated words
        if i > 1 and word.lower() == words[i-1].lower() == words[i-2].lower():
            continue
        cleaned_words.append(word)
    
    return " ".join(cleaned_words)

def main():
    """Main function"""
    audio_file = r"C:\Users\boudy\Downloads\091452-i-834-836.wav"
    result = identify_speakers_from_audio(audio_file)
    
    if result:
        print("📝 Transcription with Speaker Labels:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
        with open("speaker_transcription.txt", "w", encoding="utf-8") as f:
            f.write(result)
        print(f"✅ Result saved to speaker_transcription.txt")

if __name__ == '__main__':
    main()