#!/usr/bin/env python3
"""
Transcribe audio file with speaker separation using fine-tuned Whisper
"""

import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def extract_speaker_features(audio: np.ndarray, sr: int, start_time: float, end_time: float):
    """Extract speaker-specific features from audio segment"""
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = audio[start_sample:end_sample]
    
    if len(segment) < 1024:
        return None
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # Pitch features
    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)
    
    # Energy features
    rms = librosa.feature.rms(y=segment)
    
    # Combine features
    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [pitch_mean],
        [np.mean(spectral_centroids)],
        [np.mean(spectral_rolloff)],
        [np.mean(zero_crossing_rate)],
        [np.mean(rms)]
    ])
    
    return features

def perform_speaker_diarization(audio: np.ndarray, sr: int, segments: list):
    """Perform speaker diarization using clustering"""
    print("🔍 Performing speaker diarization...")
    
    # Extract features for each segment
    features_list = []
    valid_segments = []
    
    for segment in segments:
        features = extract_speaker_features(audio, sr, segment['start'], segment['end'])
        if features is not None:
            features_list.append(features)
            valid_segments.append(segment)
    
    if len(features_list) < 2:
        print("⚠️ Not enough segments for clustering, using simple alternating")
        for i, segment in enumerate(segments):
            segment['speaker'] = f"Speaker_{(i % 2) + 1}"
        return segments
    
    # Standardize features
    features_array = np.array(features_list)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    # Cluster speakers (assume 2 speakers)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    speaker_labels = kmeans.fit_predict(features_scaled)
    
    # Assign speaker labels
    for i, segment in enumerate(valid_segments):
        segment['speaker'] = f"Speaker_{speaker_labels[i] + 1}"
    
    # Merge consecutive segments from same speaker
    merged_segments = []
    current_segment = None
    
    for segment in valid_segments:
        if current_segment is None:
            current_segment = segment.copy()
        elif (current_segment['speaker'] == segment['speaker'] and 
              segment['start'] - current_segment['end'] < 1.0):
            # Merge segments
            current_segment['end'] = segment['end']
            current_segment['text'] += " " + segment['text']
        else:
            merged_segments.append(current_segment)
            current_segment = segment.copy()
    
    if current_segment is not None:
        merged_segments.append(current_segment)
    
    print(f"✅ Detected 2 speakers in {len(merged_segments)} segments")
    return merged_segments

def transcribe_with_speakers(audio_path: str):
    """Transcribe audio file with speaker separation"""
    
    print(f"🎤 Processing: {audio_path}")
    print("="*60)
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return
    
    # Check for fine-tuned model
    model_dir = "whisper-dailytalk-3k"
    if not os.path.exists(model_dir):
        print(f"⚠️ Fine-tuned model not found: {model_dir}")
        print("🔄 Using original Whisper model...")
        model_dir = "openai/whisper-base"
    
    print(f"🤖 Loading model: {model_dir}")
    
    try:
        # Load audio for speaker detection
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        if sr != 16000:
            print(f"🔄 Resampling from {sr}Hz to 16000Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        audio = librosa.util.normalize(audio)
        
        print(f"📊 Audio duration: {len(audio)/sr:.1f} seconds")
        
        # Load Whisper model
        processor = WhisperProcessor.from_pretrained(model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        model.eval()
        
        # Get initial transcription with timestamps
        print("🔄 Getting initial transcription...")
        
        # For speaker diarization, we need to get segments
        # Let's use a simple approach: transcribe in chunks
        chunk_duration = 5.0  # 5-second chunks
        segments = []
        
        for start_time in np.arange(0, len(audio)/sr, chunk_duration):
            end_time = min(start_time + chunk_duration, len(audio)/sr)
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            chunk_audio = audio[start_sample:end_sample]
            
            if len(chunk_audio) < 0.5 * sr:  # Skip very short chunks
                continue
            
            # Transcribe chunk
            input_features = processor.feature_extractor(
                chunk_audio, sampling_rate=sr, return_tensors="pt"
            ).input_features
            
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(
                        language="en", task="transcribe"
                    ),
                    max_length=100
                )
            
            transcription = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
            
            if transcription:
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': transcription
                })
        
        print(f"📊 Found {len(segments)} initial segments")
        
        # Perform speaker diarization
        speaker_segments = perform_speaker_diarization(audio, sr, segments)
        
        # Print results
        print(f"\n🗣️ TRANSCRIPTION WITH SPEAKERS:")
        print("="*60)
        
        for segment in speaker_segments:
            start_time = segment["start"]
            end_time = segment["end"]
            speaker = segment["speaker"]
            text = segment["text"]
            print(f"[{start_time:6.1f}s - {end_time:6.1f}s] {speaker}: {text}")
        
        # Calculate speaker statistics
        speaker_stats = {}
        for segment in speaker_segments:
            speaker = segment['speaker']
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'segments': 0,
                    'total_duration': 0.0,
                    'total_chars': 0
                }
            
            speaker_stats[speaker]['segments'] += 1
            speaker_stats[speaker]['total_duration'] += (segment['end'] - segment['start'])
            speaker_stats[speaker]['total_chars'] += len(segment['text'])
        
        print(f"\n📊 SPEAKER STATISTICS:")
        print("="*40)
        for speaker, stats in speaker_stats.items():
            print(f"{speaker}:")
            print(f"  Segments: {stats['segments']}")
            print(f"  Duration: {stats['total_duration']:.1f}s")
            print(f"  Characters: {stats['total_chars']}")
            print()
        
        # Save results
        output_file = f"speaker_transcript_{Path(audio_path).stem}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Audio file: {audio_path}\n")
            f.write(f"Model: {model_dir}\n")
            f.write(f"Duration: {len(audio)/sr:.1f}s\n\n")
            f.write("TRANSCRIPT WITH SPEAKERS:\n")
            f.write("="*60 + "\n")
            
            for segment in speaker_segments:
                start_time = segment["start"]
                end_time = segment["end"]
                speaker = segment["speaker"]
                text = segment["text"]
                f.write(f"[{start_time:6.1f}s - {end_time:6.1f}s] {speaker}: {text}\n")
            
            f.write(f"\nSPEAKER STATISTICS:\n")
            f.write("="*40 + "\n")
            for speaker, stats in speaker_stats.items():
                f.write(f"{speaker}: {stats['segments']} segments, {stats['total_duration']:.1f}s, {stats['total_chars']} chars\n")
        
        print(f"💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    transcribe_with_speakers("091452-i-834-836.wav")

if __name__ == '__main__':
    main()
