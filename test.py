#!/usr/bin/env python3
"""
Improved Whisper transcription with better speaker diarization
"""

import os
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def extract_speaker_features(audio: np.ndarray, sr: int, start_time: float, end_time: float):
    """
    Extract speaker-specific features from audio segment
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = audio[start_sample:end_sample]
    
    if len(segment) < 1024:  # Too short
        return None
    
    # Extract MFCC features (most important for speaker identification)
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # Pitch/F0 features
    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)
    
    # Energy features
    rms = librosa.feature.rms(y=segment)
    
    # Combine all features
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

def improved_speaker_diarization(audio: np.ndarray, sr: int, segments: list):
    """
    Improved speaker diarization using clustering of audio features
    """
    print("🔍 Performing improved speaker diarization...")
    
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
    
    # Determine optimal number of speakers (2-4 speakers max)
    n_speakers = min(4, max(2, len(features_list) // 3))
    
    # Cluster speakers using K-means
    kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
    speaker_labels = kmeans.fit_predict(features_scaled)
    
    # Assign speaker labels to segments
    for i, segment in enumerate(valid_segments):
        segment['speaker'] = f"Speaker_{speaker_labels[i] + 1}"
    
    # Post-process: merge consecutive segments from same speaker
    merged_segments = []
    current_segment = None
    
    for segment in valid_segments:
        if current_segment is None:
            current_segment = segment.copy()
        elif (current_segment['speaker'] == segment['speaker'] and 
              segment['start'] - current_segment['end'] < 1.0):  # Merge if gap < 1 second
            # Merge segments
            current_segment['end'] = segment['end']
            current_segment['text'] += " " + segment['text']
        else:
            # Save current segment and start new one
            merged_segments.append(current_segment)
            current_segment = segment.copy()
    
    # Don't forget the last segment
    if current_segment is not None:
        merged_segments.append(current_segment)
    
    print(f"✅ Detected {n_speakers} speakers in {len(merged_segments)} segments")
    return merged_segments

def transcribe_audio_file(audio_path: str):
    """
    Transcribe a single audio file with improved speaker diarization
    """
    print(f"\n🎤 Processing: {audio_path}")
    print("="*60)
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return None
    
    try:
        # Load audio for speaker detection
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Take first channel if stereo
        
        # Initialize Whisper model
        model = WhisperModel("base", device="cpu", compute_type="int8")
        
        # Transcribe the audio
        segments, info = model.transcribe(audio_path, language="en")
        
        # Convert segments to list
        segment_list = []
        for segment in segments:
            segment_list.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
        
        print(f"📊 Found {len(segment_list)} initial segments")
        
        # Perform improved speaker diarization
        speaker_segments = improved_speaker_diarization(audio, sr, segment_list)
        
        # Print results
        print(f"\n🗣️ TRANSCRIPT WITH IMPROVED SPEAKER SEPARATION:")
        print("-" * 60)
        
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
        
        print(f"\n📈 SPEAKER STATISTICS:")
        print("-" * 40)
        for speaker, stats in speaker_stats.items():
            print(f"{speaker}: {stats['segments']} segments, {stats['total_duration']:.1f}s, {stats['total_chars']} chars")
        
        return {
            'file': audio_path,
            'segments': speaker_segments,
            'stats': speaker_stats,
            'language': info.language,
            'confidence': info.language_probability
        }
        
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None

def main():
    """Main function to test on all available audio files."""
    print("🎵 Improved Whisper Speaker Diarization Test")
    print("="*60)
    
    # Find all available audio files
    audio_files = []
    for file in os.listdir('.'):
        if file.endswith('.wav'):
            audio_files.append(file)
    
    if not audio_files:
        print("❌ No audio files found in current directory")
        return
    
    print(f"📁 Found {len(audio_files)} audio files:")
    for file in audio_files:
        print(f"   - {file}")
    
    # Process each audio file
    results = []
    for audio_file in audio_files:
        result = transcribe_audio_file(audio_file)
        if result:
            results.append(result)
    
    # Save all results
    print(f"\n💾 SAVING RESULTS...")
    print("="*60)
    
    with open('all_transcripts_improved.txt', 'w', encoding='utf-8') as f:
        f.write("IMPROVED WHISPER TRANSCRIPTION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for result in results:
            f.write(f"FILE: {result['file']}\n")
            f.write(f"Language: {result['language']} (confidence: {result['confidence']:.2f})\n")
            f.write("-" * 40 + "\n")
            
            for segment in result['segments']:
                start_time = segment["start"]
                end_time = segment["end"]
                speaker = segment["speaker"]
                text = segment["text"]
                f.write(f"[{start_time:6.1f}s - {end_time:6.1f}s] {speaker}: {text}\n")
            
            f.write("\nSPEAKER STATISTICS:\n")
            for speaker, stats in result['stats'].items():
                f.write(f"{speaker}: {stats['segments']} segments, {stats['total_duration']:.1f}s, {stats['total_chars']} chars\n")
            
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"✅ All transcripts saved to: all_transcripts_improved.txt")
    print(f"🎯 Processed {len(results)} files successfully")

if __name__ == '__main__':
    main()
