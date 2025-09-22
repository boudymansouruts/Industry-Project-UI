#!/usr/bin/env python3
"""
Audio preprocessing utilities for Whisper speaker identification
"""

import os
import json
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        audio = librosa.util.normalize(audio)
        return audio, target_sr
        
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return np.zeros(target_sr), target_sr

def trim_audio(audio: np.ndarray, max_length_seconds: int = 20, sr: int = 16000) -> np.ndarray:
    max_samples = max_length_seconds * sr
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio

def find_speech_segments(audio: np.ndarray, sr: int = 16000, 
                        energy_threshold: float = 0.01,
                        centroid_threshold: float = 1000,
                        zcr_threshold: float = 0.1,
                        flux_threshold: float = 0.01,
                        min_segment_length: float = 0.3) -> List[Tuple[float, float]]:
    
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
    spectral_flux = np.diff(librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0])
    
    speech_frames = []
    for i in range(len(energy)):
        if (energy[i] > energy_threshold and 
            spectral_centroids[i] > centroid_threshold and 
            zcr[i] < zcr_threshold and 
            (i < len(spectral_flux) and abs(spectral_flux[i]) > flux_threshold)):
            speech_frames.append(i)
    
    if not speech_frames:
        return []
    
    segments = []
    start_frame = speech_frames[0]
    prev_frame = speech_frames[0]
    
    for frame in speech_frames[1:]:
        if frame - prev_frame > 1:
            start_time = start_frame * hop_length / sr
            end_time = prev_frame * hop_length / sr
            if end_time - start_time >= min_segment_length:
                segments.append((start_time, end_time))
            start_frame = frame
        prev_frame = frame
    
    start_time = start_frame * hop_length / sr
    end_time = prev_frame * hop_length / sr
    if end_time - start_time >= min_segment_length:
        segments.append((start_time, end_time))
    
    return segments

def extract_speaker_embeddings(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zero_crossing_rate)
        
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)
        
        pitch, _ = librosa.piptrack(y=audio, sr=sr)
        pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
        
        features = np.concatenate([
            mfccs_mean,
            [spectral_centroid_mean, spectral_rolloff_mean, zcr_mean, rms_mean, pitch_mean]
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(18)

def segment_speakers_with_global_analysis(audio: np.ndarray, sr: int = 16000) -> List[Tuple[float, float, str]]:
    speech_segments = find_speech_segments(audio, sr)
    
    if len(speech_segments) < 2:
        return [(0, len(audio) / sr, "Speaker_1")]
    
    embeddings = []
    valid_segments = []
    
    for start_time, end_time in speech_segments:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment_audio = audio[start_sample:end_sample]
        
        if len(segment_audio) > sr * 0.3:
            embedding = extract_speaker_embeddings(segment_audio, sr)
            if embedding is not None and len(embedding) > 0:
                if len(embedding) < 39:
                    embedding = np.pad(embedding, (0, 39 - len(embedding)))
                else:
                    embedding = embedding[:39]
                embeddings.append(embedding)
                valid_segments.append((start_time, end_time))
    
    if len(embeddings) < 2:
        return [(0, len(audio) / sr, "Speaker_1")]
    
    embeddings = np.array(embeddings)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    speaker_labels = kmeans.fit_predict(embeddings_scaled)
    
    speaker_labels = smooth_speaker_transitions(speaker_labels, window_size=3)
    
    result = []
    for i, (start_time, end_time) in enumerate(valid_segments):
        speaker_id = speaker_labels[i] + 1
        result.append((start_time, end_time, f"Speaker_{speaker_id}"))
    
    return result

def smooth_speaker_transitions(labels: np.ndarray, window_size: int = 3) -> np.ndarray:
    if len(labels) <= window_size:
        return labels
    
    smoothed = labels.copy()
    
    for i in range(window_size, len(labels) - window_size):
        window = labels[i-window_size:i+window_size+1]
        unique, counts = np.unique(window, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        
        if labels[i] != majority_label and counts[np.argmax(counts)] >= window_size:
            smoothed[i] = majority_label
    
    return smoothed

def load_dailytalk_dataset(data_dir: str, max_samples: int = None) -> Tuple[List[Dict], Dict]:
    samples = []
    speaker_mapping = {}
    speaker_counter = 1
    
    for speaker_dir in sorted(os.listdir(data_dir)):
        speaker_path = os.path.join(data_dir, speaker_dir)
        if not os.path.isdir(speaker_path):
            continue
        
        if speaker_dir not in speaker_mapping:
            speaker_mapping[speaker_dir] = f"Speaker_{speaker_counter}"
            speaker_counter += 1
        
        for file in os.listdir(speaker_path):
            if file.endswith('.txt'):
                txt_path = os.path.join(speaker_path, file)
                wav_path = os.path.join(speaker_path, file.replace('.txt', '.wav'))
                
                if os.path.exists(wav_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        if text:
                            samples.append({
                                'audio_path': wav_path,
                                'text': f"[{speaker_mapping[speaker_dir]}]: {text}",
                                'original_text': text,
                                'original_speaker': speaker_dir,
                                'generic_speaker': speaker_mapping[speaker_dir]
                            })
                            
                            if max_samples and len(samples) >= max_samples:
                                return samples, speaker_mapping
                                
                    except Exception as e:
                        print(f"Error processing {txt_path}: {e}")
                        continue
        
        if len(samples) % 50 == 0 and len(samples) > 0:
            print(f"   Loaded {len(samples)} samples...")
    
    return samples, speaker_mapping