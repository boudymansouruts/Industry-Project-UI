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

def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        audio, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        # Resample to target sample rate if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        return audio, target_sr
        
    except Exception as e:
        print(f"⚠️ Error loading audio {audio_path}: {e}")
        return np.zeros(target_sr), target_sr

def trim_audio(audio: np.ndarray, max_length_seconds: int = 20, sample_rate: int = 16000) -> np.ndarray:
    """
    Trim audio to maximum length
    
    Args:
        audio: Audio array
        max_length_seconds: Maximum length in seconds
        sample_rate: Sample rate of audio
    
    Returns:
        Trimmed audio array
    """
    max_length = max_length_seconds * sample_rate
    if len(audio) > max_length:
        audio = audio[:max_length]
    return audio

def detect_language(text: str) -> str:
    """
    Detect language of the text using pattern matching
    
    Args:
        text: Text to analyze
    
    Returns:
        Language code (en, es, fr, de, ar, tl, el)
    """
    text_lower = text.lower()
    
    # Enhanced language detection patterns
    en_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among']
    es_words = ['el', 'la', 'los', 'las', 'de', 'del', 'en', 'con', 'por', 'para', 'que', 'como', 'pero', 'sin', 'sobre', 'entre', 'durante', 'desde', 'hasta', 'hacia', 'según']
    fr_words = ['le', 'la', 'les', 'de', 'du', 'des', 'en', 'avec', 'pour', 'que', 'comme', 'mais', 'sans', 'sur', 'entre', 'pendant', 'depuis', 'jusqu', 'vers', 'selon']
    de_words = ['der', 'die', 'das', 'und', 'oder', 'aber', 'in', 'auf', 'zu', 'für', 'von', 'mit', 'durch', 'über', 'unter', 'zwischen', 'während', 'seit', 'bis', 'nach', 'vor']
    
    # Arabic words (transliterated)
    ar_words = ['al', 'wa', 'fi', 'min', 'ila', 'ma', 'kana', 'lam', 'la', 'bi', 'li', 'an', 'hu', 'hi', 'hum', 'hunna', 'anta', 'anti', 'antum', 'antunna']
    
    # Tagalog words
    tl_words = ['ang', 'ng', 'sa', 'na', 'ay', 'at', 'mga', 'ko', 'mo', 'niya', 'namin', 'ninyo', 'nila', 'ako', 'ikaw', 'siya', 'kami', 'kayo', 'sila', 'ito', 'iyan', 'iyon']
    
    # Greek words (transliterated)
    el_words = ['kai', 'to', 'kai', 'tha', 'einai', 'me', 'se', 'apo', 'gia', 'os', 'den', 'na', 'tha', 'mou', 'sou', 'tou', 'tis', 'mas', 'sas', 'tous']
    
    en_score = sum(1 for word in en_words if word in text_lower)
    es_score = sum(1 for word in es_words if word in text_lower)
    fr_score = sum(1 for word in fr_words if word in text_lower)
    de_score = sum(1 for word in de_words if word in text_lower)
    ar_score = sum(1 for word in ar_words if word in text_lower)
    tl_score = sum(1 for word in tl_words if word in text_lower)
    el_score = sum(1 for word in el_words if word in text_lower)
    
    scores = {'en': en_score, 'es': es_score, 'fr': fr_score, 'de': de_score, 'ar': ar_score, 'tl': tl_score, 'el': el_score}
    return max(scores, key=scores.get) if max(scores.values()) > 0 else 'en'

def format_text_with_speaker(text: str, speaker_id: int, language: str = "en", speaker_mapping: Dict[int, str] = None) -> str:
    """
    Format text with speaker identification
    
    Args:
        text: Original text
        speaker_id: Speaker ID
        language: Language code
        speaker_mapping: Optional mapping of speaker IDs to names
    
    Returns:
        Formatted text with speaker label
    """
    if speaker_mapping and speaker_id in speaker_mapping:
        speaker_name = speaker_mapping[speaker_id]
    else:
        speaker_name = f"Speaker_{speaker_id + 1}"
    
    lang_code = language.upper()[:2]
    return f"[{speaker_name} ({lang_code})]: {text}"

def load_dailytalk_dataset(data_dir: str, max_samples: Optional[int] = None) -> Tuple[List[Dict], Dict[int, str]]:
    """
    Load DailyTalk dataset with preprocessing
    
    Args:
        data_dir: Path to data directory
        max_samples: Maximum number of samples to load
    
    Returns:
        Tuple of (samples_list, speaker_mapping)
    """
    data_path = Path(data_dir)
    metadata_path = data_path.parent / "metadata.json"
    
    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        return [], {}
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    samples = []
    speaker_mapping = {}
    count = 0
    
    print(f"📖 Loading DailyTalk dataset...")
    
    for dialog_id in sorted(metadata.keys(), key=int):
        if max_samples and count >= max_samples:
            break
        
        dialog_data = metadata[dialog_id]
        dialog_dir = data_path / dialog_id
        
        if not dialog_dir.exists():
            continue
        
        for utterance_id in sorted(dialog_data.keys(), key=int):
            if max_samples and count >= max_samples:
                break
            
            utterance_data = dialog_data[utterance_id]
            speaker = utterance_data['speaker']
            text = utterance_data['text'].strip()
            
            if not text or len(text) < 3:
                continue
            
            # Detect language and format text with speaker
            detected_language = detect_language(text)
            formatted_text = format_text_with_speaker(text, speaker, detected_language, speaker_mapping)
            
            # Create speaker mapping
            if speaker not in speaker_mapping:
                speaker_mapping[speaker] = f"Speaker_{len(speaker_mapping) + 1}"
            
            audio_file = f"{utterance_id}_{speaker}_d{dialog_id}.wav"
            audio_path = dialog_dir / audio_file
            
            if audio_path.exists():
                samples.append({
                    'audio_path': str(audio_path),
                    'text': formatted_text,
                    'original_text': text,
                    'original_speaker': speaker,
                    'generic_speaker': speaker_mapping[speaker],
                    'language': detected_language
                })
                count += 1
                
                if count % 50 == 0:
                    print(f"   Loaded {count} samples...")
    
    print(f"✅ Loaded {len(samples)} samples")
    print(f"👥 Found {len(speaker_mapping)} unique speakers")
    
    return samples, speaker_mapping

def split_audio_into_chunks(audio: np.ndarray, chunk_length_seconds: int = 4, sample_rate: int = 16000) -> List[np.ndarray]:
    """
    Split audio into chunks for speaker analysis
    
    Args:
        audio: Audio array
        chunk_length_seconds: Length of each chunk in seconds
        sample_rate: Sample rate of audio
    
    Returns:
        List of audio chunks
    """
    chunk_length = chunk_length_seconds * sample_rate
    chunks = []
    
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i + chunk_length]
        if len(chunk) > 1 * sample_rate:  # Only use chunks longer than 1 second
            chunks.append(chunk)
    
    return chunks

def analyze_audio_features(audio_chunk: np.ndarray, sample_rate: int = 16000) -> Dict:
    """
    Analyze audio features for speaker identification
    
    Args:
        audio_chunk: Audio chunk to analyze
        sample_rate: Sample rate of audio
    
    Returns:
        Dictionary of audio features
    """
    # Calculate various audio features
    rms = np.sqrt(np.mean(audio_chunk**2))
    zero_crossings = np.sum(librosa.zero_crossings(audio_chunk))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_chunk, sr=sample_rate))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_chunk, sr=sample_rate))
    mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    
    return {
        'rms': rms,
        'zero_crossings': zero_crossings,
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'mfcc_mean': mfcc_mean,
        'duration': len(audio_chunk) / sample_rate
    }

def identify_speakers_by_features(chunk_features: List[Dict]) -> List[str]:
    """
    Identify speakers based on audio features using improved clustering
    
    Args:
        chunk_features: List of feature dictionaries
    
    Returns:
        List of speaker assignments
    """
    # Filter out None features
    valid_features = [f for f in chunk_features if f is not None]
    
    if len(valid_features) < 2:
        return ["Speaker_1"] * len(chunk_features)
    
    # Extract features for clustering
    features_matrix = []
    for f in valid_features:
        features_matrix.append([
            f['rms'],
            f['zero_crossings'] / 1000,  # Normalize
            f['spectral_centroid'] / 1000,  # Normalize
            f['spectral_rolloff'] / 1000,  # Normalize
            f['mfcc_mean'][0],  # First MFCC coefficient
            f['mfcc_mean'][1],  # Second MFCC coefficient
            f['mfcc_mean'][2],  # Third MFCC coefficient
        ])
    
    features_matrix = np.array(features_matrix)
    
    # Improved clustering using multiple features
    rms_values = features_matrix[:, 0]
    spectral_centroids = features_matrix[:, 2]
    mfcc1_values = features_matrix[:, 4]
    mfcc2_values = features_matrix[:, 5]
    
    # Use multiple thresholds for better separation
    rms_threshold = np.median(rms_values)
    spectral_threshold = np.median(spectral_centroids)
    mfcc1_threshold = np.median(mfcc1_values)
    
    speaker_assignments = []
    for i, f in enumerate(chunk_features):
        if f is None:
            speaker_assignments.append("Speaker_1")
        else:
            # Use multiple features for better speaker separation
            score = 0
            if f['rms'] > rms_threshold:
                score += 1
            if f['spectral_centroid'] > spectral_threshold:
                score += 1
            if f['mfcc_mean'][0] > mfcc1_threshold:
                score += 1
            
            # Assign speaker based on majority vote
            if score >= 2:
                speaker_assignments.append("Speaker_1")
            else:
                speaker_assignments.append("Speaker_2")
    
    # Apply smoothing to reduce speaker switching errors
    speaker_assignments = smooth_speaker_assignments(speaker_assignments)
    
    return speaker_assignments

def smooth_speaker_assignments(assignments: List[str]) -> List[str]:
    """
    Smooth speaker assignments to reduce rapid switching
    
    Args:
        assignments: List of speaker assignments
    
    Returns:
        Smoothed speaker assignments
    """
    if len(assignments) < 3:
        return assignments
    
    smoothed = assignments.copy()
    
    # Apply median filter to reduce noise
    for i in range(1, len(assignments) - 1):
        if assignments[i-1] == assignments[i+1] and assignments[i] != assignments[i-1]:
            # If neighbors are the same but current is different, change current
            smoothed[i] = assignments[i-1]
    
    return smoothed

def create_speaker_transcription(chunk_transcriptions: List[str], speaker_assignments: List[str]) -> str:
    """
    Create final transcription with speaker labels
    
    Args:
        chunk_transcriptions: List of chunk transcriptions
        speaker_assignments: List of speaker assignments
    
    Returns:
        Final formatted transcription
    """
    final_parts = []
    current_speaker = None
    current_text = []
    
    for i, (transcription, speaker) in enumerate(zip(chunk_transcriptions, speaker_assignments)):
        if not transcription:
            continue
            
        if speaker != current_speaker:
            # Save previous speaker's text
            if current_speaker and current_text:
                speaker_text = " ".join(current_text)
                final_parts.append(f"[{current_speaker} (EN)]: {speaker_text}")
            
            # Start new speaker
            current_speaker = speaker
            current_text = [transcription]
        else:
            # Same speaker, add to current text
            current_text.append(transcription)
    
    # Add the last speaker's text
    if current_speaker and current_text:
        speaker_text = " ".join(current_text)
        final_parts.append(f"[{current_speaker} (EN)]: {speaker_text}")
    
    return "\n".join(final_parts)

