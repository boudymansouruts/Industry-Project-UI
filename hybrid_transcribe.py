#!/usr/bin/env python3
"""
Hybrid Transcription with Fixed Speaker Diarization
"""

import os
import time
from pathlib import Path
import argparse
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import re
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")


def detect_speech_segments(audio: np.ndarray, sr: int = 16000) -> List[Tuple[float, float]]:
    """Detect all speech segments without assuming speaker changes."""
    frame_length = int(0.025 * sr)
    hop_length = int(0.01 * sr)
    
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    silence_threshold = np.percentile(rms, 40)  # Increased from 35 to 40 for more conservative splitting
    speech_frames = rms > silence_threshold
    
    frame_times = np.arange(len(rms)) * hop_length / sr
    
    speech_segments = []
    in_speech = False
    speech_start = 0
    
    for i, is_speech in enumerate(speech_frames):
        if is_speech and not in_speech:
            speech_start = frame_times[i]
            in_speech = True
        elif not is_speech and in_speech:
            speech_end = frame_times[i]
            if speech_end - speech_start >= 0.8:  # Increased from 0.5 to 0.8 for longer, more stable segments
                speech_segments.append((speech_start, speech_end))
            in_speech = False
    
    if in_speech:
        speech_end = frame_times[-1]
        if speech_end - speech_start >= 0.5:  # Increased from 0.3 to 0.5
            speech_segments.append((speech_start, speech_end))
    
    return speech_segments


def extract_voice_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract comprehensive voice features for speaker identification."""
    try:
        # Fundamental frequency
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)
        pitch_values = pitches[magnitudes > 0.1 * np.max(magnitudes)]
        
        if len(pitch_values) > 10:
            f0_mean = np.mean(pitch_values)
            f0_std = np.std(pitch_values)
            f0_min = np.min(pitch_values)
            f0_max = np.max(pitch_values)
            f0_range = f0_max - f0_min
        else:
            f0_mean = f0_std = f0_min = f0_max = f0_range = 0
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        sc_mean = np.mean(spectral_centroids)
        sc_std = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        sr_mean = np.mean(spectral_rolloff)
        sr_std = np.std(spectral_rolloff)
        
        # Voice quality
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zero_crossing_rate)
        zcr_std = np.std(zero_crossing_rate)
        
        # Energy
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=6)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        features = np.array([
            f0_mean, f0_std, f0_min, f0_max, f0_range,
            sc_mean, sc_std, sr_mean, sr_std,
            zcr_mean, zcr_std, rms_mean, rms_std,
            *mfcc_means, *mfcc_stds
        ])
        
        return features
        
    except Exception as e:
        return np.zeros(25)


def voice_based_diarization(audio: np.ndarray, sr: int = 16000) -> List[Tuple[float, float, str]]:
    """Voice-based diarization using proper clustering."""
    speech_segments = detect_speech_segments(audio, sr)
    
    if len(speech_segments) < 2:
        return [(0, len(audio) / sr, "Speaker_1")]
    
    features = []
    valid_segments = []
    
    for start_time, end_time in speech_segments:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment_audio = audio[start_sample:end_sample]
        
        if len(segment_audio) > sr * 1.0:  # Increased from 0.5 to 1.0 for longer, more reliable segments
            voice_features = extract_voice_features(segment_audio, sr)
            if not np.all(voice_features == 0):
                features.append(voice_features)
                valid_segments.append((start_time, end_time))
    
    if len(features) < 2:
        return [(0, len(audio) / sr, "Speaker_1")]
    
    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Find optimal number of speakers with better stability
    best_score = -1
    best_labels = None
    best_n_speakers = 2
    
    # Try different numbers of speakers, but prefer fewer speakers for stability
    for n_speakers in range(2, min(5, len(features) + 1)):  # Reduced max from 6 to 5
        try:
            kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=30)  # Increased n_init for stability
            labels = kmeans.fit_predict(features_scaled)
            
            if len(set(labels)) > 1:
                score = silhouette_score(features_scaled, labels)
                # Prefer fewer speakers if scores are close (within 0.05)
                if score > best_score + 0.05 or (score > best_score - 0.05 and n_speakers < best_n_speakers):
                    best_score = score
                    best_labels = labels
                    best_n_speakers = n_speakers
        except:
            continue
    
    if best_labels is None:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
        best_labels = kmeans.fit_predict(features_scaled)
    
    # Create result
    result = []
    for i, (start_time, end_time) in enumerate(valid_segments):
        speaker_id = best_labels[i] + 1
        result.append((start_time, end_time, f"Speaker_{speaker_id}"))
    
    # Sort by time and merge close segments from same speaker
    result.sort(key=lambda x: x[0])
    
    merged_result = []
    for start_time, end_time, speaker in result:
        if (merged_result and 
            merged_result[-1][2] == speaker and 
            start_time - merged_result[-1][1] < 3.0):  # Increased from 1.0 to 3.0 seconds
            merged_result[-1] = (merged_result[-1][0], end_time, speaker)
        else:
            merged_result.append((start_time, end_time, speaker))
    
    return merged_result


def transcribe_full_audio(audio_file: str, model_dir: str = None):
    if model_dir is None:
        model_dir = get_model_path()
    
    print(f"Loading audio: {Path(audio_file).name}")
    print(f"Using model: {model_dir}")

    processor = WhisperProcessor.from_pretrained(model_dir, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load audio using soundfile for better compatibility
    audio, sr = sf.read(audio_file)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # Resample to 16kHz if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    audio = librosa.util.normalize(audio)
    audio_duration = len(audio) / sr
    print(f"Audio duration: {audio_duration:.1f} seconds")

    chunk_size = 30 * sr
    overlap = 2 * sr
    transcriptions: List[str] = []

    print("Transcribing in chunks...")

    for i in range(0, len(audio), chunk_size - overlap):
        chunk = audio[i:i + chunk_size]

        if len(chunk) < sr * 0.5:
            continue

        if len(chunk) < sr:
            chunk = np.pad(chunk, (0, sr - len(chunk)))

        with torch.no_grad():
            input_features = processor.feature_extractor(
                chunk, sampling_rate=16000, return_tensors="pt",
                padding="max_length", max_length=processor.feature_extractor.n_samples
            ).input_features.to(device)

            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
                max_length=448,
                num_beams=6,
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.1,
                num_return_sequences=1,
                use_cache=True
            )

            transcription = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()

            if transcription and len(transcription) > 3:
                transcriptions.append(transcription)
                print(f"   Chunk {len(transcriptions)}: {len(transcription)} chars")

    full_transcription = merge_overlapping_transcriptions(transcriptions)
    print(f"Full transcription: {len(full_transcription.split())} words")
    return full_transcription, audio, sr


def transcribe_audio_slice(
    audio: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: torch.device
):
    """Transcribe a specific audio slice [start_time, end_time)."""
    start_sample = max(0, int(start_time * sr))
    end_sample = min(len(audio), int(end_time * sr))
    slice_audio = audio[start_sample:end_sample]

    if len(slice_audio) < sr * 0.4:
        return ""

    if len(slice_audio) < sr:
        slice_audio = np.pad(slice_audio, (0, sr - len(slice_audio)))

    with torch.no_grad():
        input_features = processor.feature_extractor(
            slice_audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
            max_length=processor.feature_extractor.n_samples,
        ).input_features.to(device)

        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
            max_length=448,
            num_beams=6,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.2,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.1,
            num_return_sequences=1,
            use_cache=True
        )

        text = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

    return text


def windowed_asr_segments(audio: np.ndarray, sr: int, model_dir: str) -> list:
    """Run ASR in fixed windows independent of diarization to get time-stamped text segments.

    We use 15s windows with 2s overlap to improve coverage, then keep windows with
    non-empty text. Start/end times are window bounds (approximate but stable).
    """
    processor = WhisperProcessor.from_pretrained(model_dir, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    window_sec = 10.0
    overlap_sec = 0.0  # No overlap for cleanest transcription
    step = int((window_sec - overlap_sec) * sr)
    size = int(window_sec * sr)

    segments = []
    for start in range(0, len(audio), step):
        end = min(len(audio), start + size)
        text = transcribe_audio_slice(audio, sr, start / sr, end / sr, processor, model, device)
        if text and len(text.strip()) > 1:
            segments.append({
                "start_time": start / sr,
                "end_time": end / sr,
                "text": text.strip(),
                "word_count": len(text.strip().split()),
            })

    # Keep fine-grained windows; do not merge here to avoid cross-speaker mixing
    return sorted(segments, key=lambda s: s["start_time"])


def assign_speakers_by_overlap(asr_segments: list, speaker_segments: list) -> list:
    """Assign each ASR segment a speaker label by maximum time overlap with diarized segments.
    Includes temporal smoothing to reduce speaker switching."""
    assigned = []
    
    for seg in asr_segments:
        s0, s1 = seg["start_time"], seg["end_time"]
        best, best_overlap = ("Speaker_1",), 0.0
        
        for (d0, d1, spk) in speaker_segments:
            ov = max(0.0, min(s1, d1) - max(s0, d0))
            if ov > best_overlap:
                best_overlap = ov
                best = (spk,)
        
        speaker = best[0] if best else "Speaker_1"
        
        # Apply temporal smoothing - if previous segment was assigned to same speaker
        # and they're close in time, keep the same speaker
        if assigned and len(assigned) > 0:
            prev_segment = assigned[-1]
            prev_speaker = prev_segment["speaker"]
            prev_end = prev_segment["end_time"]
            
            # If segments are close in time (within 3 seconds) and previous speaker
            # has some overlap, prefer continuity
            if (s0 - prev_end) < 3.0:
                # Check if previous speaker has any overlap with current segment
                prev_overlap = 0.0
                for (d0, d1, spk) in speaker_segments:
                    if spk == prev_speaker:
                        ov = max(0.0, min(s1, d1) - max(s0, d0))
                        prev_overlap = max(prev_overlap, ov)
                
                # If previous speaker has significant overlap, prefer continuity
                if prev_overlap > best_overlap * 0.6:  # 60% threshold for continuity
                    speaker = prev_speaker
        
        assigned.append({**seg, "speaker": speaker})
    
    return assigned


def transcribe_with_timestamps(audio: np.ndarray, sr: int, model_dir: str = None) -> list:
    if model_dir is None:
        model_dir = get_model_path()
    
    """Transcribe entire audio with timestamps using Whisper's built-in segmentation."""
    processor = WhisperProcessor.from_pretrained(model_dir, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Process entire audio at once to get proper segmentation
    input_features = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features
    input_features = input_features.to(device)

    # Generate with timestamps
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
            max_length=448,
            num_beams=6,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.2,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.1,
            num_return_sequences=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Decode the transcription
    transcription = processor.tokenizer.batch_decode(predicted_ids.sequences, skip_special_tokens=True)[0]
    
    # Create segments based on sentence boundaries with estimated timestamps
    sentences = transcription.split('. ')
    segments = []
    current_time = 0.0
    
    for sentence in sentences:
        if sentence.strip():
            # Estimate duration based on text length (rough approximation)
            estimated_duration = len(sentence.split()) * 0.5  # ~0.5 seconds per word
            end_time = current_time + estimated_duration
            
            segments.append({
                "start_time": current_time,
                "end_time": end_time,
                "text": sentence.strip() + ('.' if not sentence.endswith('.') else ''),
                "speaker": None,  # Will be assigned later
                "word_count": len(sentence.strip().split())  # Add word count
            })
            
            current_time = end_time
    
    return segments


def assign_speakers_to_transcript(transcript_segments: list, speaker_segments: list) -> list:
    """Assign speakers to transcript segments based on temporal overlap."""
    assigned_segments = []
    
    for transcript_seg in transcript_segments:
        start_time = transcript_seg["start_time"]
        end_time = transcript_seg["end_time"]
        
        # Find the speaker with maximum overlap
        best_speaker = "Speaker_1"
        best_overlap = 0.0
        
        for spk_start, spk_end, speaker in speaker_segments:
            overlap_start = max(start_time, spk_start)
            overlap_end = min(end_time, spk_end)
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > best_overlap:
                    best_overlap = overlap_duration
                    best_speaker = speaker
        
        # Apply temporal smoothing for continuity
        if assigned_segments and len(assigned_segments) > 0:
            prev_segment = assigned_segments[-1]
            prev_speaker = prev_segment["speaker"]
            prev_end = prev_segment["end_time"]
            
            # If segments are close in time (within 2 seconds), prefer continuity
            if (start_time - prev_end) < 2.0:
                # Check if previous speaker has any overlap
                prev_overlap = 0.0
                for spk_start, spk_end, speaker in speaker_segments:
                    if speaker == prev_speaker:
                        overlap_start = max(start_time, spk_start)
                        overlap_end = min(end_time, spk_end)
                        if overlap_start < overlap_end:
                            prev_overlap = max(prev_overlap, overlap_end - overlap_start)
                
                # If previous speaker has significant overlap, prefer continuity
                if prev_overlap > best_overlap * 0.5:  # 50% threshold for continuity
                    best_speaker = prev_speaker
        
        assigned_segments.append({
            **transcript_seg,
            "speaker": best_speaker
        })
    
    return assigned_segments


def build_speaker_based_segments(audio: np.ndarray, sr: int, model_dir: str = None) -> list:
    if model_dir is None:
        model_dir = get_model_path()
    
    # Step 1: Transcribe entire audio in windows to cover full duration
    transcript_segments = windowed_asr_segments(audio, sr, model_dir)
    
    # Step 2: Do speaker diarization separately
    speaker_segments = voice_based_diarization(audio, sr)
    if not speaker_segments:
        speaker_segments = [(0.0, len(audio) / sr, "Speaker_1")]
    
    # Step 3: Assign speakers to transcript segments
    final_segments = assign_speakers_to_transcript(transcript_segments, speaker_segments)
    
    return sorted(final_segments, key=lambda c: c["start_time"])


def merge_overlapping_transcriptions(transcriptions: List[str]) -> str:
    if not transcriptions:
        return ""
    if len(transcriptions) == 1:
        return transcriptions[0]

    merged = transcriptions[0]
    for i in range(1, len(transcriptions)):
        current = transcriptions[i]

        merged_words = merged.split()
        current_words = current.split()

        best_overlap = 0
        overlap_start = 0

        for j in range(1, min(11, len(merged_words), len(current_words)) + 1):
            merged_tail = " ".join(merged_words[-j:]).lower()
            current_head = " ".join(current_words[:j]).lower()
            if merged_tail == current_head:
                best_overlap = j
                overlap_start = j

        if best_overlap > 0:
            merged += " " + " ".join(current_words[overlap_start:])
        else:
            merged += " " + current

    return merged


def identify_speakers_in_transcription(transcription: str, audio: np.ndarray, sr: int) -> str:
    print("Identifying speakers in transcription...")

    # Use the new voice-based diarization
    speaker_segments = voice_based_diarization(audio, sr)
    print(f"Found {len(speaker_segments)} speaker segments")

    if len(speaker_segments) <= 1:
        return add_single_speaker(transcription)

    return map_speakers_to_transcription(transcription, speaker_segments, len(audio) / sr)


def map_speakers_to_transcription(transcription: str, segments: List[Tuple[float, float, str]], audio_duration: float) -> str:
    print("Mapping speakers to transcription...")

    sentences = split_into_sentences(transcription)
    if len(sentences) <= 1:
        return f"[Speaker_1]: {transcription}"

    total_chars = sum(len(s) for s in sentences)

    result: List[str] = []
    current_time = 0.0
    current_speaker_id = None
    current_speaker_name = None
    current_text: List[str] = []

    for sentence in sentences:
        sentence_duration = (len(sentence) / total_chars) * audio_duration
        sentence_time = current_time + sentence_duration / 2

        segment_idx = find_segment_for_time(sentence_time, segments)

        if segment_idx is not None and segment_idx < len(segments):
            _, _, speaker_name = segments[segment_idx]

            if current_speaker_id is None or speaker_name != current_speaker_name:
                if current_text:
                    result.append(f"[{current_speaker_name}]: {' '.join(current_text)}")
                current_speaker_name = speaker_name
                current_text = [sentence.strip()]
            else:
                current_text.append(sentence.strip())
        else:
            if current_speaker_name is None:
                current_speaker_name = "Speaker_1"
            current_text.append(sentence.strip())

        current_time += sentence_duration

    if current_text:
        result.append(f"[{current_speaker_name}]: {' '.join(current_text)}")

    return "\n\n".join(result)


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'([.!?]+\s+)', text)

    combined_sentences: List[str] = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        if i + 1 < len(sentences):
            punctuation = sentences[i + 1].strip()
            if sentence and len(sentence) > 3:
                if punctuation and punctuation[0] in '.,!?':
                    sentence += punctuation[0]
                combined_sentences.append(sentence)
        elif sentence and len(sentence) > 3:
            combined_sentences.append(sentence)

    if len(sentences) % 2 == 1 and sentences[-1].strip():
        last_sentence = sentences[-1].strip()
        if len(last_sentence) > 3:
            combined_sentences.append(last_sentence)

    refined_sentences: List[str] = []
    for sentence in combined_sentences:
        parts = re.split(r'(,\s+|;\s+|\s+and\s+|\s+but\s+|\s+so\s+)', sentence)
        current_part = ""

        for part in parts:
            if re.match(r'(,\s+|;\s+|\s+and\s+|\s+but\s+|\s+so\s+)', part):
                current_part += part
            else:
                current_part += part
                if len(current_part.strip()) > 10:
                    refined_sentences.append(current_part.strip())
                    current_part = ""

        if current_part.strip() and len(current_part.strip()) > 3:
            refined_sentences.append(current_part.strip())

    return refined_sentences if refined_sentences else combined_sentences


def find_segment_for_time(target_time: float, segments: List[Tuple[float, float, str]]):
    for i, (start, end, speaker) in enumerate(segments):
        if start <= target_time <= end:
            return i

    min_distance = float('inf')
    closest_idx = 0
    for i, (start, end, speaker) in enumerate(segments):
        mid_time = (start + end) / 2
        distance = abs(target_time - mid_time)
        if distance < min_distance:
            min_distance = distance
            closest_idx = i
    return closest_idx


def add_single_speaker(transcription: str) -> str:
    return f"[Speaker_1]: {transcription}"


def merge_consecutive_same_speaker(speaker_text: str) -> str:
    lines = [line.strip() for line in speaker_text.split('\n')]
    merged_blocks: List[str] = []
    current_speaker = None
    current_content_parts: List[str] = []

    for line in lines:
        if not line:
            continue

        if line.startswith('[') and ']: ' in line:
            speaker_tag, content = line.split(']: ', 1)
            speaker_tag = speaker_tag + ']'

            if current_speaker is None:
                current_speaker = speaker_tag
                current_content_parts = [content.strip()]
            elif speaker_tag == current_speaker:
                if content.strip():
                    current_content_parts.append(content.strip())
            else:
                merged_blocks.append(f"{current_speaker}: {' '.join(current_content_parts)}")
                current_speaker = speaker_tag
                current_content_parts = [content.strip()]
        else:
            if current_speaker is not None and line:
                current_content_parts.append(line)

    if current_speaker is not None and current_content_parts:
        merged_blocks.append(f"{current_speaker}: {' '.join(current_content_parts)}")

    return '\n\n'.join(merged_blocks)


def get_model_path():
    """Get the current model path from configuration"""
    try:
        import json
        config_file = Path("model_config.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get("model_info", {}).get("path", "openai/whisper-large-v2")
    except:
        pass
    return "openai/whisper-large-v2"  # Default fallback


def hybrid_transcribe_audio(audio_file: str, model_dir: str = None):
    print(f"HYBRID TRANSCRIPTION: {Path(audio_file).name}")
    print("=" * 60)

    # Start timing
    start_time = time.time()

    if model_dir is None:
        model_dir = get_model_path()
    
    print(f"Using model: {model_dir}")
    # Reuse loader from transcribe_full_audio for consistency
    _, audio, sr = transcribe_full_audio(audio_file, model_dir)

    print("\nSTEP 2: Transcribe first, then assign speakers")
    diarized_chunks = build_speaker_based_segments(audio, sr, model_dir)

    # Build human-readable transcript from diarized chunks
    blocks = []
    for ch in diarized_chunks:
        blocks.append(f"[{ch['speaker']}]: {ch['text']}")
    speaker_transcription = "\n\n".join(blocks)

    end_time = time.time()
    processing_time = end_time - start_time

    print("\nHYBRID TRANSCRIPTION COMPLETED")
    print(f"Total processing time: {processing_time:.1f} seconds")
    total_words = sum(ch["word_count"] for ch in diarized_chunks)
    print(f"Words: {total_words} (speaker-segmented)")
    
    # Print the results
    print("\n" + "="*50)
    print("FINAL TRANSCRIPT WITH SPEAKERS")
    print("="*50)
    print()
    print(speaker_transcription)
    print()

    return {
        'raw_transcription': " ".join(ch["text"] for ch in diarized_chunks),
        'speaker_transcription': speaker_transcription,
        'processing_time': processing_time,
        'audio_duration': len(audio) / sr,
        'segments': diarized_chunks
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid transcription with fixed speaker identification")
    parser.add_argument("audio", type=str, help="Path to input .wav file")
    parser.add_argument("--model_dir", type=str, default="openai/whisper-large-v2", help="Path to Whisper model")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write results .txt")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"File not found: {args.audio}")
        return

    result = hybrid_transcribe_audio(args.audio, args.model_dir)

    if args.out:
        audio_name = Path(args.audio).stem
        out_path = args.out
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("HYBRID TRANSCRIPTION RESULTS\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Audio: {Path(args.audio).name}\n")
            f.write(f"Duration: {result['audio_duration']:.1f} seconds\n")
            f.write(f"Processing Time: {result['processing_time']:.1f} seconds\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("RAW TRANSCRIPTION (High Quality):\n")
            f.write("-" * 35 + "\n")
            f.write(result['raw_transcription'])
            f.write("\n\n")
            f.write("SPEAKER-AWARE TRANSCRIPTION:\n")
            f.write("-" * 30 + "\n")
            f.write(result['speaker_transcription'])
            f.write("\n")
        print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()