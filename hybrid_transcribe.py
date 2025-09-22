#!/usr/bin/env python3
"""
Hybrid Transcription with Post-Processing Speaker Identification
"""

import os
import time
from pathlib import Path
import argparse
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from preprocess import load_audio, find_speech_segments, extract_speaker_embeddings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
from typing import List, Tuple


def transcribe_full_audio(audio_file: str, model_dir: str = "whisper-large-full"):
    print(f"Loading audio: {Path(audio_file).name}")

    processor = WhisperProcessor.from_pretrained(model_dir, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    audio, sr = load_audio(audio_file)
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
                max_length=400,
                num_beams=5,
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0
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

    speech_segments = find_speech_segments(audio, sr)
    print(f"Found {len(speech_segments)} speech segments")

    if len(speech_segments) < 2:
        return add_single_speaker(transcription)

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
        return add_single_speaker(transcription)

    embeddings = np.array(embeddings)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    speaker_labels = kmeans.fit_predict(embeddings_scaled)

    speaker_labels = smooth_speaker_transitions(speaker_labels, window_size=3)

    return map_speakers_to_transcription(transcription, valid_segments, speaker_labels)


def smooth_speaker_transitions(labels, window_size=3):
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


def map_speakers_to_transcription(transcription: str, segments: List[Tuple[float, float]], speaker_labels) -> str:
    print("Mapping speakers to transcription...")

    sentences = split_into_sentences(transcription)
    if len(sentences) <= 1:
        return f"[Speaker_1]: {transcription}"

    total_chars = sum(len(s) for s in sentences)
    total_duration = segments[-1][1] - segments[0][0] if segments else 60

    result: List[str] = []
    current_time = 0.0
    current_speaker_id = None
    current_speaker_name = None
    current_text: List[str] = []

    for sentence in sentences:
        sentence_duration = (len(sentence) / total_chars) * total_duration
        sentence_time = current_time + sentence_duration / 2

        segment_idx = find_segment_for_time(sentence_time, segments)

        if segment_idx is not None and segment_idx < len(speaker_labels):
            speaker = speaker_labels[segment_idx]
            speaker_name = f"Speaker_{speaker + 1}"

            if current_speaker_id is None or speaker != current_speaker_id:
                if current_text:
                    result.append(f"[{current_speaker_name}]: {' '.join(current_text)}")
                current_speaker_id = speaker
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


def find_segment_for_time(target_time: float, segments: List[Tuple[float, float]]):
    for i, (start, end) in enumerate(segments):
        if start <= target_time <= end:
            return i

    min_distance = float('inf')
    closest_idx = 0
    for i, (start, end) in enumerate(segments):
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


def hybrid_transcribe_audio(audio_file: str, model_dir: str = "whisper-large-full"):
    print(f"HYBRID TRANSCRIPTION: {Path(audio_file).name}")
    print("=" * 60)

    start_time = time.time()

    print("STEP 1: High-Quality Transcription")
    transcription, audio, sr = transcribe_full_audio(audio_file, model_dir)

    print("\nSTEP 2: Speaker Identification")
    speaker_transcription = identify_speakers_in_transcription(transcription, audio, sr)
    speaker_transcription = merge_consecutive_same_speaker(speaker_transcription)

    end_time = time.time()
    processing_time = end_time - start_time

    print("\nHYBRID TRANSCRIPTION COMPLETED")
    print(f"Total processing time: {processing_time:.1f} seconds")
    print(f"Words: {len(transcription.split())} (raw) → {len(' '.join(speaker_transcription.split()) .split())} (speaker-aware)")

    return {
        'raw_transcription': transcription,
        'speaker_transcription': speaker_transcription,
        'processing_time': processing_time,
        'audio_duration': len(audio) / sr
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid transcription with speaker identification")
    parser.add_argument("audio", type=str, help="Path to input .wav file")
    parser.add_argument("--model_dir", type=str, default="whisper-large-full", help="Path to fine-tuned Whisper model")
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


