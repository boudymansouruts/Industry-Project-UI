#!/usr/bin/env python3
"""
Test sentiment analysis on raw vs diarized transcripts
Compare results from full text, chunks, and speaker-level analysis
"""

import json
import re
from pathlib import Path
from transformers import pipeline

# Load the company sentiment model
print("Loading sentiment model...")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="models/company-sentiment",
    return_all_scores=True
)

# Load label mapping
with open("models/company-sentiment/label_mapping.json", "r") as f:
    label_data = json.load(f)
    id2label = label_data["id2label"]

print(f"Model loaded with {len(id2label)} emotion labels\n")


def analyze_text(text, context=""):
    """Analyze sentiment of a text"""
    # Clean text
    clean_text = re.sub(r'\[.*?\]', '', text).strip()
    if not clean_text or len(clean_text) < 3:
        return None
    
    # Truncate if too long
    if len(clean_text) > 512:
        clean_text = clean_text[:512]
    
    try:
        results = sentiment_analyzer(clean_text)
        best_result = max(results[0], key=lambda x: x['score'])
        
        all_scores = {r['label']: r['score'] for r in results[0]}
        
        return {
            'context': context,
            'text_length': len(clean_text),
            'predicted_emotion': best_result['label'],
            'confidence': best_result['score'],
            'all_scores': all_scores
        }
    except Exception as e:
        print(f"Error analyzing {context}: {e}")
        return None


def analyze_in_chunks(text, chunk_size=200):
    """Split text into chunks and analyze each"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i+chunk_size])
        chunks.append(chunk_text)
    
    results = []
    for i, chunk in enumerate(chunks):
        result = analyze_text(chunk, f"Chunk {i+1}/{len(chunks)}")
        if result:
            results.append(result)
    
    return results


def aggregate_chunk_results(chunk_results):
    """Aggregate results from multiple chunks"""
    if not chunk_results:
        return None
    
    # Count emotions
    emotion_counts = {}
    total_confidence = {}
    
    for result in chunk_results:
        emotion = result['predicted_emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        total_confidence[emotion] = total_confidence.get(emotion, 0) + result['confidence']
    
    # Calculate averages
    avg_scores = {}
    for emotion in emotion_counts:
        avg_scores[emotion] = total_confidence[emotion] / emotion_counts[emotion]
    
    # Get most common emotion
    most_common = max(emotion_counts.items(), key=lambda x: x[1])
    
    return {
        'most_common_emotion': most_common[0],
        'occurrences': most_common[1],
        'total_chunks': len(chunk_results),
        'emotion_distribution': emotion_counts,
        'avg_confidences': avg_scores,
        'chunk_results': chunk_results
    }


def split_sentences(text: str):
    """Naive sentence splitter for English text."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip())
    # Split on ., ?, ! while keeping simple cases
    parts = re.split(r"(?<=[\.!?])\s+", text)
    # Fallback: if very long, chunk further
    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > 400:
            # further split long sentences by commas
            subparts = [s.strip() for s in re.split(r",\s+", p) if s.strip()]
            sentences.extend(subparts)
        else:
            sentences.append(p)
    return sentences


def analyze_sentences(text: str, min_confidence: float = 0.0, top_k: int | None = None, target_emotion: str | None = None):
    """Analyze each sentence and return those with highest confidence (triggers)."""
    sentences = split_sentences(text)
    results = []
    for idx, s in enumerate(sentences):
        r = analyze_text(s, f"sentence_{idx+1}")
        if r and r['confidence'] >= min_confidence:
            if target_emotion is not None and r['predicted_emotion'].lower() != target_emotion.lower():
                continue
            results.append({
                'sentence_index': idx + 1,
                'text': s,
                'predicted_emotion': r['predicted_emotion'],
                'confidence': r['confidence'],
                'all_scores': r['all_scores']
            })
    # Sort by confidence desc and limit
    results.sort(key=lambda x: x['confidence'], reverse=True)
    if top_k is None:
        return results
    return results[:top_k]


def print_triggers(title: str, triggers: list):
    print(f"\n{'-'*70}")
    print(title)
    print(f"{'-'*70}")
    if not triggers:
        print("No high-confidence trigger sentences found.")
        return
    for t in triggers:
        print(f"- [{t['predicted_emotion'].upper()} | {t['confidence']:.1%}] {t['text']}")


def parse_diarized_transcript(text):
    """Parse diarized transcript into speaker segments"""
    speakers = {}
    current_speaker = None
    current_text = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a speaker header
        speaker_match = re.match(r'\[(Speaker \d+)\].*:', line)
        if speaker_match:
            # Save previous speaker's text
            if current_speaker and current_text:
                if current_speaker not in speakers:
                    speakers[current_speaker] = []
                speakers[current_speaker].append(' '.join(current_text))
            
            # Start new speaker
            current_speaker = speaker_match.group(1)
            # Get text after the colon
            text_part = line.split(':', 1)[1].strip() if ':' in line else ''
            current_text = [text_part] if text_part else []
        else:
            # Continue previous speaker's text
            if current_speaker:
                current_text.append(line)
    
    # Save last speaker's text
    if current_speaker and current_text:
        if current_speaker not in speakers:
            speakers[current_speaker] = []
        speakers[current_speaker].append(' '.join(current_text))
    
    # Combine all segments for each speaker
    speaker_texts = {}
    for speaker, segments in speakers.items():
        speaker_texts[speaker] = ' '.join(segments)
    
    return speaker_texts


def analyze_by_speaker(speaker_texts):
    """Analyze sentiment for each speaker"""
    results = {}
    
    for speaker, text in speaker_texts.items():
        result = analyze_text(text, f"{speaker}")
        if result:
            results[speaker] = result
    
    return results


def print_results(title, result):
    """Pretty print results"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    if result is None:
        print("No valid result")
        return
    
    if 'predicted_emotion' in result:
        print(f"Predicted Emotion: {result['predicted_emotion'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Text Length: {result.get('text_length', 'N/A')} characters")
        
        if 'all_scores' in result:
            print("\nAll Emotion Scores:")
            sorted_scores = sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)
            for emotion, score in sorted_scores[:5]:  # Top 5
                print(f"  {emotion}: {score:.1%}")
    
    elif 'most_common_emotion' in result:
        print(f"Most Common Emotion: {result['most_common_emotion'].upper()}")
        print(f"Occurrences: {result['occurrences']}/{result['total_chunks']} chunks")
        
        print("\nEmotion Distribution:")
        for emotion, count in sorted(result['emotion_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {count} chunks (avg confidence: {result['avg_confidences'][emotion]:.1%})")


def main():
    # Load transcripts
    raw_path = Path("transcripts/raw_transcript.txt")
    diarized_path = Path("transcripts/diarized_transcript.txt")
    
    if not raw_path.exists() or not diarized_path.exists():
        print("Error: Transcript files not found!")
        return
    
    raw_text = raw_path.read_text(encoding='utf-8')
    diarized_text = diarized_path.read_text(encoding='utf-8')
    
    print(f"\nLoaded transcripts:")
    print(f"  Raw: {len(raw_text)} characters")
    print(f"  Diarized: {len(diarized_text)} characters")
    
    # ===== RAW TRANSCRIPT ANALYSIS =====
    print("\n" + "█"*70)
    print("RAW TRANSCRIPT ANALYSIS")
    print("█"*70)
    
    # 1. Full text analysis
    raw_full = analyze_text(raw_text, "Full Raw Transcript")
    print_results("1. FULL TEXT ANALYSIS (Raw)", raw_full)
    # Triggers for raw (sentence-level)
    top_raw_emotion = (raw_full or {}).get('predicted_emotion', '')
    raw_triggers = analyze_sentences(raw_text, target_emotion=top_raw_emotion, min_confidence=0.0, top_k=None)
    print_triggers(f"Sentence Triggers for Top Emotion (Raw: {top_raw_emotion})", raw_triggers)
    
    # 2. Chunk analysis
    print("\n\nAnalyzing raw transcript in chunks...")
    raw_chunks = analyze_in_chunks(raw_text, chunk_size=150)
    raw_chunk_aggregate = aggregate_chunk_results(raw_chunks)
    print_results("2. CHUNK ANALYSIS (Raw)", raw_chunk_aggregate)
    # Triggers per chunk (top sentence per chunk)
    if raw_chunks:
        print("\nTop Triggers per Raw Chunk:")
        # Use most common emotion across chunks for filtering
        target_chunk_emotion = (raw_chunk_aggregate or {}).get('most_common_emotion', '')
        for i, _ in enumerate(raw_chunks):
            chunk_words = raw_text.split()[i*150:(i+1)*150]
            chunk_text = " ".join(chunk_words)
            chunk_triggers = analyze_sentences(chunk_text, top_k=None, target_emotion=target_chunk_emotion, min_confidence=0.0)
            print_triggers(f"Raw Chunk {i+1} Triggers for {target_chunk_emotion}", chunk_triggers)
    
    # ===== DIARIZED TRANSCRIPT ANALYSIS =====
    print("\n\n" + "█"*70)
    print("DIARIZED TRANSCRIPT ANALYSIS")
    print("█"*70)
    
    # 1. Full text analysis (treating diarized as plain text)
    diarized_full = analyze_text(diarized_text, "Full Diarized Transcript")
    print_results("1. FULL TEXT ANALYSIS (Diarized)", diarized_full)
    top_diarized_emotion = (diarized_full or {}).get('predicted_emotion', '')
    diarized_triggers = analyze_sentences(diarized_text, target_emotion=top_diarized_emotion, min_confidence=0.0, top_k=None)
    print_triggers(f"Sentence Triggers for Top Emotion (Diarized: {top_diarized_emotion})", diarized_triggers)
    
    # 2. Chunk analysis
    print("\n\nAnalyzing diarized transcript in chunks...")
    diarized_chunks = analyze_in_chunks(diarized_text, chunk_size=150)
    diarized_chunk_aggregate = aggregate_chunk_results(diarized_chunks)
    print_results("2. CHUNK ANALYSIS (Diarized)", diarized_chunk_aggregate)
    if diarized_chunks:
        print("\nTop Triggers per Diarized Chunk:")
        target_diar_chunk_emotion = (diarized_chunk_aggregate or {}).get('most_common_emotion', '')
        for i, _ in enumerate(diarized_chunks):
            chunk_words = diarized_text.split()[i*150:(i+1)*150]
            chunk_text = " ".join(chunk_words)
            chunk_triggers = analyze_sentences(chunk_text, top_k=None, target_emotion=target_diar_chunk_emotion, min_confidence=0.0)
            print_triggers(f"Diarized Chunk {i+1} Triggers for {target_diar_chunk_emotion}", chunk_triggers)
    
    # 3. By speaker analysis
    print("\n\nAnalyzing by speaker...")
    speaker_texts = parse_diarized_transcript(diarized_text)
    speaker_results = analyze_by_speaker(speaker_texts)
    
    print(f"\n{'='*70}")
    print("3. BY SPEAKER ANALYSIS (Diarized)")
    print(f"{'='*70}")
    
    for speaker, result in speaker_results.items():
        print(f"\n{speaker}:")
        print(f"  Emotion: {result['predicted_emotion'].upper()}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Text Length: {result['text_length']} characters")
        print(f"  Top 3 Emotions:")
        sorted_scores = sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)
        for emotion, score in sorted_scores[:3]:
            print(f"    {emotion}: {score:.1%}")
        # Triggers per speaker for their top emotion
        speaker_texts = parse_diarized_transcript(diarized_text)
        speaker_top = result.get('predicted_emotion', '')
        triggers = analyze_sentences(speaker_texts.get(speaker, ""), top_k=None, target_emotion=speaker_top, min_confidence=0.0)
        print_triggers(f"{speaker} Triggers for {speaker_top}", triggers)
    
    # ===== COMPARISON =====
    print("\n\n" + "█"*70)
    print("COMPARISON SUMMARY")
    print("█"*70)
    
    print("\nFULL TEXT ANALYSIS:")
    print(f"  Raw:      {raw_full['predicted_emotion'].upper()} ({raw_full['confidence']:.1%})")
    print(f"  Diarized: {diarized_full['predicted_emotion'].upper()} ({diarized_full['confidence']:.1%})")
    
    print("\nCHUNK ANALYSIS (Most Common):")
    print(f"  Raw:      {raw_chunk_aggregate['most_common_emotion'].upper()} ({raw_chunk_aggregate['occurrences']}/{raw_chunk_aggregate['total_chunks']} chunks)")
    print(f"  Diarized: {diarized_chunk_aggregate['most_common_emotion'].upper()} ({diarized_chunk_aggregate['occurrences']}/{diarized_chunk_aggregate['total_chunks']} chunks)")
    
    print("\nSPEAKER ANALYSIS:")
    for speaker, result in speaker_results.items():
        print(f"  {speaker}: {result['predicted_emotion'].upper()} ({result['confidence']:.1%})")
    
    # Do not save results per user request
    pass


if __name__ == "__main__":
    main()

