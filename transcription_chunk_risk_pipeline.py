#!/usr/bin/env python3
"""
Risk-Focused Audio Transcription and Emotion Recognition Pipeline
Uses actual transcription chunks from Whisper processing for emotion analysis
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import transcription components
from hybrid_transcribe import transcribe_full_audio
from pyannote.audio import Pipeline as PyAnnotePipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
from config import HUGGINGFACE_TOKEN

# Import emotion recognition components with robust path handling (works in SageMaker & local)
import sys
from pathlib import Path

# Try to import directly; if it fails, add likely paths and retry
try:
    from inference import HealthRiskPredictor, PredictionResult
    from config import *  # noqa: F401,F403
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent
    candidates = [
        project_root,
        Path.cwd(),
    ]
    for c in candidates:
        c_str = str(c)
        if c_str not in sys.path:
            sys.path.append(c_str)
    try:
        from inference import HealthRiskPredictor, PredictionResult
        from config import *  # noqa: F401,F403
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Could not import emotion recognition modules. Ensure inference.py and config.py are in the project root."
        ) from e

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
if not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
    logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger(__name__)


@dataclass
class RiskChunk:
    """Represents a transcription chunk with risk analysis"""
    speaker: str
    text: str
    start_time: float
    end_time: float
    word_count: int
    emotion: str
    confidence: float
    risk_level: str
    all_probabilities: Dict[str, float]
    top_k_predictions: List[Tuple[str, float]]
    chunk_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskFocusedResult:
    """Complete risk-focused analysis result"""
    audio_file: str
    audio_duration: float
    processing_time: float
    total_chunks: int
    high_risk_chunks: List[RiskChunk]
    moderate_risk_chunks: List[RiskChunk]
    risk_summary: Dict[str, Any]
    transcript_chunks: List[Dict[str, Any]]
    raw_transcription: str
    overall_raw_sentiment: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'audio_file': self.audio_file,
            'audio_duration': self.audio_duration,
            'processing_time': self.processing_time,
            'total_chunks': self.total_chunks,
            'high_risk_chunks': [chunk.to_dict() for chunk in self.high_risk_chunks],
            'moderate_risk_chunks': [chunk.to_dict() for chunk in self.moderate_risk_chunks],
            'risk_summary': self.risk_summary,
            'transcript_chunks': self.transcript_chunks,
            'raw_transcription': self.raw_transcription,
            'overall_raw_sentiment': self.overall_raw_sentiment
        }


class TranscriptionChunkRiskPipeline:
    """
    Risk-focused pipeline that uses actual transcription chunks from Whisper processing
    """
    
    def __init__(self):
        """Initialize the pipeline"""
        self.emotion_predictor = None
        self._init_emotion_predictor()
        
        # Risk level mapping
        self.risk_levels = {
            'depression': 'HIGH', 'anxiety': 'HIGH', 'stress': 'MODERATE', 'anger': 'MODERATE',
            'loneliness': 'HIGH', 'confusion': 'MODERATE', 'physical_pain': 'HIGH', 'shame_guilt': 'MODERATE',
            'happiness': 'LOW', 'love_affection': 'LOW', 'excitement': 'LOW', 'calm_neutral': 'LOW'
        }
        
        logger.info("TranscriptionChunkRiskPipeline initialized")
    
    def _init_emotion_predictor(self):
        """Initialize the emotion recognition model"""
        try:
            self.emotion_predictor = HealthRiskPredictor()
            logger.info("Emotion predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize emotion predictor: {e}")
            raise
    
    def process_audio(self, audio_file: str, progress_callback: Optional[callable] = None) -> RiskFocusedResult:
        """
        Process audio file with risk-focused analysis using transcription chunks
        
        Args:
            audio_file: Path to audio file
            progress_callback: Optional callback for progress updates
            
        Returns:
            RiskFocusedResult with risk analysis
        """
        start_time = time.time()
        
        if progress_callback:
            progress_callback(5, "Starting risk-focused analysis...")
        
        logger.info(f"Processing audio file: {audio_file}")
        
        # Step 1: Get transcription chunks from Whisper processing
        if progress_callback:
            progress_callback(10, "Transcribing audio and extracting chunks...")
        
        transcription_chunks = self._get_transcription_chunks(audio_file)
        # Keep raw transcription string for debugging/visibility
        raw_transcription = " ".join(chunk['text'] for chunk in transcription_chunks)
        
        if progress_callback:
            progress_callback(30, f"Found {len(transcription_chunks)} transcription chunks")
        
        # Step 2: Analyze each chunk for risk
        if progress_callback:
            progress_callback(40, "Analyzing chunks for risk...")
        
        risk_chunks = self._analyze_chunks_for_risk(transcription_chunks, progress_callback)
        
        # Step 3: Categorize by risk level
        high_risk_chunks = [chunk for chunk in risk_chunks if chunk.risk_level == 'HIGH']
        moderate_risk_chunks = [chunk for chunk in risk_chunks if chunk.risk_level == 'MODERATE']
        
        if progress_callback:
            progress_callback(90, "Generating risk summary...")
        
        # Step 4: Generate risk summary
        risk_summary = self._generate_risk_summary(
            risk_chunks, high_risk_chunks, moderate_risk_chunks
        )

        # Overall sentiment on the full raw transcript (speaker-agnostic) + evidence
        try:
            overall_pred = self.emotion_predictor.predict_single(raw_transcription)
            # Select top evidence chunks by the predicted emotion probability from chunk-level predictions
            evidence = self._extract_evidence_chunks(overall_pred.predicted_emotion, risk_chunks, top_k=5)
            overall_raw_sentiment = {
                'predicted_emotion': overall_pred.predicted_emotion,
                'confidence': overall_pred.confidence,
                'top_k_predictions': overall_pred.top_k_predictions,
                'evidence_chunks': evidence,
            }
        except Exception as e:
            logger.error(f"Failed overall sentiment on raw transcript: {e}")
            overall_raw_sentiment = {
                'predicted_emotion': 'calm_neutral',
                'confidence': 0.0,
                'top_k_predictions': [],
                'evidence_chunks': []
            }

        # Step 5: Attach risk annotations back onto transcript chunks for UI
        # Default every chunk to LOW unless elevated risk exists
        index_to_risk = {c.chunk_index: c for c in (high_risk_chunks + moderate_risk_chunks)}
        enriched_transcription_chunks = []
        for chunk in transcription_chunks:
            idx = chunk.get('chunk_index', 0)
            risk_chunk = index_to_risk.get(idx)
            if risk_chunk is not None:
                # Elevated risk
                chunk_with_risk = {
                    **chunk,
                    'emotion': risk_chunk.emotion,
                    'confidence': risk_chunk.confidence,
                    'risk_level': risk_chunk.risk_level,
                    'risk_level_badge': 'danger' if risk_chunk.risk_level == 'HIGH' else 'warning',
                    'risk_level_class': 'risk-high' if risk_chunk.risk_level == 'HIGH' else 'risk-moderate',
                }
            else:
                # Low/default risk
                chunk_with_risk = {
                    **chunk,
                    'emotion': 'calm_neutral',
                    'confidence': 0.0,
                    'risk_level': 'LOW',
                    'risk_level_badge': 'success',
                    'risk_level_class': 'risk-low',
                }
            enriched_transcription_chunks.append(chunk_with_risk)
        
        processing_time = time.time() - start_time
        
        if progress_callback:
            progress_callback(100, "Risk analysis completed!")
        
        result = RiskFocusedResult(
            audio_file=audio_file,
            audio_duration=transcription_chunks[0]['audio_duration'] if transcription_chunks else 0.0,
            processing_time=processing_time,
            total_chunks=len(transcription_chunks),
            high_risk_chunks=high_risk_chunks,
            moderate_risk_chunks=moderate_risk_chunks,
            risk_summary=risk_summary,
            transcript_chunks=enriched_transcription_chunks,
            raw_transcription=raw_transcription,
            overall_raw_sentiment=overall_raw_sentiment
        )
        
        logger.info(f"Risk analysis completed: {len(high_risk_chunks)} high risk, {len(moderate_risk_chunks)} moderate risk chunks")
        
        return result

    def _extract_evidence_chunks(self, target_emotion: str, risk_chunks: List[RiskChunk], top_k: int = 5) -> List[Dict[str, Any]]:
        """Pick chunk snippets that best support the overall sentiment.
        Preference is given to chunks whose emotion matches target_emotion,
        sorted by confidence, then include next best confidences if insufficient.
        """
        if not risk_chunks:
            return []

        # First take matching-emotion chunks
        matching = [c for c in risk_chunks if c.emotion == target_emotion]
        matching.sort(key=lambda c: c.confidence, reverse=True)
        selected = matching[:top_k]

        # If not enough, fill with highest confidence remaining
        if len(selected) < top_k:
            remaining = [c for c in risk_chunks if c not in selected]
            remaining.sort(key=lambda c: c.confidence, reverse=True)
            selected.extend(remaining[: max(0, top_k - len(selected))])

        evidence = []
        for c in selected:
            evidence.append({
                'speaker': c.speaker,
                'text': c.text,
                'start_time': c.start_time,
                'end_time': c.end_time,
                'emotion': c.emotion,
                'confidence': c.confidence,
                'chunk_index': c.chunk_index,
            })
        return evidence
    
    def _get_transcription_chunks(self, audio_file: str) -> List[Dict[str, Any]]:
        """
        Use pyannote diarization + Whisper transcription (2 speakers enforced)
        """
        logger.info("Using pyannote diarization + Whisper transcription...")
        
        # Step 1: Load audio
        with sf.SoundFile(audio_file) as f:
            sr_orig = f.samplerate
            frames_total = f.frames
            audio_data = f.read(frames=frames_total, dtype='float32')
            if sr_orig != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr_orig, target_sr=16000)
                sr = 16000
            else:
                sr = sr_orig

        audio_duration = len(audio_data) / sr

        # Step 2: Pyannote diarization (enforce 2 speakers)
        temp_audio_path = "temp_diarization.wav"
        sf.write(temp_audio_path, audio_data, sr, format='WAV', subtype='PCM_16')
        
        diarizer = PyAnnotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_TOKEN
        )
        diarization = diarizer(temp_audio_path, num_speakers=2)
        os.remove(temp_audio_path)

        # Step 3: Load Whisper model
        model_dir = "models/whisper-base-finetuned"
        processor = WhisperProcessor.from_pretrained(model_dir, language="en", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Step 4: Transcribe each diarized segment
        segments = []
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)
            segment_audio = audio_data[start_sample:end_sample]

            inputs = processor(segment_audio, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                predicted_ids = model.generate(inputs["input_features"])
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            if text and len(text) > 1:
                segments.append({
                    'speaker': speaker_label,  # raw: SPEAKER_00 / SPEAKER_01
                    'text': text,
                    'start_time': turn.start,
                    'end_time': turn.end,
                    'word_count': len(text.split())
                })

        transcription_chunks = []
        # Map raw diarizer labels 'SPEAKER_00/01' → 'Speaker 1/2'; leave others as-is
        import re
        speaker_pattern = re.compile(r"^SPEAKER[_\s]?0*(\d+)$", re.IGNORECASE)
        for i, seg in enumerate(segments):
            raw_speaker = (seg.get('speaker') or '').strip()
            speaker_key = raw_speaker.upper()
            friendly = raw_speaker
            m = speaker_pattern.match(speaker_key.replace(' ', '_'))
            if m:
                try:
                    friendly = f"Speaker {int(m.group(1)) + 1}"
                except Exception:
                    friendly = raw_speaker or 'Speaker 1'
            chunk_info = {
                'speaker': friendly,
                'text': seg['text'],
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'word_count': seg['word_count'],
                'audio_duration': audio_duration,
                'chunk_index': i
            }
            transcription_chunks.append(chunk_info)
            logger.info(f"Chunk {i+1}: [{chunk_info['speaker']}] {seg['start_time']:.1f}s-{seg['end_time']:.1f}s, {seg['word_count']} words")

        if not transcription_chunks:
            logger.warning("No chunks returned from diarization; using fallback single chunk")
            full_text, _, _ = transcribe_full_audio(audio_file, "models/whisper-base-finetuned")
            transcription_chunks.append({
                'speaker': 'Speaker 1',
                'text': full_text,
                'start_time': 0.0,
                'end_time': audio_duration,
                'word_count': len(full_text.split()),
                'audio_duration': audio_duration,
                'chunk_index': 0
            })

        logger.info(f"Extracted {len(transcription_chunks)} diarized chunks")
        return transcription_chunks
    
    def _analyze_chunks_for_risk(
        self, 
        transcription_chunks: List[Dict[str, Any]], 
        progress_callback: Optional[callable] = None
    ) -> List[RiskChunk]:
        """Analyze transcription chunks for risk"""
        risk_chunks = []
        
        for i, chunk in enumerate(transcription_chunks):
            if progress_callback and i % 5 == 0:
                progress = 40 + (i / len(transcription_chunks)) * 40
                progress_callback(int(progress), f"Analyzing chunk {i+1}/{len(transcription_chunks)}...")
            
            try:
                # Predict emotion for this chunk
                prediction = self.emotion_predictor.predict_single(chunk['text'])
                
                # Get risk level
                risk_level = self.risk_levels.get(prediction.predicted_emotion, 'LOW')
                
                # Only keep HIGH and MODERATE risk chunks
                if risk_level in ['HIGH', 'MODERATE']:
                    risk_chunk = RiskChunk(
                        speaker=chunk['speaker'],
                        text=chunk['text'],
                        start_time=chunk['start_time'],
                        end_time=chunk['end_time'],
                        word_count=chunk['word_count'],
                        emotion=prediction.predicted_emotion,
                        confidence=prediction.confidence,
                        risk_level=risk_level,
                        all_probabilities=prediction.all_probabilities,
                        top_k_predictions=prediction.top_k_predictions,
                        chunk_index=i
                    )
                    risk_chunks.append(risk_chunk)
                    
                    logger.info(f"Risk detected: [{chunk['speaker']}] {prediction.predicted_emotion} ({risk_level}) - {prediction.confidence:.1%}")
                    logger.info(f"  Text: {chunk['text'][:100]}...")
                
            except Exception as e:
                logger.error(f"Error analyzing chunk {i}: {e}")
                continue
        
        logger.info(f"Found {len(risk_chunks)} risk chunks out of {len(transcription_chunks)} total chunks")
        return risk_chunks
    
    def _generate_risk_summary(
        self, 
        all_chunks: List[RiskChunk], 
        high_risk_chunks: List[RiskChunk], 
        moderate_risk_chunks: List[RiskChunk]
    ) -> Dict[str, Any]:
        """Generate comprehensive risk summary (duration-weighted).
        Percentages are based on total time covered by chunks of each risk level
        divided by the total time covered by all chunks.
        """

        # Compute total durations
        def duration(c: RiskChunk) -> float:
            try:
                return max(0.0, float(c.end_time) - float(c.start_time))
            except Exception:
                return 0.0

        total_duration = sum(duration(c) for c in all_chunks) or 0.0
        high_duration = sum(duration(c) for c in high_risk_chunks)
        moderate_duration = sum(duration(c) for c in moderate_risk_chunks)

        # Duration-weighted percentages
        high_percentage = (high_duration / total_duration * 100.0) if total_duration > 0 else 0.0
        moderate_percentage = (moderate_duration / total_duration * 100.0) if total_duration > 0 else 0.0
        
        # Determine overall risk level
        if high_percentage >= 20:
            overall_risk = 'HIGH'
        elif high_percentage >= 10 or moderate_percentage >= 30:
            overall_risk = 'MODERATE'
        else:
            overall_risk = 'LOW'
        
        # Emotion distribution
        emotion_counts = {}
        for chunk in high_risk_chunks + moderate_risk_chunks:
            emotion_counts[chunk.emotion] = emotion_counts.get(chunk.emotion, 0) + 1
        
        # Speaker risk summary
        speaker_risk = {}
        for chunk in high_risk_chunks + moderate_risk_chunks:
            speaker = chunk.speaker
            if speaker not in speaker_risk:
                speaker_risk[speaker] = {'HIGH': 0, 'MODERATE': 0}
            speaker_risk[speaker][chunk.risk_level] += 1
        
        return {
            'overall_risk_level': overall_risk,
            'high_risk_percentage': high_percentage,
            'moderate_risk_percentage': moderate_percentage,
            'total_risk_chunks': len(high_risk_chunks) + len(moderate_risk_chunks),
            'high_risk_duration_sec': high_duration,
            'moderate_risk_duration_sec': moderate_duration,
            'total_analyzed_duration_sec': total_duration,
            'emotion_distribution': emotion_counts,
            'speaker_risk_summary': speaker_risk,
            'risk_thresholds': {
                'high_risk_threshold': 20,
                'moderate_risk_threshold': 10
            }
        }


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Risk-Focused Audio Analysis using Transcription Chunks")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    pipeline = TranscriptionChunkRiskPipeline()
    
    # Process audio
    result = pipeline.process_audio(args.audio_file)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"Results saved to: {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("RISK-FOCUSED ANALYSIS RESULTS")
    print("="*60)
    print(f"Audio file: {result.audio_file}")
    print(f"Duration: {result.audio_duration:.1f}s")
    print(f"Processing time: {result.processing_time:.1f}s")
    print(f"Total chunks: {result.total_chunks}")
    print(f"High risk chunks: {len(result.high_risk_chunks)}")
    print(f"Moderate risk chunks: {len(result.moderate_risk_chunks)}")
    print(f"Overall risk level: {result.risk_summary['overall_risk_level']}")
    
    # Print raw transcript and overall sentiment
    if result.raw_transcription:
        print("\nRAW TRANSCRIPT (no speakers):")
        print("-"*30)
        print(result.raw_transcription)
        if getattr(result, 'overall_raw_sentiment', None):
            ors = result.overall_raw_sentiment
            print("\nOverall transcript sentiment:")
            print(f"  {ors.get('predicted_emotion','unknown')} ({ors.get('confidence',0):.1%})")
    
    if result.high_risk_chunks:
        print(f"\n🚨 HIGH RISK CHUNKS:")
        for chunk in result.high_risk_chunks:
            print(f"  [{chunk.speaker}] {chunk.emotion} ({chunk.confidence:.1%}) - {chunk.start_time:.1f}s-{chunk.end_time:.1f}s")
            print(f"    Text: {chunk.text[:100]}...")
    
    if result.moderate_risk_chunks:
        print(f"\n⚠️ MODERATE RISK CHUNKS:")
        for chunk in result.moderate_risk_chunks:
            print(f"  [{chunk.speaker}] {chunk.emotion} ({chunk.confidence:.1%}) - {chunk.start_time:.1f}s-{chunk.end_time:.1f}s")
            print(f"    Text: {chunk.text[:100]}...")


if __name__ == "__main__":
    main()
