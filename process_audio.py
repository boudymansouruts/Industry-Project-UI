#!/usr/bin/env python3
"""
Simple CLI function to process the specific audio file
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, continue without it

# Import configuration
try:
    from config_local import get_huggingface_token
    HUGGINGFACE_TOKEN = get_huggingface_token()
except ImportError:
    from config import HUGGINGFACE_TOKEN

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import pyannote for speaker diarization
from pyannote.audio import Pipeline

def perform_speaker_diarization(audio_file, num_speakers=2):
    """
    Perform speaker diarization on an audio file using pyannote
    
    Args:
        audio_file (str): Path to the audio file
        num_speakers (int): Expected number of speakers (default: 2)
    
    Returns:
        tuple: (diarization_result, rttm_file_path)
    """
    print(f"🎭 Starting speaker diarization for: {audio_file}")
    
    try:
        # Working method: Soundfile conversion
        import soundfile as sf
        import numpy as np
        
        print("🔄 Converting audio format for diarization...")
        
        # Read the entire file in chunks to handle non-seekable files
        with sf.SoundFile(audio_file) as f:
            # Get file info
            sr_original = f.samplerate
            frames = f.frames
            
            # Read all frames at once
            audio_data = f.read(frames=frames, dtype='float32')
            
            # Resample to 16kHz if needed
            if sr_original != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr_original, target_sr=16000)
                sr = 16000
            else:
                sr = sr_original
        
        # Create a temporary converted file
        temp_file = "temp_diarization_audio.wav"
        sf.write(temp_file, audio_data, sr, format='WAV', subtype='PCM_16')
        
        # Instantiate the pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_TOKEN
        )
        
        # Run the pipeline on the converted audio file
        diarization = pipeline(temp_file, num_speakers=num_speakers)
        
        # Clean up temporary file
        os.remove(temp_file)
        
        # Create output directory for diarization results
        os.makedirs("results", exist_ok=True)
        
        # Generate RTTM filename based on input file
        audio_filename = Path(audio_file).stem
        rttm_file = f"results/{audio_filename}_diarization.rttm"
        
        # Dump the diarization output to disk using RTTM format
        with open(rttm_file, "w") as rttm:
            diarization.write_rttm(rttm)
        
        print(f"✅ Diarization completed! Results saved to: {rttm_file}")
        
        # Print diarization summary
        print("\n🎭 SPEAKER DIARIZATION SUMMARY:")
        print("-" * 40)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_name = f"Speaker {int(speaker.split('_')[1]) + 1}"
            print(f"{speaker_name}: {turn.start:.1f}s - {turn.end:.1f}s")
        
        return diarization, rttm_file
        
    except Exception as e:
        print(f"❌ Error during diarization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e  # Re-raise the exception instead of returning None

def process_audio_with_speaker_chunks(audio_file, diarization_result):
    """
    Process audio using speaker diarization results as chunks
    
    Args:
        audio_file (str): Path to the audio file
        diarization_result: Pyannote diarization result
    
    Returns:
        dict: Processing results with speaker-aware chunks
    """
    print("🎯 Processing audio with speaker-aware chunks...")
    
    try:
        import librosa
        import soundfile as sf
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch
        
        # Load audio using soundfile (same method as diarization)
        
        with sf.SoundFile(audio_file) as f:
            sr_original = f.samplerate
            frames = f.frames
            audio_data = f.read(frames=frames, dtype='float32')
            
            # Resample to 16kHz if needed
            if sr_original != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr_original, target_sr=16000)
                sr = 16000
            else:
                sr = sr_original
        
        # Load Whisper model
        model_dir = "models/whisper-base-finetuned"
        processor = WhisperProcessor.from_pretrained(model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        
        speaker_chunks = []
        total_duration = len(audio_data) / sr
        
        print(f"📊 Processing {len(list(diarization_result.itertracks(yield_label=True)))} speaker segments...")
        
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            # Extract audio segment
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)
            segment_audio = audio_data[start_sample:end_sample]
            
            # Transcribe segment
            inputs = processor(segment_audio, sampling_rate=sr, return_tensors="pt")
            
            with torch.no_grad():
                predicted_ids = model.generate(inputs["input_features"])
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Store chunk info
            speaker_name = f"Speaker {int(speaker.split('_')[1]) + 1}"
            chunk_info = {
                'speaker': speaker_name,
                'start_time': turn.start,
                'end_time': turn.end,
                'duration': turn.end - turn.start,
                'text': transcription.strip(),
                'word_count': len(transcription.split())
            }
            
            speaker_chunks.append(chunk_info)
            print(f"  {speaker_name}: {turn.start:.1f}s-{turn.end:.1f}s ({chunk_info['word_count']} words)")
        
        return {
            'audio_file': audio_file,
            'total_duration': total_duration,
            'speaker_chunks': speaker_chunks,
            'total_speakers': len(set(chunk['speaker'] for chunk in speaker_chunks)),
            'total_words': sum(chunk['word_count'] for chunk in speaker_chunks)
        }
        
    except Exception as e:
        print(f"❌ Error in speaker-aware processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e  # Re-raise the exception instead of returning None

def process_specific_file():
    """Process the specific audio file mentioned by the user"""
    
    # The specific file path
    audio_file = "uploads/20251002_133634_093407-i-837-61455560954.wav"
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file not found: {audio_file}")
        print("Available files in uploads:")
        uploads_dir = Path("uploads")
        if uploads_dir.exists():
            for file in uploads_dir.glob("*.wav"):
                print(f"   • {file}")
        return False
    
    print(f"🎤 Processing: {audio_file}")
    print("=" * 60)
    
    try:
        # First, perform speaker diarization
        print("\n🎭 STEP 1: Speaker Diarization")
        print("-" * 30)
        diarization_result, rttm_file = perform_speaker_diarization(audio_file, num_speakers=2)
        
        # Use speaker-aware processing
        print("\n🎯 STEP 2: Speaker-Aware Transcription")
        print("-" * 30)
        speaker_result = process_audio_with_speaker_chunks(audio_file, diarization_result)
        
        # Create a result object compatible with the existing code
        class SpeakerAwareResult:
            def __init__(self, speaker_data, diarization_result):
                self.audio_file = speaker_data['audio_file']
                self.audio_duration = speaker_data['total_duration']
                self.processing_time = 0  # Will be calculated
                self.total_chunks = len(speaker_data['speaker_chunks'])
                self.high_risk_chunks = []
                self.moderate_risk_chunks = []
                self.risk_summary = {'overall_risk_level': 'LOW'}
                self.raw_transcription = self._create_raw_transcription(speaker_data['speaker_chunks'])
                self.transcript_chunks = speaker_data['speaker_chunks']
                self.overall_raw_sentiment = self._analyze_full_transcript_sentiment(speaker_data['speaker_chunks'])
                self.diarization_result = diarization_result
                self.speaker_sentiments = self._analyze_speaker_sentiments(speaker_data['speaker_chunks'])
            
            def _create_raw_transcription(self, chunks):
                return " ".join(chunk['text'] for chunk in chunks)
            
            def _analyze_full_transcript_sentiment(self, chunks):
                """Comprehensive sentiment analysis using multiple approaches"""
                try:
                    import re
                    
                    # Combine all text for full context analysis
                    full_text = " ".join(chunk['text'] for chunk in chunks)
                    
                    # Clean text for better analysis
                    cleaned_text = re.sub(r'\[.*?\]', '', full_text)  # Remove [BLANK_AUDIO], [inaudible], etc.
                    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                    
                    print(f"🔍 Analyzing sentiment on {len(cleaned_text)} characters...")
                    
                    # OPTION 1: Full Transcript Analysis (Better Context)
                    print("📊 Option 1: Full Transcript Analysis")
                    full_sentiment = self._analyze_with_simple_model(cleaned_text, "Full Transcript")
                    
                    # OPTION 2: Chunk-based Analysis (Individual Segments)
                    print("📊 Option 2: Chunk-based Analysis")
                    chunk_sentiments = []
                    for chunk in chunks:
                        chunk_text = re.sub(r'\[.*?\]', '', chunk['text']).strip()
                        if chunk_text and len(chunk_text) > 3:  # Skip very short chunks
                            sentiment = self._analyze_with_simple_model(chunk_text, f"Chunk {chunk['start_time']:.1f}s")
                            chunk_sentiments.append(sentiment)
                    
                    # OPTION 3: Speaker-aware Analysis (Per Speaker)
                    print("📊 Option 3: Speaker-aware Analysis")
                    speaker_sentiments = self._analyze_speaker_sentiments_simple(chunks)
                    
                    # Combine results for final decision
                    combined_result = self._combine_sentiment_results(
                        full_sentiment, chunk_sentiments, speaker_sentiments
                    )
                    
                    return combined_result
                    
                except Exception as e:
                    print(f"⚠️  Sentiment analysis failed: {str(e)}")
                    return {'predicted_emotion': 'neutral', 'confidence': 0.5}
            
            def _analyze_with_simple_model(self, text, context=""):
                """Analyze sentiment using the fine-tuned model"""
                try:
                    from transformers import pipeline
                    import json
                    import re
                    
                    # Clean text
                    clean_text = re.sub(r'\[.*?\]', '', text).strip()
                    if not clean_text or len(clean_text) < 3:
                        return {'predicted_emotion': 'neutral', 'confidence': 0.5, 'context': context}
                    
                    # Truncate text if too long
                    if len(clean_text) > 512:  # DistilBERT has 512 token limit
                        clean_text = clean_text[:512]
                    
                    # Use the fine-tuned model
                    sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="models/sentiment-model/checkpoint-500",
                        return_all_scores=True
                    )
                    
                    # Load label mapping
                    try:
                        with open("models/sentiment-model/label_mapping.json", "r") as f:
                            label_mapping = json.load(f)
                    except:
                        # Fallback mapping
                        label_mapping = {"0": "negative", "1": "neutral", "2": "positive"}
                    
                    results = sentiment_analyzer(clean_text)
                    
                    # Get the highest confidence result
                    best_result = max(results[0], key=lambda x: x['score'])
                    
                    # Map the label to our custom labels
                    predicted_label = label_mapping.get(str(best_result['label']), best_result['label'])
                    
                    return {
                        'predicted_emotion': predicted_label,
                        'confidence': best_result['score'],
                        'all_scores': {label_mapping.get(str(r['label']), r['label']): r['score'] for r in results[0]},
                        'context': context,
                        'text_length': len(clean_text)
                    }
                    
                except Exception as e:
                    print(f"⚠️  Fine-tuned model analysis failed for {context}: {str(e)}")
                    # Fallback to rule-based analysis
                    return self._rule_based_sentiment(text, context)
            
            def _rule_based_sentiment(self, text, context=""):
                """Fallback rule-based sentiment analysis"""
                try:
                    positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 'happy', 'pleased', 'thank', 'thanks', 'yes', 'okay', 'ok', 'fine', 'well', 'better', 'best', 'love', 'like', 'enjoy', 'welcome']
                    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry', 'mad', 'upset', 'sad', 'disappointed', 'frustrated', 'no', 'not', 'never', 'worst', 'wrong', 'problem', 'issue', 'difficult', 'hard', 'trouble']
                    
                    text_lower = text.lower()
                    positive_count = sum(1 for word in positive_words if word in text_lower)
                    negative_count = sum(1 for word in negative_words if word in text_lower)
                    
                    if positive_count > negative_count:
                        emotion = 'positive'
                        confidence = min(0.6 + (positive_count - negative_count) * 0.1, 0.9)
                    elif negative_count > positive_count:
                        emotion = 'negative'
                        confidence = min(0.6 + (negative_count - positive_count) * 0.1, 0.9)
                    else:
                        emotion = 'neutral'
                        confidence = 0.5
                    
                    return {
                        'predicted_emotion': emotion,
                        'confidence': confidence,
                        'all_scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                        'context': context,
                        'text_length': len(text),
                        'method': 'rule_based'
                    }
                    
                except Exception as e:
                    print(f"⚠️  Rule-based analysis failed for {context}: {str(e)}")
                    return {'predicted_emotion': 'neutral', 'confidence': 0.5, 'context': context}
            
            def _combine_sentiment_results(self, full_sentiment, chunk_sentiments, speaker_sentiments):
                """Combine multiple sentiment analysis results for better accuracy"""
                try:
                    # Weight different approaches
                    weights = {
                        'full_transcript': 0.5,    # Full context is most important
                        'chunk_average': 0.3,      # Individual segments
                        'speaker_consensus': 0.2   # Speaker-specific patterns
                    }
                    
                    # Calculate chunk average
                    if chunk_sentiments:
                        chunk_avg = {
                            'positive': sum(s.get('all_scores', {}).get('positive', 0) for s in chunk_sentiments) / len(chunk_sentiments),
                            'negative': sum(s.get('all_scores', {}).get('negative', 0) for s in chunk_sentiments) / len(chunk_sentiments),
                            'neutral': sum(s.get('all_scores', {}).get('neutral', 0) for s in chunk_sentiments) / len(chunk_sentiments)
                        }
                    else:
                        chunk_avg = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                    
                    # Calculate speaker consensus
                    if speaker_sentiments:
                        speaker_avg = {
                            'positive': sum(s.get('all_scores', {}).get('positive', 0) for s in speaker_sentiments.values()) / len(speaker_sentiments),
                            'negative': sum(s.get('all_scores', {}).get('negative', 0) for s in speaker_sentiments.values()) / len(speaker_sentiments),
                            'neutral': sum(s.get('all_scores', {}).get('neutral', 0) for s in speaker_sentiments.values()) / len(speaker_sentiments)
                        }
                    else:
                        speaker_avg = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                    
                    # Combine weighted scores
                    full_scores = full_sentiment.get('all_scores', {})
                    combined_scores = {}
                    
                    for emotion in ['positive', 'negative', 'neutral']:
                        combined_scores[emotion] = (
                            weights['full_transcript'] * full_scores.get(emotion, 0.33) +
                            weights['chunk_average'] * chunk_avg.get(emotion, 0.33) +
                            weights['speaker_consensus'] * speaker_avg.get(emotion, 0.33)
                        )
                    
                    # Get the emotion with highest combined score
                    best_emotion = max(combined_scores, key=combined_scores.get)
                    
                    return {
                        'predicted_emotion': best_emotion,
                        'confidence': combined_scores[best_emotion],
                        'all_scores': combined_scores,
                        'analysis_breakdown': {
                            'full_transcript': full_sentiment,
                            'chunk_average': chunk_avg,
                            'speaker_consensus': speaker_avg,
                            'weights': weights
                        },
                        'method': 'combined_weighted_analysis'
                    }
                    
                except Exception as e:
                    print(f"⚠️  Result combination failed: {str(e)}")
                    return full_sentiment  # Fallback to full transcript result
            
            def _analyze_speaker_sentiments_simple(self, chunks):
                """Analyze sentiment for each speaker separately using simple model"""
                try:
                    from transformers import pipeline
                    import re
                    
                    sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="models/sentiment-model/checkpoint-500",
                        return_all_scores=True
                    )
                    
                    speaker_texts = {}
                    for chunk in chunks:
                        speaker = chunk['speaker']
                        if speaker not in speaker_texts:
                            speaker_texts[speaker] = []
                        # Clean text
                        clean_text = re.sub(r'\[.*?\]', '', chunk['text']).strip()
                        if clean_text and len(clean_text) > 3:
                            speaker_texts[speaker].append(clean_text)
                    
                    speaker_sentiments = {}
                    for speaker, texts in speaker_texts.items():
                        if not texts:
                            continue
                            
                        full_text = " ".join(texts)
                        
                        # Truncate if too long
                        if len(full_text) > 512:
                            full_text = full_text[:512]
                        
                        try:
                            results = sentiment_analyzer(full_text)
                            best_result = max(results[0], key=lambda x: x['score'])
                            
                            speaker_sentiments[speaker] = {
                                'predicted_emotion': best_result['label'].lower(),
                                'confidence': best_result['score'],
                                'all_scores': {r['label'].lower(): r['score'] for r in results[0]},
                                'word_count': sum(len(text.split()) for text in texts),
                                'text_length': len(full_text)
                            }
                        except Exception as e:
                            print(f"⚠️  Speaker {speaker} analysis failed: {str(e)}")
                            # Fallback to rule-based
                            speaker_sentiments[speaker] = self._rule_based_sentiment(full_text, f"Speaker {speaker}")
                    
                    return speaker_sentiments
                    
                except Exception as e:
                    print(f"⚠️  Speaker sentiment analysis failed: {str(e)}")
                    return {}
            
            def to_dict(self):
                return {
                    'audio_file': self.audio_file,
                    'audio_duration': self.audio_duration,
                    'processing_time': self.processing_time,
                    'total_chunks': self.total_chunks,
                    'high_risk_chunks': self.high_risk_chunks,
                    'moderate_risk_chunks': self.moderate_risk_chunks,
                    'risk_summary': self.risk_summary,
                    'raw_transcription': self.raw_transcription,
                    'transcript_chunks': self.transcript_chunks,
                    'overall_raw_sentiment': self.overall_raw_sentiment,
                    'speaker_sentiments': self.speaker_sentiments
                }
        
        result = SpeakerAwareResult(speaker_result, diarization_result)
        
        if result:
            print("\n✅ Processing completed successfully!")
            
            # Save results to JSON
            output_file = "results/audio_analysis_results.json"
            os.makedirs("results", exist_ok=True)
            
            import json
            result_dict = result.to_dict()
            
            # Add diarization information to results if available
            if diarization_result is not None:
                result_dict['diarization'] = {
                    'rttm_file': rttm_file,
                    'speaker_segments': []
                }
                
                # Extract speaker segments for JSON output
                for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                    result_dict['diarization']['speaker_segments'].append({
                        'speaker': speaker,
                        'start_time': turn.start,
                        'end_time': turn.end,
                        'duration': turn.end - turn.start
                    })
            
            with open(output_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            print(f"📊 Results saved to: {output_file}")
            if rttm_file:
                print(f"🎭 Diarization results saved to: {rttm_file}")
            
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
            
            # Print speaker-aware summary
            if hasattr(result, 'diarization_result') and result.diarization_result is not None:
                print(f"\n🎭 SPEAKER-AWARE PROCESSING SUMMARY:")
                print("-" * 40)
                speaker_stats = {}
                for chunk in result.transcript_chunks:
                    speaker = chunk['speaker']
                    if speaker not in speaker_stats:
                        speaker_stats[speaker] = {'segments': 0, 'total_duration': 0, 'words': 0}
                    speaker_stats[speaker]['segments'] += 1
                    speaker_stats[speaker]['total_duration'] += chunk['duration']
                    speaker_stats[speaker]['words'] += chunk['word_count']
                
                for speaker, stats in speaker_stats.items():
                    print(f"Speaker {speaker}: {stats['segments']} segments, {stats['total_duration']:.1f}s, {stats['words']} words")
                    
                    # Show speaker sentiment if available
                    if hasattr(result, 'speaker_sentiments') and speaker in result.speaker_sentiments:
                        sentiment = result.speaker_sentiments[speaker]
                        print(f"  Sentiment: {sentiment['predicted_emotion']} ({sentiment['confidence']:.1%})")
            
            # Print raw transcript and overall sentiment
            if getattr(result, 'raw_transcription', None):
                print("\nRAW TRANSCRIPT:")
                print("-"*30)
                print(result.raw_transcription)
                
                # Print speaker-aware transcript if available
                if hasattr(result, 'transcript_chunks') and result.transcript_chunks:
                    print("\nSPEAKER-AWARE TRANSCRIPT:")
                    print("-"*30)
                    speaker_transcript = ""
                    current_speaker = None
                    
                    for chunk in result.transcript_chunks:
                        if chunk.get('speaker') != current_speaker:
                            if speaker_transcript:
                                speaker_transcript += "\n"
                            speaker_transcript += f"[{chunk['speaker']}] ({chunk['start_time']:.1f}s-{chunk['end_time']:.1f}s): "
                            current_speaker = chunk['speaker']
                        else:
                            speaker_transcript += " "
                        speaker_transcript += chunk['text']
                    
                    print(speaker_transcript)
                
                if getattr(result, 'overall_raw_sentiment', None):
                    ors = result.overall_raw_sentiment
                    print("\n🎭 COMPREHENSIVE SENTIMENT ANALYSIS:")
                    print("-" * 40)
                    
                    # Overall result
                    print(f"🎯 FINAL RESULT: {ors.get('predicted_emotion','unknown').upper()} ({ors.get('confidence',0):.1%})")
                    print(f"📊 Method: {ors.get('method', 'unknown')}")
                    
                    # Show breakdown if available
                    if 'analysis_breakdown' in ors:
                        breakdown = ors['analysis_breakdown']
                        print(f"\n📈 Analysis Breakdown:")
                        print(f"  Full Transcript: {breakdown['full_transcript'].get('predicted_emotion', 'unknown')} ({breakdown['full_transcript'].get('confidence', 0):.1%})")
                        print(f"  Chunk Average: {breakdown['chunk_average']}")
                        print(f"  Speaker Consensus: {breakdown['speaker_consensus']}")
                        print(f"  Weights: {breakdown['weights']}")
                    
                    # Show all emotion scores
                    if 'all_scores' in ors:
                        print(f"\n📊 Emotion Scores:")
                        for emotion, score in ors['all_scores'].items():
                            print(f"  {emotion.capitalize()}: {score:.1%}")
            
            return True
        else:
            print("❌ Processing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎤 Starting audio processing with 2 expected speakers...")
    success = process_specific_file()
    sys.exit(0 if success else 1)
