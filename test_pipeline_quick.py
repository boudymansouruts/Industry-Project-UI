#!/usr/bin/env python3
from transcription_chunk_risk_pipeline import TranscriptionChunkRiskPipeline

print('Initializing pipeline...')
pipe = TranscriptionChunkRiskPipeline()

print('Processing audio...')
result = pipe.process_audio('uploads/084140-i-800-61418885308.wav')

chunks = result.transcript_chunks
speakers = set(c['speaker'] for c in chunks)
print(f'\nSuccess! Found {len(chunks)} chunks')
print(f'Speakers detected: {speakers}')

overall = result.overall_raw_sentiment
print(f'Overall sentiment: {overall.get("predicted_emotion")} ({overall.get("confidence", 0):.2%})')
print('Test complete.')


