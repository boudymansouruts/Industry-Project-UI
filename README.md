# Advanced Speaker Identification with Whisper

This project provides high-quality speech transcription with speaker identification using a fine-tuned Whisper model. The system uses a hybrid approach that prioritizes transcription accuracy while maintaining speaker identification capabilities.

## Features

- **Hybrid Transcription Approach**: High-quality transcription first, then speaker identification
- **Generic Speaker Identification**: Identifies speakers as Speaker_1, Speaker_2, etc.
- **Voice Activity Detection**: Automatically detects speech segments for better processing
- **Speaker Clustering**: Uses advanced audio features (MFCC, pitch, spectral) for speaker identification
- **Punctuation Preservation**: Maintains proper punctuation in transcriptions
- **Consecutive Speaker Merging**: Intelligently merges consecutive segments from the same speaker
- **High Accuracy**: Fine-tuned on DailyTalk dataset with improved training parameters

## Files

- `train_model.py` - Train the speaker identification model with enhanced parameters
- `hybrid_transcribe.py` - Hybrid transcription with speaker identification
- `preprocess.py` - Audio preprocessing and speaker analysis utilities
- `requirements.txt` - Required dependencies

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Transcribe Audio with Speaker Identification
```bash
python hybrid_transcribe.py
```

## Training Configuration

The model is trained with enhanced parameters for better quality:
- **10 epochs** (increased from 5)
- **Batch size: 2** (reduced for better gradient updates)
- **Accumulation steps: 8** (increased for stability)
- **292 audio samples** from DailyTalk dataset
- **30 unique speakers** mapped to generic Speaker_1, Speaker_2, etc.

## Hybrid Approach

The system uses a two-stage process:

1. **High-Quality Transcription**: Uses Whisper to transcribe audio in overlapping chunks for maximum accuracy
2. **Speaker Identification**: Applies voice activity detection and speaker clustering to identify speakers
3. **Intelligent Merging**: Combines overlapping transcriptions and merges consecutive same-speaker segments

## Output Format

The system produces transcriptions in the following format:

```
[Speaker_1]: Hello, how are you today?
[Speaker_2]: I'm doing well, thank you for asking.
[Speaker_1]: That's great to hear!
[Speaker_2]: Yes, I'm looking forward to our meeting.
```

## Technical Details

### Voice Activity Detection
- Energy threshold analysis
- Spectral centroid detection
- Zero-crossing rate analysis
- Spectral flux analysis
- Minimum segment length: 0.3 seconds

### Speaker Features
- **MFCC coefficients** (13 features)
- **Pitch analysis** (fundamental frequency)
- **Spectral features** (centroid, rolloff, bandwidth)
- **RMS energy** and **Zero-crossing rate**
- **KMeans clustering** for speaker assignment

### Model Architecture
- **Base Model**: OpenAI Whisper Base
- **Language**: English
- **Task**: Transcribe
- **Processor**: WhisperProcessor with language and task specification

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Librosa
- SoundFile
- NumPy
- Scikit-learn
- SciPy

## Performance

- **Training Loss**: Final average loss of 0.2100
- **Memory Optimized**: Efficient processing for various system configurations
- **High Accuracy**: Improved transcription quality with proper punctuation
- **Speaker Stability**: Temporal smoothing reduces rapid speaker switching

## Notes

- The model works best with clear audio recordings
- Speaker identification is based on comprehensive audio characteristics
- Results are automatically saved with descriptive filenames
- The hybrid approach ensures both transcription accuracy and speaker identification quality

## Model Files

- `whisper-new-model/` - Latest trained model with enhanced parameters
- `whisper-generic-speakers/` - Previous model version (if available)

## Recent Updates

- Enhanced training quality with improved hyperparameters
- Implemented hybrid transcription approach
- Added comprehensive voice activity detection
- Improved speaker clustering with multiple audio features
- Added punctuation preservation and consecutive speaker merging
- Cleaned up codebase and removed debugging files