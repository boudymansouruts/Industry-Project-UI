# Speaker Identification with Whisper

This project provides speaker identification and transcription using a fine-tuned Whisper model that can distinguish between different speakers as Speaker_1, Speaker_2, etc.

## Features

- **Generic Speaker Identification**: Identifies speakers as Speaker_1, Speaker_2, etc.
- **Multi-language Support**: Supports English, Spanish, French, German, Arabic, Tagalog, and Greek
- **High Accuracy**: Fine-tuned on DailyTalk dataset
- **Memory Efficient**: Optimized for 16GB RAM systems

## Files

- `train_model.py` - Train the speaker identification model
- `transcribe_audio.py` - Transcribe audio with speaker identification
- `main.py` - Main script for easy usage
- `requirements.txt` - Required dependencies

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (if not already trained)
```bash
python train_model.py
```

### 3. Transcribe Audio with Speaker Identification
```bash
python main.py "path/to/your/audio.wav"
```

### 4. Save to Specific File
```bash
python main.py "path/to/your/audio.wav" --output "my_transcription.txt"
```

## Example Usage

```bash
# Basic usage
python main.py "C:\Users\boudy\Downloads\091452-i-834-836.wav"

# With custom output file
python main.py "audio.wav" --output "result.txt"

# With custom model
python main.py "audio.wav" --model "my_custom_model"
```

## Output Format

The system produces transcriptions in the following format:

```
[Speaker_1 (EN)]: Hello. How are you today?
[Speaker_2 (EN)]: I'm doing well, thank you for asking.
[Speaker_1 (EN)]: That's great to hear!
```

## Model Training

The model is trained on the DailyTalk dataset with:
- 292 audio samples
- 2 speakers mapped to generic Speaker_1 and Speaker_2
- 5 epochs of training
- Batch size of 8
- Memory optimizations for 16GB RAM

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Librosa
- SoundFile
- NumPy

## Notes

- The model works best with clear audio recordings
- Speaker identification is based on audio characteristics
- Results are saved to `speaker_transcription.txt` by default

