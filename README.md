# DailyTalk Whisper Fine-tuning Pipeline

This repository contains a complete pipeline for fine-tuning OpenAI's Whisper model on the DailyTalk conversational dataset for improved speech-to-text transcription with speaker diarization.

## 🎯 Overview

The system fine-tunes Whisper (a state-of-the-art speech recognition model) on the DailyTalk dataset to improve transcription accuracy for conversational speech. It includes advanced speaker diarization to separate different speakers in audio recordings.

## 🏗️ Architecture

- **Base Model**: OpenAI Whisper-base (290M parameters)
- **Fine-tuning**: Custom training on DailyTalk conversational data
- **Input**: Raw audio waveforms (16kHz)
- **Output**: High-quality transcribed text with speaker separation
- **Speaker Diarization**: MFCC-based clustering with advanced audio features

## 📁 Project Structure

```
Industry Project/
├── 📄 preprocess.py                       # Data preprocessing script
├── 📄 finetune.py                         # Whisper fine-tuning script  
├── 📄 test.py                             # Speaker diarization + transcription
├── 📄 transcribe_with_speakers.py         # Speaker-separated transcription
├── 📄 transcribe_full.py                  # Full audio transcription
├── 📄 test_finetuned_model.py             # Model testing utilities
├── 📄 requirements.txt                    # Python dependencies
├── 📄 README.md                           # This file
├── 📁 dailytalk/                          # Original DailyTalk dataset
│   ├── 📁 data/                          # Audio files and transcripts
│   │   ├── 📁 0/, 1/, 10/, 100/, ...     # Dialog directories
│   │   │   ├── 📄 *.wav                  # Audio files
│   │   │   └── 📄 *.txt                  # Transcript files
│   └── 📄 metadata.json                  # Dataset metadata
├── 📁 whisper-dailytalk-3k/              # Fine-tuned Whisper model
│   ├── 📄 model.safetensors              # Model weights
│   ├── 📄 config.json                    # Model configuration
│   └── 📄 ... (other model files)
└── 📁 preprocessed_whisper/               # Preprocessed data splits
    ├── 📄 train.json                     # Training split
    ├── 📄 val.json                       # Validation split
    ├── 📄 test.json                      # Test split
    └── 📄 metadata.json                  # Processing metadata
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing

```bash
# Preprocess DailyTalk dataset (adjust sample size as needed)
python preprocess.py --max_samples 3000

# This will create:
# - preprocessed_whisper/ directory with train/val/test splits
# - Normalized audio files at 16kHz
# - Metadata with duration and text statistics
```

### 3. Fine-tune Whisper Model

```bash
# Fine-tune Whisper on DailyTalk dataset
python finetune.py

# This will:
# - Load 3000 samples from DailyTalk
# - Fine-tune Whisper-base for 3 epochs
# - Save the fine-tuned model to whisper-dailytalk-3k/
# - Test the model on sample audio files
```

### 4. Test the Fine-tuned Model

```bash
# Test with speaker diarization
python transcribe_with_speakers.py

# Test full transcription only
python transcribe_full.py

# Compare original vs fine-tuned models
python test_finetuned_model.py compare
```

## 🎤 Usage Examples

### Basic Transcription
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import librosa

# Load fine-tuned model
processor = WhisperProcessor.from_pretrained("whisper-dailytalk-3k")
model = WhisperForConditionalGeneration.from_pretrained("whisper-dailytalk-3k")

# Load and preprocess audio
audio, sr = sf.read("your_audio.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# Transcribe
input_features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
predicted_ids = model.generate(input_features)
transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
```

### Speaker Diarization
```bash
# Transcribe with automatic speaker separation
python transcribe_with_speakers.py

# Output format:
# [0.0s - 30.0s] Speaker_1: Hello. Hi, Shelley. How are you?...
# [30.0s - 34.2s] Speaker_2: Thank you so much. Bye.
```

## 📊 Model Performance

### Training Results
- **Dataset Size**: 3000+ samples from DailyTalk
- **Training Loss**: Reduced from 5.22 → 0.46 over 3 epochs
- **Accuracy**: Significant improvement on conversational speech
- **Specialization**: Optimized for dialogue and casual conversation

### Before vs After Fine-tuning
| Original Whisper | Fine-tuned Whisper |
|------------------|---------------------|
| "I'm fearing an all of my budget" | "I'm figuring out my budget" ✅ |
| Basic punctuation | Enhanced punctuation ✅ |
| Generic speech model | Conversational specialist ✅ |

## 🔧 Configuration

### Preprocessing Settings
- **Sample Rate**: 16kHz
- **Audio Format**: Mono, normalized
- **Text Processing**: UTF-8, cleaned transcripts
- **Dataset Splits**: 80% train, 10% validation, 10% test

### Fine-tuning Parameters
- **Base Model**: openai/whisper-base
- **Batch Size**: 4 (effective 16 with gradient accumulation)
- **Learning Rate**: 1e-5
- **Epochs**: 3
- **Language**: English
- **Task**: Transcription

### Speaker Diarization Features
- **MFCC Features**: 13 coefficients + deltas
- **Spectral Features**: Centroid, rolloff, zero-crossing rate
- **Clustering**: K-means with feature standardization
- **Post-processing**: Segment merging and noise filtering

## 📈 Evaluation Metrics

The system evaluates performance using:
- **Word Error Rate (WER)**: Measures transcription accuracy
- **Speaker Accuracy**: Correct speaker assignment rate
- **Conversation Flow**: Natural dialogue preservation
- **Processing Speed**: Real-time transcription capability

## 🛠️ Utilities

### Available Scripts

1. **`preprocess.py`** - Data preprocessing and splitting
2. **`finetune.py`** - Whisper model fine-tuning
3. **`test.py`** - Advanced speaker diarization testing
4. **`transcribe_with_speakers.py`** - Speaker-separated transcription
5. **`transcribe_full.py`** - Full audio transcription
6. **`test_finetuned_model.py`** - Model comparison and testing

### Command Line Usage

```bash
# Preprocess with custom sample size
python preprocess.py --max_samples 5000 --output_dir custom_preprocessed

# Fine-tune with specific configuration
python finetune.py  # Uses default 3000 samples

# Test on specific audio file
python transcribe_with_speakers.py  # Edit script to change audio file
```

## 📋 Requirements

### System Requirements
- **Python**: 3.8+
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for models and data
- **CPU**: Multi-core recommended for faster processing

### Dependencies
- **torch**: Deep learning framework
- **transformers**: Hugging Face transformers library
- **librosa**: Audio processing
- **soundfile**: Audio I/O
- **scikit-learn**: Machine learning utilities
- **faster-whisper**: Optimized Whisper inference
- **numpy, scipy**: Numerical computing

## 🎯 Use Cases

### Primary Applications
1. **Call Center Transcription**: Automatic conversation logging
2. **Meeting Transcription**: Multi-speaker meeting notes
3. **Interview Processing**: Journalist and researcher tools
4. **Accessibility**: Real-time conversation subtitles
5. **Content Creation**: Podcast and video transcription

### Specialized Features
- **Conversational Context**: Trained on natural dialogue
- **Speaker Identification**: Automatic speaker separation
- **High Accuracy**: Fine-tuned for casual conversation patterns
- **Real-time Processing**: Efficient inference pipeline

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors:**
```bash
# Ensure model directory exists
ls whisper-dailytalk-3k/
# If missing, run fine-tuning again
python finetune.py
```

**Audio Processing Issues:**
```bash
# Check audio file format
python -c "import soundfile as sf; print(sf.info('your_audio.wav'))"
```

**Memory Issues:**
```bash
# Reduce batch size in finetune.py
# Change: per_device_train_batch_size=4 to per_device_train_batch_size=2
```

**Poor Transcription Quality:**
```bash
# Increase training samples
python preprocess.py --max_samples 5000
python finetune.py  # Will use more data automatically
```

## 📚 References

- **Whisper Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **DailyTalk Dataset**: Conversational speech dataset for dialogue systems
- **Transformers Library**: [Hugging Face Transformers](https://github.com/huggingface/transformers)
- **Speaker Diarization**: MFCC-based clustering techniques

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with sample audio files
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please respect the licenses of the underlying models and datasets.

---

**Note**: This pipeline is specifically optimized for conversational speech patterns found in the DailyTalk dataset. For other domains, consider retraining with domain-specific data.