# Risk-Focused Audio Analysis Pipeline

A comprehensive CLI-based audio analysis system that transcribes speech, identifies speakers, and detects emotional risk indicators in conversations.

## Features

- **High-Quality Transcription**: Uses OpenAI's Whisper Large model for accurate speech-to-text conversion
- **Speaker Identification**: Advanced speaker diarization to identify different speakers
- **Risk Detection**: Analyzes emotional content to identify HIGH and MODERATE risk indicators
- **CLI Interface**: Simple command-line interface for easy processing
- **JSON Output**: Structured results for integration with other systems

## Risk Categories

### 🚨 HIGH RISK
- Depression
- Anxiety  
- Loneliness
- Physical Pain

### ⚠️ MODERATE RISK
- Stress
- Anger
- Confusion
- Shame/Guilt

### ✅ LOW RISK (Ignored)
- Happiness
- Calm
- Excitement
- Other positive emotions

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/boudymansouruts/Industry-Project-UI.git
   cd Industry-Project-UI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Process Specific Audio File
```bash
# Process the default audio file
python process_audio.py
```

### Process Any Audio File
```bash
# Process any audio file
python cli_audio_processor.py path/to/audio.wav

# Process with custom output directory
python cli_audio_processor.py path/to/audio.wav --output results/

# Process with verbose logging
python cli_audio_processor.py path/to/audio.wav --verbose
```

### Direct Pipeline Usage
```bash
# Use the main pipeline directly
python transcription_chunk_risk_pipeline.py audio_file.wav --output results.json
```

## Architecture

### Core Components

- **`hybrid_transcribe.py`**: Main transcription and speaker diarization logic
- **`transcription_chunk_risk_pipeline.py`**: Orchestrates the complete analysis pipeline
- **`process_audio.py`**: Simple CLI script for processing specific audio files
- **`cli_audio_processor.py`**: General-purpose CLI processor
- **`Emotion_Recognition/`**: Emotion analysis models and utilities

### Processing Pipeline

1. **Audio Loading**: Load and normalize audio files
2. **Transcription**: Convert speech to text using Whisper Large
3. **Speaker Diarization**: Identify different speakers
4. **Chunking**: Segment text by speaker for analysis
5. **Emotion Analysis**: Analyze each chunk for emotional content
6. **Risk Assessment**: Identify HIGH and MODERATE risk indicators
7. **Results**: Generate comprehensive analysis report

## Model Configuration

The system uses `model_config.json` to manage different Whisper models:

```json
{
  "current_model": "whisper-large",
  "models": {
    "whisper-base": {
      "path": "openai/whisper-base",
      "description": "Whisper Base - Fast CPU processing"
    },
    "whisper-large": {
      "path": "openai/whisper-large-v2", 
      "description": "Whisper Large - Better accuracy, slower"
    }
  }
}
```

## File Structure

```
Industry-Project-UI/
├── hybrid_transcribe.py              # Core transcription logic
├── transcription_chunk_risk_pipeline.py  # Main pipeline
├── process_audio.py                  # Simple CLI processor
├── cli_audio_processor.py            # General CLI processor
├── model_config.json                 # Model configuration
├── requirements.txt                  # Python dependencies
├── uploads/                          # Sample audio files
├── results/                          # Analysis results
└── Emotion_Recognition/              # Emotion analysis module
    ├── inference.py                  # Emotion inference
    ├── models/                       # Trained models
    └── ...
```

## Output Format

Results are saved as JSON files with the following structure:

```json
{
  "audio_file": "path/to/audio.wav",
  "audio_duration": 203.5,
  "processing_time": 45.2,
  "total_chunks": 15,
  "high_risk_chunks": [...],
  "moderate_risk_chunks": [...],
  "risk_summary": {
    "overall_risk_level": "MODERATE",
    "high_risk_count": 2,
    "moderate_risk_count": 5
  }
}
```

## Performance

- **Processing Time**: ~5-10 minutes for 5-minute audio on CPU (Whisper Large)
- **Accuracy**: High-quality transcription with Whisper Large
- **Memory Usage**: ~4-6GB RAM recommended for Whisper Large

## Requirements

- Python 3.8+
- 8GB+ RAM recommended
- GPU recommended for faster processing
- Audio files in WAV, MP3, MP4, or other supported formats

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

## Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for the Transformers library
- The emotion recognition research community