# Risk-Focused Audio Analysis Pipeline

A comprehensive audio analysis system that transcribes speech, identifies speakers, and detects emotional risk indicators in conversations.

## Features

- **High-Quality Transcription**: Uses OpenAI's Whisper Large model for accurate speech-to-text conversion
- **Speaker Identification**: Advanced speaker diarization to identify different speakers
- **Risk Detection**: Analyzes emotional content to identify HIGH and MODERATE risk indicators
- **Web Interface**: User-friendly Flask web application for easy interaction
- **Real-time Processing**: Live progress tracking during audio analysis

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
   git clone https://github.com/yourusername/risk-audio-analysis.git
   cd risk-audio-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python start_risk_server.py
   ```

4. **Access the web interface**:
   Open your browser and go to `http://localhost:5000`

## Usage

### Web Interface
1. Upload an audio file (WAV, MP3, MP4, etc.)
2. Wait for processing to complete
3. View transcription with speaker identification
4. Review risk analysis results
5. Download results as JSON

### Command Line
```bash
# Process a single audio file
python hybrid_transcribe.py audio_file.wav

# Process with custom model
python hybrid_transcribe.py audio_file.wav --model_dir openai/whisper-large-v2
```

## Architecture

### Core Components

- **`hybrid_transcribe.py`**: Main transcription and speaker diarization logic
- **`transcription_chunk_risk_pipeline.py`**: Orchestrates the complete analysis pipeline
- **`risk_web_app.py`**: Flask web application
- **`Emotion_Recognition/`**: Emotion analysis models and utilities

### Processing Pipeline

1. **Audio Loading**: Load and normalize audio files
2. **Transcription**: Convert speech to text using Whisper Large
3. **Speaker Diarization**: Identify different speakers
4. **Chunking**: Segment text by speaker for analysis
5. **Emotion Analysis**: Analyze each chunk for emotional content
6. **Risk Assessment**: Identify HIGH and MODERATE risk indicators
7. **Results**: Generate comprehensive analysis report

## Model Information

- **Transcription Model**: `openai/whisper-large-v2` (No fine-tuning)
- **Emotion Model**: Custom BioBERT-based model trained on emotional datasets
- **Speaker Diarization**: Voice-based clustering with temporal smoothing

## File Structure

```
risk-audio-analysis/
├── hybrid_transcribe.py              # Core transcription logic
├── transcription_chunk_risk_pipeline.py  # Main pipeline
├── risk_web_app.py                   # Flask web app
├── start_risk_server.py             # Server startup script
├── requirements.txt                  # Python dependencies
├── templates/                        # HTML templates
├── static/                          # CSS/JS assets
├── uploads/                         # Uploaded audio files
├── results/                         # Analysis results
└── Emotion_Recognition/             # Emotion analysis module
    ├── inference.py                 # Emotion inference
    ├── models/                      # Trained models
    └── ...
```

## API Endpoints

- `GET /`: Main upload interface
- `POST /upload`: Upload audio file
- `GET /processing/<id>`: Processing status
- `GET /progress/<id>`: Real-time progress
- `GET /results/<id>`: Analysis results
- `GET /transcript/<id>`: Full transcript

## Performance

- **Processing Time**: ~2-5 minutes for 5-minute audio (depending on hardware)
- **Accuracy**: High-quality transcription with Whisper Large
- **Scalability**: Designed for both local and cloud deployment

## Deployment Options

### Local Deployment
```bash
python start_risk_server.py
```

### Cloud Deployment
- **AWS SageMaker**: Use the provided SageMaker deployment scripts
- **Docker**: Containerize the application
- **Heroku**: Deploy as a web service

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