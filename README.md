# 🎤 Audio Analysis & Risk Detection System

A comprehensive audio analysis pipeline powered by Whisper Large, BioBERT, and PyAnnote for transcription, emotion recognition, and mental health risk assessment.

## Features

- **🎙️ High-Quality Transcription** - Using OpenAI's Whisper Large model
- **👥 Speaker Diarization** - Advanced speaker identification with PyAnnote
- **😊 Emotion Recognition** - BioBERT-based emotion analysis
- **⚠️ Risk Detection** - Automatic identification of HIGH and MODERATE risk indicators
- **🌐 Web Interface** - Clean, modern static HTML UI
- **📊 Detailed Reports** - Comprehensive analysis with visualizations

## Risk Categories

### 🚨 HIGH RISK
- Depression
- Anxiety
- Loneliness

### ⚠️ MODERATE RISK
- Stress
- Anger
- Confusion
- Physical Pain
- Shame/Guilt

### ✅ LOW RISK
- Happiness
- Love/Affection
- Excitement
- Calm/Neutral

## Environment Setup

### Required Environment Variables

**Option 1: Using .env file (Recommended)**

1. Copy the example file:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` and add your token:
   ```bash
   HUGGINGFACE_TOKEN=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
   ```

**Option 2: Environment Variables**

Set the environment variable directly:

```bash
# Windows (PowerShell)
$env:HUGGINGFACE_TOKEN="your_token_here"

# Windows (Command Prompt)
set HUGGINGFACE_TOKEN=your_token_here

# Linux/Mac
export HUGGINGFACE_TOKEN=your_token_here
```

### GitHub Secrets (for CI/CD)

If using GitHub Actions, add the following secret in your repository settings:
- **Secret Name**: `HUGGINGFACE_TOKEN`
- **Secret Value**: Your Hugging Face token

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/boudymansouruts/Industry-Project-UI.git
cd Industry-Project-UI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Required Models
The models will be automatically downloaded on first use.

## Usage

### Option 1: Interactive Python Script (Recommended for SageMaker)

**Perfect for environments without web browser access!**

```bash
# Interactive mode - select from available files
python analyze_audio_interactive.py

# Analyze a specific file
python analyze_audio_interactive.py --file uploads/your-audio.wav

# List available files
python analyze_audio_interactive.py --list
```

This runs directly in your terminal/Jupyter and displays results immediately - no web server or browser needed!

### Option 2: Flask Web UI

1. **Start the Flask server:**
```bash
python app.py
```

2. **Access the UI:**
   - **Local**: Open `http://localhost:5000` in your browser
   - **SageMaker**: Open `http://[instance-ip]:5000` or use port forwarding

3. **Upload and Analyze:**
   - Click the upload area or drag & drop an audio file
   - Supported formats: WAV, MP3, M4A, FLAC
   - Click "Analyze Audio" to process
   - View results including transcription, emotion analysis, and risk assessment

### Option 3: Command Line Processing

```bash
python process_audio.py
```

### Fine-tuning Whisper

To fine-tune the Whisper model on your own data:

```bash
python train_whisper_finetune.py
```

## Project Structure

```
Industry-Project-UI/
├── app.py                              # Flask web server
├── templates/
│   └── index.html                      # Main UI
├── static/
│   ├── css/style.css                   # Styling
│   └── js/app.js                       # JavaScript logic
├── transcription_chunk_risk_pipeline.py # Main pipeline
├── hybrid_transcribe.py                # Whisper transcription
├── inference.py                        # Emotion recognition
├── config.py                           # Configuration
├── process_audio.py                    # CLI tool
├── train_whisper_finetune.py          # Model training
├── model_config.json                   # Model configuration
├── dailytalk/                          # Training dataset
├── uploads/                            # User uploads
└── models/                             # Trained models
```

## API Endpoints

### `POST /api/analyze`
Analyze an uploaded audio file.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `audio` file

**Response:**
```json
{
  "success": true,
  "result": {
    "audio_duration": 45.2,
    "total_chunks": 12,
    "high_risk_chunks": [...],
    "moderate_risk_chunks": [...],
    "raw_transcription": "...",
    "risk_summary": {...}
  }
}
```

### `GET /api/files`
List available audio files.

### `GET /api/status`
Check API status.

## Configuration

Edit `model_config.json` to switch between models:

```json
{
  "current_model": "whisper-large",
  "models": {
    "whisper-large": {
      "path": "openai/whisper-large-v2",
      "description": "Whisper Large - Better accuracy"
    }
  }
}
```

## Models Used

- **Transcription**: OpenAI Whisper Large v2
- **Speaker Diarization**: PyAnnote Audio
- **Emotion Recognition**: BioBERT (dmis-lab/biobert-base-cased-v1.2)

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 10GB+ disk space for models

## Troubleshooting

### Out of Memory Errors
- Use Whisper Base instead of Whisper Large
- Reduce audio length
- Enable gradient checkpointing in training

### Port Already in Use
```bash
# Kill existing process
pkill -f "python app.py"

# Or use a different port
python app.py --port 5001
```

### Models Not Downloading
Check internet connection and Hugging Face access. Models are cached in `~/.cache/huggingface/`.

## License

This project is for educational and research purposes.

## Acknowledgments

- OpenAI Whisper
- Hugging Face Transformers
- PyAnnote Audio
- BioBERT

## Support

For issues and questions, please open an issue on GitHub.
