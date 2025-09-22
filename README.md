Whisper Large Training (DailyTalk)

Overview
This project trains OpenAI Whisper Large v2 on the DailyTalk dataset. Audio preprocessing (resample to 16 kHz, normalize, trim) happens on-the-fly inside the training pipeline; no preprocessed files are written to disk.

Project layout
- preprocess.py: Audio utilities and dataset loading from DailyTalk
- train_whisper_large_full.py: Full training loop and evaluation
- diarize.py: Basic diarization (segment and cluster speakers) utility
- dailytalk/data/: DailyTalk dataset root (speaker subfolders with .wav and .txt)

Setup
1) Python 3.10+ recommended
2) Install dependencies:
   pip install -r requirements.txt

Dataset
- Place DailyTalk under dailytalk/data (e.g., dailytalk/data/0, dailytalk/data/1, ...)
- Each speaker folder should contain paired .wav and .txt files with matching names.

Run training
python train_whisper_large_full.py

What happens
- Loads Whisper Large v2 and moves it to CUDA if available
- Builds a dataset from dailytalk/data
- Preprocesses audio per batch (load -> 16 kHz -> normalize -> trim)
- Trains with gradient accumulation, LR scheduler, and early stopping
- Prints periodic progress and sample transcriptions

Outputs
- Model and tokenizer are saved to whisper-large-full/
- Training configuration is saved to whisper-large-full/training_config.json

Speaker diarization
Two options are available:

1) Robust diarization (recommended) using pyannote.audio
   - Requires a Hugging Face token for best performance: set HF_TOKEN environment variable.
   - Install dependencies: pip install -r requirements.txt
   - Single file:
     python diarize_pyannote.py file path/to/audio.wav --out output.json
   - Directory:
     python diarize_pyannote.py dir dailytalk/data diarization_outputs

2) Lightweight baseline (no external models)
   - Single file:
     python diarize.py file path/to/audio.wav --out output.json
   - Directory:
     python diarize.py dir dailytalk/data --out_dir diarization_outputs



