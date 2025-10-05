"""
Configuration file for BioBERT Health Risk Detection Pipeline
This file contains all hyperparameters, paths, and settings for the project
"""

import torch
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "cache"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model Configuration
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
MAX_LENGTH = 128  # Reduced from 256 for CPU efficiency
NUM_CLASSES = 12  # Total number of emotion categories

# Comprehensive Emotion Categories
EMOTION_CATEGORIES = {
    # Negative Health Risk Emotions
    "depression": ["sadness", "hopelessness", "despair", "grief", "emptiness", "melancholy", 
                   "sorrow", "misery", "gloom", "dejection", "despondent"],
    "anxiety": ["fear", "nervousness", "worry", "panic", "restlessness", "apprehension",
                "uneasiness", "dread", "trepidation", "agitation", "scared", "anxious"],
    "stress": ["overwhelm", "pressure", "frustration", "burnout", "tension", "strain",
               "exhaustion", "distress", "burden", "stressed", "overwhelmed"],
    "anger": ["rage", "irritation", "hostility", "resentment", "aggression", "fury",
              "wrath", "indignation", "annoyance", "mad", "angry", "furious"],
    "loneliness": ["isolation", "abandonment", "withdrawal", "disconnection", "solitude",
                   "alienation", "lonely", "alone", "isolated", "excluded"],
    "confusion": ["disorientation", "uncertainty", "bewilderment", "perplexity", "puzzled",
                  "confused", "baffled", "unclear", "lost", "uncertain"],
    "physical_pain": ["pain", "fatigue", "exhausted", "tired", "ache", "hurt", "sore",
                      "discomfort", "suffering", "unwell", "sick", "ill"],
    "shame_guilt": ["shame", "guilt", "embarrassment", "regret", "remorse", "humiliation",
                    "ashamed", "guilty", "embarrassed", "mortified", "worthless"],
    
    # Positive/Neutral Emotions
    "happiness": ["joy", "contentment", "delight", "satisfaction", "euphoria", "bliss",
                  "cheerful", "happy", "glad", "pleased", "elated", "joyful"],
    "love_affection": ["love", "warmth", "caring", "compassion", "tenderness", "attachment",
                       "affection", "fondness", "adoration", "devoted", "loving"],
    "excitement": ["enthusiasm", "anticipation", "energy", "thrill", "motivation", "eager",
                   "excited", "energized", "pumped", "inspired", "passionate"],
    "calm_neutral": ["peace", "tranquility", "balanced", "stable", "neutral", "calm",
                     "serene", "relaxed", "composed", "centered", "okay", "fine"]
}

# Create reverse mapping for emotion detection
EMOTION_MAPPING = {}
for category, keywords in EMOTION_CATEGORIES.items():
    for keyword in keywords:
        EMOTION_MAPPING[keyword.lower()] = category

# Class labels
CLASS_LABELS = list(EMOTION_CATEGORIES.keys())
LABEL_TO_ID = {label: idx for idx, label in enumerate(CLASS_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# Training Configuration (CPU-optimized)
BATCH_SIZE = 8  # Reduced from 16 for CPU
LEARNING_RATE = 3e-5  # Slightly higher for faster convergence
NUM_EPOCHS = 5  # Reduced from 10 to save time
WARMUP_STEPS = 200  # Reduced from 500
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.3
GRADIENT_ACCUMULATION_STEPS = 4  # Increased to maintain effective batch size
MAX_GRAD_NORM = 1.0

# Data Configuration
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SAMPLES_PER_CLASS = 800  # Reduced from 1000 for faster training
RANDOM_SEED = 42
MIN_TEXT_LENGTH = 10  # Minimum text length in characters
MAX_TEXT_LENGTH = 1000  # Maximum text length in characters

# Data Augmentation Configuration
AUGMENTATION_ENABLED = True
AUGMENTATION_FACTOR = 0.2  # Reduced from 0.3 for faster processing
AUGMENTATION_TECHNIQUES = ["synonym_replacement", "random_insertion", "paraphrase"]

# Training Settings
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_DELTA = 0.001
CHECKPOINT_SAVE_STEPS = 200  # More frequent saves for CPU training
LOGGING_STEPS = 25  # More frequent logging
EVALUATION_STEPS = 50  # More frequent evaluation
SAVE_TOTAL_LIMIT = 3

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
FP16_TRAINING = True if torch.cuda.is_available() else False

# Optimization Configuration
USE_SCHEDULER = True
SCHEDULER_TYPE = "linear"  # Options: "linear", "cosine", "polynomial"
ADAM_EPSILON = 1e-8
ADAM_BETAS = (0.9, 0.999)

# Evaluation Configuration
METRICS_TO_COMPUTE = ["accuracy", "precision", "recall", "f1", "roc_auc"]
CONFIDENCE_THRESHOLD = 0.5
SAVE_PREDICTIONS = True
SAVE_ATTENTION_WEIGHTS = True

# Cross-Validation Configuration
CROSS_VALIDATION_ENABLED = False
CV_FOLDS = 5

# Logging Configuration
LOG_LEVEL = "INFO"
TENSORBOARD_ENABLED = True
WANDB_ENABLED = False  # Set to True if using Weights & Biases
WANDB_PROJECT = "biobert-health-risk-detection"
WANDB_ENTITY = None  # Set your W&B entity if using

# Model Checkpoint Settings
BEST_MODEL_PATH = MODELS_DIR / "best_model"
FINAL_MODEL_PATH = MODELS_DIR / "final_model"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

# Results and Outputs
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
ERROR_ANALYSIS_PATH = RESULTS_DIR / "error_analysis.csv"

# Data Paths
RAW_DATA_PATH = DATA_DIR / "dailytalk.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
VAL_DATA_PATH = DATA_DIR / "val.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

# Preprocessing Configuration
TEXT_CLEANING_OPTIONS = {
    "lowercase": False,  # Keep original casing for BioBERT
    "remove_urls": True,
    "remove_emails": True,
    "remove_phone_numbers": True,
    "remove_special_chars": False,  # Keep punctuation for context
    "remove_extra_whitespace": True,
    "expand_contractions": True,
    "remove_html_tags": True,
    "remove_emoji": False,  # Keep emojis as they convey emotion
    "normalize_whitespace": True
}

# Inference Configuration
INFERENCE_BATCH_SIZE = 32
RETURN_ALL_SCORES = True
TOP_K_PREDICTIONS = 3

print(f"Configuration loaded successfully!")
print(f"Device: {DEVICE}")
print(f"Number of GPUs: {N_GPU}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Model: {MODEL_NAME}")
