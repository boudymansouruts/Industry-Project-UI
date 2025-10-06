"""
Inference Module for BioBERT Health Risk Detection
Handles predictions on new text inputs with confidence scoring
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from typing import Dict, List, Union, Optional, Tuple
import logging
from pathlib import Path
import json
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# Import configuration
from config import *
import torch.nn as nn

# Define BioBERTClassifier inline
class BioBERTClassifier(nn.Module):
    """BioBERT-based classifier for emotion detection"""
    def __init__(self, num_classes, model_name=MODEL_NAME, dropout_rate=DROPOUT_RATE):
        super(BioBERTClassifier, self).__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Define TextPreprocessor inline
class TextPreprocessor:
    """ASR-aware text preprocessor for conversational transcripts"""
    def __init__(self):
        import re
        self._re = re
        # patterns to remove common ASR artifacts
        self.bracket_noise = self._re.compile(r"\[(?:inaudible|noise|music|laughter|silence|blank_audio)\]", self._re.I)
        self.multispace = self._re.compile(r"\s+")
        self.dup_char = self._re.compile(r"\b(\w{2,})\s+\1\b", self._re.I)  # dedupe repeated words
    
    def preprocess(self, text: str) -> str:
        """Clean ASR artifacts, normalize whitespace, keep content."""
        if not text or not isinstance(text, str):
            return ""
        t = text.strip()
        t = self.bracket_noise.sub("", t)
        # normalize common filler tokens
        t = t.replace(" uh ", " ").replace(" um ", " ")
        # de-duplicate immediate repeated words
        t = self.dup_char.sub(r"\1", t)
        # collapse whitespace
        t = self.multispace.sub(" ", t)
        return t.strip()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    Data class for prediction results
    """
    text: str
    predicted_emotion: str
    confidence: float
    risk_level: str
    all_probabilities: Dict[str, float]
    top_k_predictions: List[Tuple[str, float]]
    attention_weights: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'predicted_emotion': self.predicted_emotion,
            'confidence': self.confidence,
            'risk_level': self.risk_level,
            'all_probabilities': self.all_probabilities,
            'top_k_predictions': self.top_k_predictions
        }
    
    def __str__(self) -> str:
        """String representation"""
        return (
            f"Emotion: {self.predicted_emotion} (Confidence: {self.confidence:.2%})\n"
            f"Risk Level: {self.risk_level}\n"
            f"Top 3: {', '.join([f'{e}: {p:.1%}' for e, p in self.top_k_predictions[:3]])}"
        )


class HealthRiskPredictor:
    """
    Main inference class for health risk detection
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to use for inference
        """
        self.device = device or DEVICE
        # Explicitly default to the latest trained checkpoint
        # e.g., C:\...\Emotion_Recognition\models\checkpoints\best_model.pt
        self.model_path = model_path or CHECKPOINT_DIR / "best_model.pt"
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.preprocessor = TextPreprocessor()
        
        # Risk level mappings
        self.risk_levels = {
            'depression': 'HIGH',
            'anxiety': 'HIGH',
            'stress': 'MODERATE',
            'anger': 'MODERATE',
            'loneliness': 'HIGH',
            'confusion': 'MODERATE',
            'physical_pain': 'HIGH',
            'shame_guilt': 'MODERATE',
            'happiness': 'LOW',
            'love_affection': 'LOW',
            'excitement': 'LOW',
            'calm_neutral': 'LOW',
            # Company-specific labels
            'frustration': 'MODERATE',
            'client_wants_to_leave': 'HIGH',
            'risk_issue': 'HIGH',
            'anger': 'MODERATE',
            'urgency': 'MODERATE',
            'escalation': 'MODERATE'
        }
        
        # Load model and tokenizer
        self.load_model()
        
    def load_model(self):
        """Load trained model and tokenizer"""
        # Resolve model path (prefer explicit path, then checkpoints/best_model.pt, then BEST_MODEL_PATH)
        candidate_paths = []
        if self.model_path is not None:
            candidate_paths.append(Path(self.model_path))
        try:
            # Prefer checkpoints/best_model.pt if present
            candidate_paths.append(CHECKPOINT_DIR / "best_model.pt")
        except NameError:
            pass
        # Also try BEST_MODEL_PATH/best_model.pt
        candidate_paths.append(BEST_MODEL_PATH / "best_model.pt")

        resolved_path = next((p for p in candidate_paths if p and Path(p).exists()), None)
        if resolved_path is None:
            resolved_path = candidate_paths[0]

        self.model_path = Path(resolved_path)

        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR
        )

        # Initialize plain BioBERT classifier (skip company-specific model)
        self.model = BioBERTClassifier(
            model_name=MODEL_NAME,
            num_classes=NUM_CLASSES,
            dropout_rate=0.0  # No dropout during inference
        )
        
        # Load checkpoint if exists
        if self.model_path.exists():
            # PyTorch 2.6+ defaults to weights_only=True which can block older checkpoints.
            # Our checkpoints are locally trained and trusted, so explicitly allow full load.
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model checkpoint not found at {self.model_path}")
            logger.warning("Using untrained model for demonstration")
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        return self.preprocessor.preprocess(text)
    
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize preprocessed text
        
        Args:
            text: Preprocessed text
            
        Returns:
            Tokenized inputs
        """
        # Plain fixed-length tokenization for stable inference
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        return {
            'input_ids': input_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device)
        }
    
    def predict_single(
        self,
        text: str,
        return_attention: bool = False,
        top_k: int = TOP_K_PREDICTIONS
    ) -> PredictionResult:
        """
        Make prediction for single text
        
        Args:
            text: Input text
            return_attention: Whether to return attention weights
            top_k: Number of top predictions to return
            
        Returns:
            Prediction result
        """
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        if not cleaned_text:
            logger.warning("Empty text after preprocessing")
            return self._empty_prediction(text)
        
        # Tokenize
        inputs = self.tokenize_text(cleaned_text)
        
        # Plain single-pass prediction
        with torch.no_grad():
            logits = self.model(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            probabilities = F.softmax(logits, dim=1).squeeze(0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
        
        # Map to emotion label
        predicted_emotion = ID_TO_LABEL[predicted_idx]

        # Confidence floor to neutral
        if confidence < 0.55:
            predicted_emotion = 'calm_neutral'
            confidence = float(confidence)
        risk_level = self.risk_levels[predicted_emotion]
        
        # Get all probabilities
        all_probs = {
            ID_TO_LABEL[i]: float(probabilities[i])
            for i in range(NUM_CLASSES)
        }
        
        # Get top-k predictions
        top_k_indices = torch.topk(probabilities, min(top_k, NUM_CLASSES)).indices
        top_k_predictions = [
            (ID_TO_LABEL[idx.item()], float(probabilities[idx]))
            for idx in top_k_indices
        ]
        
        # Get attention weights if requested
        attention_weights = None
        if return_attention:
            # For now, return None since we don't have attention weights
            attention_weights = None
        
        return PredictionResult(
            text=text[:200] + '...' if len(text) > 200 else text,
            predicted_emotion=predicted_emotion,
            confidence=confidence,
            risk_level=risk_level,
            all_probabilities=all_probs,
            top_k_predictions=top_k_predictions,
            attention_weights=attention_weights
        )

    # Rule-based overrides removed for plain model usage
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = INFERENCE_BATCH_SIZE,
        show_progress: bool = True
    ) -> List[PredictionResult]:
        """
        Make predictions for batch of texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for inference
            show_progress: Whether to show progress bar
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        from tqdm import tqdm
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Preprocess batch
            cleaned_texts = [self.preprocess_text(text) for text in batch_texts]
            
            # Skip empty texts
            valid_indices = [j for j, text in enumerate(cleaned_texts) if text]
            valid_texts = [cleaned_texts[j] for j in valid_indices]
            
            if not valid_texts:
                # Add empty predictions for all texts in batch
                for text in batch_texts:
                    results.append(self._empty_prediction(text))
                continue
            
            # Tokenize batch
            encodings = self.tokenizer(
                valid_texts,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Make predictions
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probabilities = F.softmax(logits, dim=1)
            
            # Process results
            result_idx = 0
            for j, original_text in enumerate(batch_texts):
                if j in valid_indices:
                    probs = probabilities[result_idx]
                    predicted_idx = torch.argmax(probs).item()
                    confidence = probs[predicted_idx].item()
                    
                    predicted_emotion = ID_TO_LABEL[predicted_idx]
                    risk_level = self.risk_levels[predicted_emotion]
                    
                    all_probs = {
                        ID_TO_LABEL[k]: float(probs[k])
                        for k in range(NUM_CLASSES)
                    }
                    
                    top_k_indices = torch.topk(probs, min(TOP_K_PREDICTIONS, NUM_CLASSES)).indices
                    top_k_predictions = [
                        (ID_TO_LABEL[idx.item()], float(probs[idx]))
                        for idx in top_k_indices
                    ]
                    
                    results.append(PredictionResult(
                        text=original_text[:200] + '...' if len(original_text) > 200 else original_text,
                        predicted_emotion=predicted_emotion,
                        confidence=confidence,
                        risk_level=risk_level,
                        all_probabilities=all_probs,
                        top_k_predictions=top_k_predictions
                    ))
                    
                    result_idx += 1
                else:
                    results.append(self._empty_prediction(original_text))
        
        return results
    
    def predict_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        batch_size: int = INFERENCE_BATCH_SIZE
    ) -> pd.DataFrame:
        """
        Make predictions for dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            batch_size: Batch size for inference
            
        Returns:
            Dataframe with predictions
        """
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts, batch_size)
        
        # Add predictions to dataframe
        df['predicted_emotion'] = [p.predicted_emotion for p in predictions]
        df['confidence'] = [p.confidence for p in predictions]
        df['risk_level'] = [p.risk_level for p in predictions]
        
        # Add top-3 predictions
        for i in range(3):
            df[f'top_{i+1}_emotion'] = [
                p.top_k_predictions[i][0] if len(p.top_k_predictions) > i else None
                for p in predictions
            ]
            df[f'top_{i+1}_confidence'] = [
                p.top_k_predictions[i][1] if len(p.top_k_predictions) > i else 0.0
                for p in predictions
            ]
        
        return df
    
    def analyze_conversation(
        self,
        messages: List[str],
        speaker_ids: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze emotional trajectory in conversation
        
        Args:
            messages: List of conversation messages
            speaker_ids: Optional list of speaker IDs
            
        Returns:
            Conversation analysis results
        """
        # Make predictions for all messages
        predictions = self.predict_batch(messages)
        
        # Analyze emotional trajectory
        emotions = [p.predicted_emotion for p in predictions]
        confidences = [p.confidence for p in predictions]
        risk_levels = [p.risk_level for p in predictions]
        
        # Calculate statistics
        emotion_counts = pd.Series(emotions).value_counts().to_dict()
        risk_counts = pd.Series(risk_levels).value_counts().to_dict()
        
        # Detect emotional shifts
        shifts = []
        for i in range(1, len(emotions)):
            if emotions[i] != emotions[i-1]:
                shifts.append({
                    'position': i,
                    'from': emotions[i-1],
                    'to': emotions[i],
                    'risk_change': f"{risk_levels[i-1]} -> {risk_levels[i]}"
                })
        
        # Calculate overall risk
        high_risk_ratio = risk_levels.count('HIGH') / len(risk_levels)
        if high_risk_ratio > 0.5:
            overall_risk = 'HIGH'
        elif high_risk_ratio > 0.25:
            overall_risk = 'MODERATE'
        else:
            overall_risk = 'LOW'
        
        analysis = {
            'total_messages': len(messages),
            'emotion_distribution': emotion_counts,
            'risk_distribution': risk_counts,
            'overall_risk': overall_risk,
            'high_risk_ratio': high_risk_ratio,
            'average_confidence': np.mean(confidences),
            'emotional_shifts': shifts,
            'dominant_emotion': max(emotion_counts, key=emotion_counts.get),
            'message_predictions': [p.to_dict() for p in predictions]
        }
        
        # Add speaker analysis if provided
        if speaker_ids:
            speaker_emotions = {}
            for speaker, pred in zip(speaker_ids, predictions):
                if speaker not in speaker_emotions:
                    speaker_emotions[speaker] = []
                speaker_emotions[speaker].append(pred.predicted_emotion)
            
            analysis['speaker_analysis'] = {
                speaker: pd.Series(emotions).value_counts().to_dict()
                for speaker, emotions in speaker_emotions.items()
            }
        
        return analysis
    
    def _empty_prediction(self, text: str) -> PredictionResult:
        """
        Return empty prediction for invalid text
        
        Args:
            text: Original text
            
        Returns:
            Empty prediction result
        """
        return PredictionResult(
            text=text[:200] + '...' if len(text) > 200 else text,
            predicted_emotion='calm_neutral',
            confidence=0.0,
            risk_level='LOW',
            all_probabilities={label: 0.0 for label in CLASS_LABELS},
            top_k_predictions=[('calm_neutral', 0.0)]
        )
    
    def save_model(self, path: Path):
        """
        Save model checkpoint
        
        Args:
            path: Path to save model
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")


def interactive_prediction():
    """
    Interactive command-line prediction interface
    """
    print("\n" + "="*60)
    print("BioBERT Health Risk Detection - Interactive Mode")
    print("="*60)
    print("Type 'quit' to exit\n")
    
    # Initialize predictor
    predictor = HealthRiskPredictor()
    
    while True:
        # Get user input
        text = input("\nEnter text to analyze: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            print("Please enter some text to analyze")
            continue
        
        # Make prediction
        result = predictor.predict_single(text, top_k=5)
        
        # Display results
        print("\n" + "-"*40)
        print("ANALYSIS RESULTS:")
        print("-"*40)
        print(f"Predicted Emotion: {result.predicted_emotion.replace('_', ' ').title()}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Risk Level: {result.risk_level}")
        print("\nTop 5 Predictions:")
        for i, (emotion, prob) in enumerate(result.top_k_predictions[:5], 1):
            print(f"  {i}. {emotion.replace('_', ' ').title()}: {prob:.1%}")
        
        # Show risk warning if needed
        if result.risk_level == 'HIGH':
            print("\n⚠️  HIGH RISK DETECTED")
            print("This text indicates potential mental health concerns.")
            print("Professional support may be beneficial.")


if __name__ == "__main__":
    # Run interactive mode
    interactive_prediction()
