"""
Emotion Mapping Module for BioBERT Health Risk Detection
Maps various emotion labels to 12 comprehensive health-related categories
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter
from fuzzywuzzy import fuzz
from pathlib import Path

# Import configuration
from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmotionMapper:
    """
    Maps raw emotion labels to 12 comprehensive health risk categories
    """
    
    def __init__(self):
        """
        Initialize emotion mapper with predefined mappings
        """
        self.categories = EMOTION_CATEGORIES
        self.mapping = EMOTION_MAPPING
        
        # Extended mappings for common emotion labels in datasets
        self.extended_mappings = {
            # Depression mappings
            "sad": "depression",
            "depressed": "depression",
            "unhappy": "depression",
            "miserable": "depression",
            "gloomy": "depression",
            "downcast": "depression",
            "blue": "depression",
            "down": "depression",
            "dejected": "depression",
            "forlorn": "depression",
            "melancholic": "depression",
            
            # Anxiety mappings
            "anxious": "anxiety",
            "worried": "anxiety",
            "nervous": "anxiety",
            "fearful": "anxiety",
            "afraid": "anxiety",
            "panicked": "anxiety",
            "terrified": "anxiety",
            "uneasy": "anxiety",
            "tense": "anxiety",
            "apprehensive": "anxiety",
            "jittery": "anxiety",
            
            # Stress mappings
            "stressed": "stress",
            "overwhelmed": "stress",
            "pressured": "stress",
            "strained": "stress",
            "burdened": "stress",
            "overloaded": "stress",
            "frazzled": "stress",
            "burnt out": "stress",
            "exhausted": "stress",
            
            # Anger mappings
            "angry": "anger",
            "mad": "anger",
            "furious": "anger",
            "irritated": "anger",
            "annoyed": "anger",
            "enraged": "anger",
            "hostile": "anger",
            "aggressive": "anger",
            "irate": "anger",
            "livid": "anger",
            "outraged": "anger",
            
            # Loneliness mappings
            "lonely": "loneliness",
            "isolated": "loneliness",
            "alone": "loneliness",
            "abandoned": "loneliness",
            "alienated": "loneliness",
            "disconnected": "loneliness",
            "excluded": "loneliness",
            "solitary": "loneliness",
            
            # Confusion mappings
            "confused": "confusion",
            "bewildered": "confusion",
            "puzzled": "confusion",
            "perplexed": "confusion",
            "disoriented": "confusion",
            "uncertain": "confusion",
            "unclear": "confusion",
            "baffled": "confusion",
            "muddled": "confusion",
            
            # Physical pain mappings
            "pain": "physical_pain",
            "hurt": "physical_pain",
            "aching": "physical_pain",
            "sore": "physical_pain",
            "tired": "physical_pain",
            "fatigued": "physical_pain",
            "weak": "physical_pain",
            "sick": "physical_pain",
            "ill": "physical_pain",
            "unwell": "physical_pain",
            
            # Shame/Guilt mappings
            "ashamed": "shame_guilt",
            "guilty": "shame_guilt",
            "embarrassed": "shame_guilt",
            "remorseful": "shame_guilt",
            "regretful": "shame_guilt",
            "humiliated": "shame_guilt",
            "mortified": "shame_guilt",
            "disgrace": "shame_guilt",
            "worthless": "shame_guilt",
            
            # Happiness mappings
            "happy": "happiness",
            "joyful": "happiness",
            "cheerful": "happiness",
            "delighted": "happiness",
            "pleased": "happiness",
            "content": "happiness",
            "satisfied": "happiness",
            "glad": "happiness",
            "elated": "happiness",
            "euphoric": "happiness",
            "blissful": "happiness",
            "positive": "happiness",
            
            # Love/Affection mappings
            "love": "love_affection",
            "loving": "love_affection",
            "caring": "love_affection",
            "affectionate": "love_affection",
            "tender": "love_affection",
            "warm": "love_affection",
            "compassionate": "love_affection",
            "devoted": "love_affection",
            "fond": "love_affection",
            "attached": "love_affection",
            
            # Excitement mappings
            "excited": "excitement",
            "enthusiastic": "excitement",
            "eager": "excitement",
            "thrilled": "excitement",
            "energized": "excitement",
            "motivated": "excitement",
            "inspired": "excitement",
            "passionate": "excitement",
            "animated": "excitement",
            "pumped": "excitement",
            
            # Calm/Neutral mappings
            "calm": "calm_neutral",
            "peaceful": "calm_neutral",
            "relaxed": "calm_neutral",
            "tranquil": "calm_neutral",
            "serene": "calm_neutral",
            "neutral": "calm_neutral",
            "balanced": "calm_neutral",
            "composed": "calm_neutral",
            "stable": "calm_neutral",
            "centered": "calm_neutral",
            "okay": "calm_neutral",
            "fine": "calm_neutral",
            "normal": "calm_neutral",
        }
        
        # Combine with config mappings
        self.mapping.update(self.extended_mappings)
        
        # Common dataset-specific mappings (for various emotion datasets)
        self.dataset_specific_mappings = {
            # DailyDialog dataset emotions
            "no emotion": "calm_neutral",
            "happiness": "happiness",
            "sadness": "depression",
            "anger": "anger",
            "surprise": "excitement",
            "disgust": "anger",
            "fear": "anxiety",
            
            # GoEmotions mappings
            "admiration": "love_affection",
            "amusement": "happiness",
            "approval": "happiness",
            "gratitude": "happiness",
            "optimism": "excitement",
            "pride": "happiness",
            "realization": "calm_neutral",
            "relief": "calm_neutral",
            "desire": "excitement",
            "disappointment": "depression",
            "disapproval": "anger",
            "disgust": "anger",
            "grief": "depression",
            "nervousness": "anxiety",
            "remorse": "shame_guilt",
            "surprise": "excitement",
            
            # EmoContext mappings
            "others": "calm_neutral",
            "neutral": "calm_neutral",
            
            # Common sentiment labels
            "positive": "happiness",
            "negative": "depression",  # Default negative to depression
            "very_positive": "excitement",
            "very_negative": "depression",
            
            # Additional clinical terms
            "suicidal": "depression",
            "manic": "excitement",
            "psychotic": "confusion",
            "traumatized": "anxiety",
            "numb": "depression",
            "empty": "depression",
        }
        
        # Add dataset-specific mappings
        self.mapping.update(self.dataset_specific_mappings)
        
    def map_emotion(self, emotion: str) -> str:
        """
        Map a single emotion to one of 12 categories
        
        Args:
            emotion: Raw emotion label
            
        Returns:
            Mapped emotion category
        """
        if pd.isna(emotion):
            return "calm_neutral"
        
        emotion_lower = str(emotion).lower().strip()
        
        # Direct mapping
        if emotion_lower in self.mapping:
            return self.mapping[emotion_lower]
        
        # Fuzzy matching for close matches
        best_match = self.fuzzy_match_emotion(emotion_lower)
        if best_match:
            return best_match
        
        # Keyword-based mapping
        keyword_match = self.keyword_based_mapping(emotion_lower)
        if keyword_match:
            return keyword_match
        
        # Default to neutral if no match found
        logger.warning(f"No mapping found for emotion: '{emotion}'. Defaulting to 'calm_neutral'")
        return "calm_neutral"
    
    def fuzzy_match_emotion(self, emotion: str, threshold: int = 85) -> Optional[str]:
        """
        Use fuzzy matching to find closest emotion mapping
        
        Args:
            emotion: Emotion string to match
            threshold: Minimum similarity score (0-100)
            
        Returns:
            Matched category or None
        """
        best_score = 0
        best_match = None
        
        for key, category in self.mapping.items():
            score = fuzz.ratio(emotion, key)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = category
        
        if best_match:
            logger.debug(f"Fuzzy matched '{emotion}' to '{best_match}' with score {best_score}")
        
        return best_match
    
    def keyword_based_mapping(self, emotion: str) -> Optional[str]:
        """
        Map emotion based on keywords within the text
        
        Args:
            emotion: Emotion string
            
        Returns:
            Matched category or None
        """
        # Check if emotion contains keywords from categories
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword.lower() in emotion or emotion in keyword.lower():
                    logger.debug(f"Keyword matched '{emotion}' to '{category}' via '{keyword}'")
                    return category
        
        return None
    
    def map_dataframe(self, df: pd.DataFrame, emotion_column: str = 'emotion') -> pd.DataFrame:
        """
        Map emotions for entire dataframe
        
        Args:
            df: Input dataframe
            emotion_column: Name of emotion column
            
        Returns:
            Dataframe with mapped emotions
        """
        logger.info(f"Mapping {len(df)} emotions to 12 categories...")
        
        # Keep original emotion
        df['original_emotion'] = df[emotion_column].copy()
        
        # Map to new categories
        df['mapped_emotion'] = df[emotion_column].apply(self.map_emotion)
        
        # Replace the emotion column with mapped values
        df[emotion_column] = df['mapped_emotion']
        
        # Add emotion category ID
        df['emotion_id'] = df[emotion_column].apply(lambda x: LABEL_TO_ID[x])
        
        # Log distribution
        distribution = df[emotion_column].value_counts()
        logger.info("Emotion distribution after mapping:")
        for emotion, count in distribution.items():
            logger.info(f"  {emotion}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def validate_mapping(self, df: pd.DataFrame) -> Dict:
        """
        Validate emotion mapping and provide statistics
        
        Args:
            df: Dataframe with mapped emotions
            
        Returns:
            Dictionary of validation statistics
        """
        stats = {
            'total_samples': len(df),
            'unique_original': df['original_emotion'].nunique(),
            'unique_mapped': df['mapped_emotion'].nunique(),
            'distribution': df['mapped_emotion'].value_counts().to_dict(),
            'unmapped_count': 0,
            'class_balance': {}
        }
        
        # Check class balance
        for category in CLASS_LABELS:
            count = len(df[df['mapped_emotion'] == category])
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            stats['class_balance'][category] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        # Check for any unmapped (defaulted to neutral)
        original_emotions = df['original_emotion'].str.lower().unique()
        unmapped = []
        for emotion in original_emotions:
            if pd.notna(emotion) and emotion not in self.mapping:
                unmapped.append(emotion)
        
        stats['unmapped_emotions'] = unmapped
        stats['unmapped_count'] = len(unmapped)
        
        return stats


def augment_emotion_data(df: pd.DataFrame, augmentation_factor: float = AUGMENTATION_FACTOR) -> pd.DataFrame:
    """
    Augment emotion data to increase dataset size and diversity
    
    Args:
        df: Input dataframe
        augmentation_factor: Percentage of data to augment
        
    Returns:
        Augmented dataframe
    """
    logger.info(f"Augmenting data by {augmentation_factor*100}%...")
    
    augmented_samples = []
    n_augment = int(len(df) * augmentation_factor)
    
    # Sample rows to augment
    rows_to_augment = df.sample(n=n_augment, random_state=RANDOM_SEED)
    
    for _, row in rows_to_augment.iterrows():
        # Create variations
        text = row['cleaned_text'] if 'cleaned_text' in row else row['text']
        
        # Simple augmentation techniques
        augmentations = [
            # Add emotional intensifiers
            f"I really feel {text.lower()}",
            f"Honestly, {text}",
            f"{text} and it's affecting me deeply",
            
            # Add context
            f"Lately, {text.lower()}",
            f"{text}. This has been going on for a while",
            
            # Rephrase
            text.replace("I'm", "I am").replace("don't", "do not"),
        ]
        
        # Randomly select one augmentation
        aug_text = np.random.choice(augmentations)
        
        augmented_row = row.copy()
        augmented_row['cleaned_text'] = aug_text
        augmented_row['is_augmented'] = True
        augmented_samples.append(augmented_row)
    
    # Add augmented samples to dataframe
    df['is_augmented'] = False
    augmented_df = pd.DataFrame(augmented_samples)
    df = pd.concat([df, augmented_df], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    logger.info(f"Dataset augmented from {len(df) - len(augmented_df)} to {len(df)} samples")
    
    return df


def process_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to process and map emotions
    
    Args:
        df: Input dataframe with emotion column
        
    Returns:
        Processed dataframe with mapped emotions
    """
    logger.info("Starting emotion processing...")
    
    # Initialize mapper
    mapper = EmotionMapper()
    
    # Map emotions
    df = mapper.map_dataframe(df)
    
    # Validate mapping
    stats = mapper.validate_mapping(df)
    logger.info(f"Validation stats: {stats}")
    
    # Apply augmentation if enabled
    if AUGMENTATION_ENABLED:
        df = augment_emotion_data(df)
    
    return df


if __name__ == "__main__":
    # Test emotion mapping
    test_emotions = [
        "happy", "sad", "anxious", "angry", "love", "excited",
        "stressed", "confused", "pain", "guilty", "calm", "neutral",
        "depression", "fear", "joy", "frustrated", "lonely", "ashamed"
    ]
    
    mapper = EmotionMapper()
    
    print("Emotion Mapping Test:")
    print("-" * 50)
    for emotion in test_emotions:
        mapped = mapper.map_emotion(emotion)
        print(f"{emotion:20s} -> {mapped}")
    
    print("\nEmotion categories available:")
    for category in CLASS_LABELS:
        print(f"  - {category}")
