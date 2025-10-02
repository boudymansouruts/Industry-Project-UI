"""
Data Preprocessing Module for BioBERT Health Risk Detection
Handles loading, cleaning, and preprocessing of conversational text data
"""

import pandas as pd
import numpy as np
import re
import string
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import contractions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Comprehensive text preprocessing for conversational data
    """
    
    def __init__(self, options: Dict = None):
        """
        Initialize text preprocessor with cleaning options
        
        Args:
            options: Dictionary of preprocessing options
        """
        self.options = options or TEXT_CLEANING_OPTIONS
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def clean_text(self, text: str) -> str:
        """
        Apply comprehensive text cleaning
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not text:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove HTML tags
        if self.options.get('remove_html_tags', True):
            text = self.html_pattern.sub(' ', text)
        
        # Expand contractions
        if self.options.get('expand_contractions', True):
            text = contractions.fix(text)
        
        # Remove URLs
        if self.options.get('remove_urls', True):
            text = self.url_pattern.sub(' ', text)
        
        # Remove emails
        if self.options.get('remove_emails', True):
            text = self.email_pattern.sub(' ', text)
        
        # Remove phone numbers
        if self.options.get('remove_phone_numbers', True):
            text = self.phone_pattern.sub(' ', text)
        
        # Handle mentions and hashtags (keep the text, remove symbols)
        text = self.mention_pattern.sub(lambda m: m.group()[1:], text)
        text = self.hashtag_pattern.sub(lambda m: m.group()[1:], text)
        
        # Lowercase if specified
        if self.options.get('lowercase', False):
            text = text.lower()
        
        # Normalize whitespace
        if self.options.get('normalize_whitespace', True):
            text = self.whitespace_pattern.sub(' ', text)
            text = text.strip()
        
        # Length constraints
        if len(text) < MIN_TEXT_LENGTH:
            return ""
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
        
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Preprocess entire dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            
        Returns:
            Preprocessed dataframe
        """
        logger.info(f"Preprocessing {len(df)} samples...")
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > MIN_TEXT_LENGTH]
        
        logger.info(f"After preprocessing: {len(df)} samples remain")
        
        return df


class DataLoader:
    """
    Handles loading and initial processing of DailyTalk dataset
    """
    
    def __init__(self, data_path: Path = RAW_DATA_PATH):
        """
        Initialize data loader
        
        Args:
            data_path: Path to raw data file
        """
        self.data_path = data_path
        self.preprocessor = TextPreprocessor()
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            Loaded dataframe
        """
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(self.data_path, encoding=encoding)
                    logger.info(f"Successfully loaded data with {encoding} encoding")
                    break
                except:
                    continue
            
            # Check required columns
            required_columns = ['text', 'emotion']
            if not all(col in df.columns for col in required_columns):
                # If columns have different names, try common alternatives
                if 'utterance' in df.columns:
                    df.rename(columns={'utterance': 'text'}, inplace=True)
                if 'label' in df.columns:
                    df.rename(columns={'label': 'emotion'}, inplace=True)
                if 'sentiment' in df.columns:
                    df.rename(columns={'sentiment': 'emotion'}, inplace=True)
            
            logger.info(f"Loaded {len(df)} samples from {self.data_path}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Create sample data if file doesn't exist
            return self.create_sample_data()
    
    def create_sample_data(self) -> pd.DataFrame:
        """
        Create sample dataset for testing if real data not available
        
        Returns:
            Sample dataframe
        """
        logger.warning("Creating sample dataset for demonstration...")
        
        sample_texts = [
            # Depression samples
            "I feel so hopeless and empty inside, nothing seems to matter anymore",
            "Every day is a struggle, I can't find joy in anything",
            "I'm so tired of feeling this way, the sadness never goes away",
            
            # Anxiety samples
            "I'm constantly worried about everything, my heart won't stop racing",
            "I feel so nervous all the time, like something bad is about to happen",
            "The panic attacks are getting worse, I can't breathe properly",
            
            # Stress samples
            "Work is overwhelming me, I can't handle all this pressure",
            "I'm so stressed out with everything, feeling completely burned out",
            "The deadlines are killing me, I'm under so much strain",
            
            # Anger samples
            "I'm so angry at everyone, I just want to scream",
            "This frustration is eating me alive, I can't control my rage",
            "I hate feeling this hostile all the time",
            
            # Loneliness samples
            "I feel so alone even when surrounded by people",
            "Nobody understands me, I'm completely isolated",
            "The loneliness is crushing, I have no one to talk to",
            
            # Confusion samples
            "I'm so confused about everything, my mind is foggy",
            "I can't think clearly anymore, everything is uncertain",
            "My thoughts are all jumbled, I feel so disoriented",
            
            # Physical pain samples
            "My body aches all over, I'm exhausted all the time",
            "The chronic pain never stops, I'm so tired",
            "I feel sick and weak, my energy is completely gone",
            
            # Shame/Guilt samples
            "I'm so ashamed of myself, I feel worthless",
            "The guilt is eating me up inside",
            "I'm embarrassed about everything I've done",
            
            # Happiness samples
            "I feel so happy and content with life right now!",
            "Today was amazing, I'm filled with joy and satisfaction",
            "Everything is going great, I'm really pleased with how things are",
            
            # Love/Affection samples
            "I love spending time with my family, they mean everything to me",
            "Feeling so much warmth and affection for my friends",
            "My heart is full of love and compassion today",
            
            # Excitement samples
            "I'm so excited about the new project, can't wait to start!",
            "Feeling energized and motivated to tackle new challenges",
            "The anticipation is thrilling, I'm pumped for what's coming",
            
            # Calm/Neutral samples
            "I'm feeling pretty balanced today, everything is okay",
            "Things are calm and peaceful, I'm in a good place",
            "Feeling neutral about the situation, just taking it as it comes",
        ]
        
        # Create more samples through variation
        expanded_texts = []
        expanded_emotions = []
        
        emotions = ["sadness", "fear", "stressed", "angry", "lonely", "confused", 
                   "pain", "ashamed", "happy", "love", "excited", "calm"]
        
        for i, text in enumerate(sample_texts):
            emotion_idx = i // 3  # 3 samples per emotion
            if emotion_idx < len(emotions):
                emotion = emotions[emotion_idx]
                # Add original
                expanded_texts.append(text)
                expanded_emotions.append(emotion)
                
                # Add variations
                variations = [
                    text.replace("I'm", "I am"),
                    text.replace("so", "very"),
                    text + " It's been really hard lately.",
                    "Honestly, " + text.lower(),
                    text.replace(".", "..."),
                ]
                
                for var in variations[:2]:  # Add 2 variations per sample
                    expanded_texts.append(var)
                    expanded_emotions.append(emotion)
        
        df = pd.DataFrame({
            'text': expanded_texts,
            'emotion': expanded_emotions
        })
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        logger.info(f"Created sample dataset with {len(df)} samples")
        
        return df
    
    def balance_dataset(self, df: pd.DataFrame, target_samples: int = SAMPLES_PER_CLASS) -> pd.DataFrame:
        """
        Balance dataset to have roughly equal samples per class
        
        Args:
            df: Input dataframe
            target_samples: Target number of samples per class
            
        Returns:
            Balanced dataframe
        """
        logger.info("Balancing dataset...")
        
        balanced_dfs = []
        
        for emotion in df['mapped_emotion'].unique():
            emotion_df = df[df['mapped_emotion'] == emotion]
            current_count = len(emotion_df)
            
            if current_count > target_samples:
                # Downsample
                emotion_df = emotion_df.sample(n=target_samples, random_state=RANDOM_SEED)
            elif current_count < target_samples:
                # Upsample with replacement
                n_additional = target_samples - current_count
                if current_count > 0:
                    additional_samples = emotion_df.sample(
                        n=min(n_additional, current_count * 10),  # Limit upsampling
                        replace=True,
                        random_state=RANDOM_SEED
                    )
                    emotion_df = pd.concat([emotion_df, additional_samples])
            
            balanced_dfs.append(emotion_df)
            logger.info(f"{emotion}: {len(emotion_df)} samples")
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        logger.info(f"Balanced dataset: {len(balanced_df)} total samples")
        
        return balanced_df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input dataframe
            
        Returns:
            Train, validation, and test dataframes
        """
        # First split: train+val and test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=TEST_SPLIT, 
            random_state=RANDOM_SEED,
            stratify=df['mapped_emotion']
        )
        
        # Second split: train and val
        val_size = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=val_size, 
            random_state=RANDOM_SEED,
            stratify=train_val_df['mapped_emotion']
        )
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df


def prepare_data():
    """
    Main function to prepare data for training
    """
    logger.info("Starting data preparation...")
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load raw data
    df = loader.load_data()
    
    # Preprocess text
    df = loader.preprocessor.preprocess_dataframe(df)
    
    # Map emotions to categories (will be done in emotion_mapping.py)
    # For now, we'll assume the mapping is done
    
    # Save processed data
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Saved processed data to {PROCESSED_DATA_PATH}")
    
    return df


if __name__ == "__main__":
    # Run data preparation
    df = prepare_data()
    print(f"Data preparation complete! Total samples: {len(df)}")
