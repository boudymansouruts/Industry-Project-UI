"""
Dataset Creation Module for BioBERT Health Risk Detection
Creates PyTorch datasets with proper tokenization for BioBERT
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer
import logging
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

# Import configuration
from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthRiskDataset(Dataset):
    """
    PyTorch Dataset for health risk emotion detection
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_length: int = MAX_LENGTH,
        text_column: str = 'cleaned_text',
        label_column: str = 'emotion_id'
    ):
        """
        Initialize dataset
        
        Args:
            data: DataFrame with text and labels
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            text_column: Name of text column
            label_column: Name of label column
        """
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        
        # Use cleaned_text if available, otherwise use text
        if text_column not in self.data.columns and 'text' in self.data.columns:
            self.text_column = 'text'
        
        logger.info(f"Created dataset with {len(self.data)} samples")
        
    def __len__(self) -> int:
        """Return dataset length"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single item from dataset
        
        Args:
            idx: Index of item
            
        Returns:
            Dictionary with tokenized inputs and label
        """
        row = self.data.iloc[idx]
        
        # Get text and label
        text = str(row[self.text_column])
        label = int(row[self.label_column])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text  # Keep original text for analysis
        }


class DataCollator:
    """
    Custom data collator for batching
    """
    
    def __init__(self, return_text: bool = False):
        """
        Initialize collator
        
        Args:
            return_text: Whether to return original text in batch
        """
        self.return_text = return_text
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched dictionary
        """
        # Stack tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        # Optionally include text
        if self.return_text:
            result['text'] = [item['text'] for item in batch]
        
        return result


class DatasetBuilder:
    """
    Builds and manages datasets for training
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize dataset builder
        
        Args:
            model_name: Name of model for tokenizer
        """
        self.tokenizer = self._load_tokenizer(model_name)
        self.class_weights = None
        
    def _load_tokenizer(self, model_name: str):
        """
        Load tokenizer for model
        
        Args:
            model_name: Model name
            
        Returns:
            Loaded tokenizer
        """
        logger.info(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            do_lower_case=False  # BioBERT is cased
        )
        return tokenizer
    
    def compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """
        Compute class weights for imbalanced data
        
        Args:
            labels: Array of labels
            
        Returns:
            Tensor of class weights for all NUM_CLASSES classes
        """
        logger.info("Computing class weights for balanced training...")
        
        # Initialize weights for all classes
        all_class_weights = np.ones(NUM_CLASSES)
        
        # Get unique labels present in data
        unique_labels = np.unique(labels)
        
        # Compute weights only for classes that have samples
        if len(unique_labels) > 0:
            class_weights_present = compute_class_weight(
                class_weight='balanced',
                classes=unique_labels,
                y=labels
            )
            
            # Assign computed weights to corresponding classes
            for idx, label in enumerate(unique_labels):
                all_class_weights[label] = class_weights_present[idx]
        
        # Convert to tensor
        class_weights = torch.tensor(all_class_weights, dtype=torch.float32)
        
        logger.info(f"Class weights computed for {len(unique_labels)} out of {NUM_CLASSES} classes")
        logger.info(f"Class weights: {class_weights.tolist()}")
        
        return class_weights
    
    def create_dataset(
        self,
        data: pd.DataFrame,
        text_column: str = 'cleaned_text',
        label_column: str = 'emotion_id'
    ) -> HealthRiskDataset:
        """
        Create dataset from dataframe
        
        Args:
            data: Input dataframe
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Created dataset
        """
        dataset = HealthRiskDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=MAX_LENGTH,
            text_column=text_column,
            label_column=label_column
        )
        
        return dataset
    
    def create_weighted_sampler(self, dataset: HealthRiskDataset) -> WeightedRandomSampler:
        """
        Create weighted sampler for balanced batches
        
        Args:
            dataset: Input dataset
            
        Returns:
            Weighted sampler
        """
        logger.info("Creating weighted sampler for balanced batches...")
        
        # Get labels
        labels = dataset.data['emotion_id'].values
        
        # Compute sample weights
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        return sampler
    
    def create_dataloader(
        self,
        dataset: HealthRiskDataset,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        use_sampler: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True,
        return_text: bool = False
    ) -> DataLoader:
        """
        Create DataLoader from dataset
        
        Args:
            dataset: Input dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            use_sampler: Whether to use weighted sampler
            num_workers: Number of workers for loading
            pin_memory: Whether to pin memory for GPU
            return_text: Whether to return original text
            
        Returns:
            Created DataLoader
        """
        # Create collator
        collator = DataCollator(return_text=return_text)
        
        # Create sampler if needed
        sampler = None
        if use_sampler:
            sampler = self.create_weighted_sampler(dataset)
            shuffle = False  # Sampler handles shuffling
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=collator,
            drop_last=False
        )
        
        logger.info(f"Created DataLoader with {len(dataloader)} batches")
        
        return dataloader
    
    def prepare_data_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        use_weighted_sampling: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare all data splits for training
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            use_weighted_sampling: Whether to use weighted sampling for training
            
        Returns:
            Train, validation, and test DataLoaders
        """
        logger.info("Preparing data splits...")
        
        # Create datasets
        train_dataset = self.create_dataset(train_df)
        val_dataset = self.create_dataset(val_df)
        test_dataset = self.create_dataset(test_df)
        
        # Compute class weights from training data
        self.class_weights = self.compute_class_weights(train_df['emotion_id'].values)
        
        # Create dataloaders
        train_loader = self.create_dataloader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            use_sampler=use_weighted_sampling
        )
        
        val_loader = self.create_dataloader(
            val_dataset,
            batch_size=BATCH_SIZE * 2,  # Can use larger batch for validation
            shuffle=False,
            use_sampler=False
        )
        
        test_loader = self.create_dataloader(
            test_dataset,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            use_sampler=False,
            return_text=True  # Return text for analysis
        )
        
        logger.info(f"Data splits prepared:")
        logger.info(f"  Train: {len(train_loader)} batches")
        logger.info(f"  Val: {len(val_loader)} batches")
        logger.info(f"  Test: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader


def tokenize_examples(texts: List[str], tokenizer, max_length: int = MAX_LENGTH) -> Dict:
    """
    Tokenize a list of texts
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of tokenized inputs
    """
    return tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )


def analyze_token_lengths(df: pd.DataFrame, tokenizer, text_column: str = 'cleaned_text') -> Dict:
    """
    Analyze token lengths in dataset to determine optimal max_length
    
    Args:
        df: Input dataframe
        tokenizer: Tokenizer to use
        text_column: Name of text column
        
    Returns:
        Dictionary of statistics
    """
    logger.info("Analyzing token lengths...")
    
    lengths = []
    for text in df[text_column].values:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(tokens))
    
    lengths = np.array(lengths)
    
    stats = {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'percentile_50': np.percentile(lengths, 50),
        'percentile_90': np.percentile(lengths, 90),
        'percentile_95': np.percentile(lengths, 95),
        'percentile_99': np.percentile(lengths, 99)
    }
    
    logger.info("Token length statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:.2f}")
    
    return stats


if __name__ == "__main__":
    # Test dataset creation
    logger.info("Testing dataset creation...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'cleaned_text': [
            "I feel very anxious about the future",
            "Today was a great day, feeling happy!",
            "So stressed with work deadlines",
            "Feeling lonely and isolated",
            "Excited about the new project!"
        ],
        'emotion_id': [1, 8, 2, 4, 10],
        'mapped_emotion': ['anxiety', 'happiness', 'stress', 'loneliness', 'excitement']
    })
    
    # Initialize builder
    builder = DatasetBuilder()
    
    # Analyze token lengths
    stats = analyze_token_lengths(sample_data, builder.tokenizer)
    
    # Create dataset
    dataset = builder.create_dataset(sample_data)
    
    # Test getting an item
    sample = dataset[0]
    print(f"\nSample item:")
    print(f"  Input shape: {sample['input_ids'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  Label: {sample['label']}")
    
    # Create dataloader
    dataloader = builder.create_dataloader(dataset, batch_size=2)
    
    # Test batch
    for batch in dataloader:
        print(f"\nBatch shapes:")
        print(f"  Input IDs: {batch['input_ids'].shape}")
        print(f"  Attention mask: {batch['attention_mask'].shape}")
        print(f"  Labels: {batch['labels'].shape}")
        break
    
    print("\nDataset creation test complete!")
