"""
Model Training Module for BioBERT Health Risk Detection
Handles fine-tuning of BioBERT for emotion classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, PolynomialLR
from transformers import (
    AutoModel, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
from tqdm import tqdm
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Import configuration
from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BioBERTClassifier(nn.Module):
    """
    BioBERT model with custom classification head for health risk detection
    """
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = DROPOUT_RATE,
        freeze_bert: bool = False
    ):
        """
        Initialize BioBERT classifier
        
        Args:
            model_name: Name of pretrained model
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            freeze_bert: Whether to freeze BERT layers
        """
        super(BioBERTClassifier, self).__init__()
        
        # Load BioBERT
        self.bert = AutoModel.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR
        )
        
        # Get hidden size
        self.hidden_size = self.bert.config.hidden_size
        
        # Freeze BERT if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head with enhanced architecture
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with logits and optionally attention weights
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attention
        )
        
        # Use pooled output (CLS token)
        pooled_output = outputs.pooler_output
        
        # Classification
        logits = self.classifier(pooled_output)
        
        result = {'logits': logits}
        
        # Add attention weights if requested
        if return_attention:
            result['attentions'] = outputs.attentions
        
        return result


class ModelTrainer:
    """
    Handles model training and optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = DEVICE,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            device: Device to use
            class_weights: Class weights for imbalanced data
        """
        self.model = model.to(device)
        self.device = device
        self.class_weights = class_weights
        
        if class_weights is not None:
            self.class_weights = class_weights.to(device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_f1 = 0
        self.best_model_state = None
        self.patience_counter = 0
        
    def setup_training(
        self,
        train_dataloader,
        val_dataloader,
        num_epochs: int = NUM_EPOCHS
    ):
        """
        Setup training components
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
        """
        # Calculate total training steps
        num_training_steps = len(train_dataloader) * num_epochs
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler(num_training_steps)
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        logger.info(f"Training setup complete:")
        logger.info(f"  Total steps: {num_training_steps}")
        logger.info(f"  Warmup steps: {WARMUP_STEPS}")
        logger.info(f"  Learning rate: {LEARNING_RATE}")
        
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Setup optimizer with differential learning rates
        
        Returns:
            Configured optimizer
        """
        # Different learning rates for different layers
        bert_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                bert_params.append(param)
            else:
                classifier_params.append(param)
        
        optimizer = AdamW([
            {'params': bert_params, 'lr': LEARNING_RATE},
            {'params': classifier_params, 'lr': LEARNING_RATE * 10}  # Higher LR for classifier
        ], 
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY
        )
        
        return optimizer
    
    def _setup_scheduler(self, num_training_steps: int):
        """
        Setup learning rate scheduler
        
        Args:
            num_training_steps: Total training steps
            
        Returns:
            Configured scheduler
        """
        if not USE_SCHEDULER:
            return None
        
        if SCHEDULER_TYPE == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=WARMUP_STEPS,
                num_training_steps=num_training_steps
            )
        elif SCHEDULER_TYPE == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=WARMUP_STEPS,
                num_training_steps=num_training_steps
            )
        else:  # polynomial
            scheduler = PolynomialLR(
                self.optimizer,
                total_iters=num_training_steps,
                power=2.0
            )
        
        return scheduler
    
    def train_epoch(self, dataloader, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average loss and accuracy
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Gradient accumulation
            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            
            # Update weights
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_acc = correct_predictions / total_samples
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = correct_predictions / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self, dataloader) -> Tuple[float, float, float]:
        """
        Validate model
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Average loss, accuracy, and F1 score
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def train(
        self,
        train_dataloader,
        val_dataloader,
        num_epochs: int = NUM_EPOCHS,
        save_dir: Path = CHECKPOINT_DIR
    ):
        """
        Full training loop
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
        """
        logger.info("Starting training...")
        
        # Setup training
        self.setup_training(train_dataloader, val_dataloader, num_epochs)
        
        # Create save directory
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_dataloader, epoch)
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate(val_dataloader)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Check for improvement
            if val_f1 > self.best_val_f1:
                logger.info(f"New best model! F1: {val_f1:.4f}")
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict()
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(save_dir / "best_model.pt", epoch, val_f1)
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}/{EARLY_STOPPING_PATIENCE}")
            
            # Early stopping
            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered!")
                break
            
            # Save checkpoint
            if epoch % 2 == 0:  # Save every 2 epochs
                self.save_checkpoint(save_dir / f"checkpoint_epoch_{epoch}.pt", epoch, val_f1)
        
        # Save training history
        self.save_history(RESULTS_DIR / "training_history.json")
        
        logger.info(f"Training complete! Best F1: {self.best_val_f1:.4f}")
        
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int, val_f1: float):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            val_f1: Validation F1 score
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_f1': val_f1,
            'history': self.history,
            'config': {
                'num_classes': NUM_CLASSES,
                'model_name': MODEL_NAME,
                'max_length': MAX_LENGTH
            }
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded from {path}")
        
        return checkpoint
    
    def save_history(self, path: Path):
        """
        Save training history
        
        Args:
            path: Path to save history
        """
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {path}")


def create_model(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE
) -> BioBERTClassifier:
    """
    Create BioBERT classifier model
    
    Args:
        model_name: Name of pretrained model
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        
    Returns:
        Created model
    """
    logger.info(f"Creating BioBERT model...")
    
    model = BioBERTClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    logger.info("Testing model creation...")
    
    # Create model
    model = create_model()
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    
    # Create dummy inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs['logits']
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected shape: ({batch_size}, {NUM_CLASSES})")
    
    print("\nModel testing complete!")
