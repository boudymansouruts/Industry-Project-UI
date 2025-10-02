"""
Utility Module for BioBERT Health Risk Detection
Contains helper functions, visualizations, and utilities
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from pathlib import Path
from datetime import datetime
import random
import os
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
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

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def set_seed(seed: int = RANDOM_SEED):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def create_directories():
    """Create necessary project directories"""
    directories = [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR, CACHE_DIR, CHECKPOINT_DIR]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")


def save_json(data: Dict, path: Path):
    """
    Save dictionary to JSON file
    
    Args:
        data: Data to save
        path: Path to save file
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Data saved to {path}")


def load_json(path: Path) -> Dict:
    """
    Load JSON file
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        data = json.load(f)
    logger.info(f"Data loaded from {path}")
    return data


def plot_training_history(
    history: Dict,
    save_path: Path = RESULTS_DIR / "training_curves.png",
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot training history curves
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot F1 score
    axes[1, 0].plot(history['val_f1'], label='Val F1', marker='s', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    axes[1, 1].plot(history['learning_rates'], label='Learning Rate', marker='o', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Training history plot saved to {save_path}")


def plot_emotion_distribution(
    df: pd.DataFrame,
    column: str = 'mapped_emotion',
    save_path: Path = RESULTS_DIR / "emotion_distribution.png",
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot emotion distribution in dataset
    
    Args:
        df: Dataframe with emotions
        column: Column name with emotions
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Count emotions
    emotion_counts = df[column].value_counts()
    
    # Create bar plot
    colors = sns.color_palette("husl", len(emotion_counts))
    bars = plt.bar(range(len(emotion_counts)), emotion_counts.values, color=colors)
    
    # Customize plot
    plt.xlabel('Emotion Category', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Distribution of Emotions in Dataset', fontsize=16, fontweight='bold')
    plt.xticks(range(len(emotion_counts)), 
               [e.replace('_', ' ').title() for e in emotion_counts.index],
               rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, emotion_counts.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Emotion distribution plot saved to {save_path}")


def create_word_cloud(
    texts: List[str],
    emotion_category: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Create word cloud for specific emotion category
    
    Args:
        texts: List of texts
        emotion_category: Emotion category name
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Combine all texts
    combined_text = ' '.join(texts)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(combined_text)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - {emotion_category.replace("_", " ").title()}',
              fontsize=16, fontweight='bold')
    plt.tight_layout(pad=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_attention_weights(
    text: str,
    tokens: List[str],
    attention_weights: np.ndarray,
    layer: int = -1,
    head: int = 0,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Visualize attention weights as heatmap
    
    Args:
        text: Original text
        tokens: List of tokens
        attention_weights: Attention weight matrix
        layer: Which layer to visualize (-1 for last)
        head: Which attention head to visualize
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Select specific layer and head
    if len(attention_weights.shape) == 4:  # [layers, heads, seq_len, seq_len]
        attention = attention_weights[layer, head]
    else:
        attention = attention_weights
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(attention, cmap='Blues', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add title
    ax.set_title(f'Attention Weights\nLayer {layer}, Head {head}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_interactive_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_labels: List[str] = CLASS_LABELS,
    save_path: Optional[Path] = None
):
    """
    Create interactive confusion matrix using Plotly
    
    Args:
        confusion_matrix: Confusion matrix array
        class_labels: List of class labels
        save_path: Optional path to save HTML
    """
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Create hovertext
    hovertext = []
    for i in range(len(class_labels)):
        row = []
        for j in range(len(class_labels)):
            text = f'True: {class_labels[i]}<br>'
            text += f'Predicted: {class_labels[j]}<br>'
            text += f'Count: {confusion_matrix[i, j]}<br>'
            text += f'Percentage: {cm_normalized[i, j]:.2%}'
            row.append(text)
        hovertext.append(row)
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=[label.replace('_', ' ').title() for label in class_labels],
        y=[label.replace('_', ' ').title() for label in class_labels],
        hovertext=hovertext,
        hovertemplate='%{hovertext}<extra></extra>',
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title='Proportion')
    ))
    
    # Update layout
    fig.update_layout(
        title='Interactive Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=900,
        height=800,
        xaxis=dict(tickangle=45)
    )
    
    # Show figure
    fig.show()
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Interactive confusion matrix saved to {save_path}")


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = CLASS_LABELS,
    method: str = 'tsne',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Visualize high-dimensional embeddings in 2D
    
    Args:
        embeddings: Embedding vectors
        labels: Class labels
        class_names: Names of classes
        method: Dimensionality reduction method ('tsne' or 'pca')
        save_path: Optional path to save figure
        figsize: Figure size
    """
    logger.info(f"Reducing dimensions using {method.upper()}...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:  # PCA
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=RANDOM_SEED)
        embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot each class
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_name.replace('_', ' ').title(),
            alpha=0.7,
            s=50
        )
    
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.title(f'Embedding Visualization ({method.upper()})', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def generate_summary_report(
    model_metrics: Dict,
    training_time: float,
    save_path: Path = RESULTS_DIR / "summary_report.txt"
):
    """
    Generate comprehensive summary report
    
    Args:
        model_metrics: Dictionary of model metrics
        training_time: Training time in seconds
        save_path: Path to save report
    """
    report = []
    report.append("=" * 80)
    report.append("BIOBERT HEALTH RISK DETECTION - SUMMARY REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Model Configuration
    report.append("MODEL CONFIGURATION:")
    report.append(f"  Base Model: {MODEL_NAME}")
    report.append(f"  Number of Classes: {NUM_CLASSES}")
    report.append(f"  Max Sequence Length: {MAX_LENGTH}")
    report.append(f"  Batch Size: {BATCH_SIZE}")
    report.append(f"  Learning Rate: {LEARNING_RATE}")
    report.append(f"  Dropout Rate: {DROPOUT_RATE}")
    report.append("")
    
    # Training Details
    report.append("TRAINING DETAILS:")
    report.append(f"  Number of Epochs: {NUM_EPOCHS}")
    report.append(f"  Training Time: {training_time/60:.2f} minutes")
    report.append(f"  Device: {DEVICE}")
    report.append(f"  Mixed Precision: {FP16_TRAINING}")
    report.append("")
    
    # Performance Metrics
    report.append("PERFORMANCE METRICS:")
    report.append(f"  Overall Accuracy: {model_metrics.get('accuracy', 0):.4f}")
    report.append(f"  Weighted F1-Score: {model_metrics.get('weighted_f1', 0):.4f}")
    report.append(f"  Macro F1-Score: {model_metrics.get('macro_f1', 0):.4f}")
    report.append(f"  ROC-AUC Score: {model_metrics.get('roc_auc_ovr', 0):.4f}")
    report.append("")
    
    # Class Distribution
    report.append("EMOTION CATEGORIES:")
    for i, category in enumerate(CLASS_LABELS):
        report.append(f"  {i+1:2d}. {category.replace('_', ' ').title()}")
    report.append("")
    
    # High-Risk Categories
    report.append("HIGH-RISK CATEGORIES:")
    high_risk = ['depression', 'anxiety', 'loneliness', 'physical_pain']
    for category in high_risk:
        if 'per_class' in model_metrics and category in model_metrics['per_class']:
            metrics = model_metrics['per_class'][category]
            report.append(f"  {category.replace('_', ' ').title()}:")
            report.append(f"    - F1-Score: {metrics['f1']:.4f}")
            report.append(f"    - Precision: {metrics['precision']:.4f}")
            report.append(f"    - Recall: {metrics['recall']:.4f}")
    report.append("")
    
    # Files Generated
    report.append("OUTPUT FILES:")
    report.append(f"  Model: {BEST_MODEL_PATH}")
    report.append(f"  Predictions: {PREDICTIONS_PATH}")
    report.append(f"  Confusion Matrix: {CONFUSION_MATRIX_PATH}")
    report.append(f"  Classification Report: {CLASSIFICATION_REPORT_PATH}")
    report.append(f"  Training History: {TRAINING_HISTORY_PATH}")
    report.append("")
    
    report.append("=" * 80)
    report.append("Report generation complete!")
    
    # Save report
    report_text = '\n'.join(report)
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    logger.info(f"Summary report saved to {save_path}")
    
    return report_text


def calculate_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Calculate model size and parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'model_size_mb': model_size_mb,
        'param_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024
    }


def monitor_gpu_usage():
    """
    Monitor GPU memory usage
    
    Returns:
        Dictionary with GPU usage information
    """
    if not torch.cuda.is_available():
        return {'gpu_available': False}
    
    gpu_info = {
        'gpu_available': True,
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
        'device_name': torch.cuda.get_device_name(),
        'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        'max_memory_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        'memory_free_gb': (torch.cuda.get_device_properties(0).total_memory - 
                          torch.cuda.memory_allocated()) / 1024**3
    }
    
    return gpu_info


if __name__ == "__main__":
    # Test utilities
    logger.info("Testing utility functions...")
    
    # Set seed
    set_seed()
    
    # Create directories
    create_directories()
    
    # Monitor GPU
    gpu_info = monitor_gpu_usage()
    print("\nGPU Information:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    
    print("\nUtility module ready for use!")
