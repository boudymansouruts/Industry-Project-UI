"""
Evaluation Module for BioBERT Health Risk Detection
Comprehensive model evaluation with metrics and visualizations
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve
)
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm
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

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive evaluation for health risk detection model
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = DEVICE,
        class_labels: List[str] = CLASS_LABELS
    ):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            device: Device to use
            class_labels: List of class label names
        """
        self.model = model.to(device)
        self.device = device
        self.class_labels = class_labels
        self.results = {}
        
    def evaluate(
        self,
        dataloader,
        save_predictions: bool = True,
        return_attention: bool = False
    ) -> Dict:
        """
        Comprehensive evaluation on test set
        
        Args:
            dataloader: Test data loader
            save_predictions: Whether to save predictions
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting comprehensive evaluation...")
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_texts = []
        all_attentions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get texts if available
                if 'text' in batch:
                    all_texts.extend(batch['text'])
                
                # Forward pass
                outputs = self.model(
                    input_ids, 
                    attention_mask,
                    return_attention=return_attention
                )
                logits = outputs['logits']
                
                # Get probabilities
                probabilities = F.softmax(logits, dim=1)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Store attention weights if available
                if return_attention and 'attentions' in outputs:
                    # Average attention across all layers and heads
                    attention = torch.stack(outputs['attentions']).mean(dim=(0, 1))
                    all_attentions.extend(attention.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        self.results = self.calculate_metrics(
            all_labels,
            all_predictions,
            all_probabilities
        )
        
        # Add predictions to results
        self.results['predictions'] = all_predictions
        self.results['labels'] = all_labels
        self.results['probabilities'] = all_probabilities
        
        if all_texts:
            self.results['texts'] = all_texts
        
        if all_attentions:
            self.results['attentions'] = all_attentions
        
        # Save predictions if requested
        if save_predictions:
            self.save_predictions()
        
        return self.results
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(NUM_CLASSES)
        )
        
        # Weighted metrics
        metrics['weighted_precision'] = np.average(precision, weights=support)
        metrics['weighted_recall'] = np.average(recall, weights=support)
        metrics['weighted_f1'] = np.average(f1, weights=support)
        
        # Macro metrics
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        # Per-class detailed metrics
        metrics['per_class'] = {}
        for i, label in enumerate(self.class_labels):
            metrics['per_class'][label] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(
            y_true, y_pred, labels=range(NUM_CLASSES)
        )
        
        # ROC-AUC (one-vs-rest)
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='weighted'
            )
            metrics['roc_auc_ovo'] = roc_auc_score(
                y_true, y_prob, multi_class='ovo', average='weighted'
            )
        except:
            metrics['roc_auc_ovr'] = 0.0
            metrics['roc_auc_ovo'] = 0.0
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, 
            target_names=self.class_labels,
            output_dict=True
        )
        
        # Log key metrics
        logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        save_path: Path = CONFUSION_MATRIX_PATH,
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Plot confusion matrix
        
        Args:
            save_path: Path to save figure
            normalize: Whether to normalize values
            figsize: Figure size
        """
        cm = self.results['confusion_matrix']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_labels,
            yticklabels=self.class_labels,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def plot_per_class_metrics(
        self,
        save_path: Path = RESULTS_DIR / "per_class_metrics.png",
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Plot per-class precision, recall, and F1
        
        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        # Prepare data
        classes = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for label in self.class_labels:
            classes.append(label.replace('_', ' ').title())
            precisions.append(self.results['per_class'][label]['precision'])
            recalls.append(self.results['per_class'][label]['recall'])
            f1_scores.append(self.results['per_class'][label]['f1'])
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars1 = ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Emotion Category', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Per-class metrics plot saved to {save_path}")
    
    def plot_roc_curves(
        self,
        save_path: Path = RESULTS_DIR / "roc_curves.png",
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Plot ROC curves for each class
        
        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        y_true = self.results['labels']
        y_prob = self.results['probabilities']
        
        plt.figure(figsize=figsize)
        
        for i, label in enumerate(self.class_labels):
            # Binary classification: current class vs rest
            y_binary = (y_true == i).astype(int)
            y_score = y_prob[:, i]
            
            fpr, tpr, _ = roc_curve(y_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{label.replace("_", " ").title()} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for All Classes', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"ROC curves saved to {save_path}")
    
    def analyze_errors(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analyze misclassified examples
        
        Args:
            top_n: Number of top errors to analyze
            
        Returns:
            DataFrame with error analysis
        """
        if 'texts' not in self.results:
            logger.warning("No texts available for error analysis")
            return pd.DataFrame()
        
        y_true = self.results['labels']
        y_pred = self.results['predictions']
        y_prob = self.results['probabilities']
        texts = self.results['texts']
        
        # Find misclassified samples
        errors_mask = y_true != y_pred
        error_indices = np.where(errors_mask)[0]
        
        if len(error_indices) == 0:
            logger.info("No errors found!")
            return pd.DataFrame()
        
        # Calculate confidence for wrong predictions
        wrong_confidences = []
        for idx in error_indices:
            wrong_confidence = y_prob[idx, y_pred[idx]]
            wrong_confidences.append(wrong_confidence)
        
        # Sort by confidence (most confident errors first)
        sorted_indices = error_indices[np.argsort(wrong_confidences)[::-1]]
        
        # Create error analysis dataframe
        error_data = []
        for idx in sorted_indices[:top_n]:
            error_data.append({
                'text': texts[idx][:100] + '...' if len(texts[idx]) > 100 else texts[idx],
                'true_label': self.class_labels[y_true[idx]],
                'predicted_label': self.class_labels[y_pred[idx]],
                'confidence': y_prob[idx, y_pred[idx]],
                'true_label_prob': y_prob[idx, y_true[idx]]
            })
        
        error_df = pd.DataFrame(error_data)
        
        # Save error analysis
        error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)
        logger.info(f"Error analysis saved to {ERROR_ANALYSIS_PATH}")
        
        return error_df
    
    def get_prediction_confidence_distribution(self):
        """
        Analyze prediction confidence distribution
        
        Returns:
            Dictionary with confidence statistics
        """
        y_prob = self.results['probabilities']
        max_probs = np.max(y_prob, axis=1)
        
        confidence_stats = {
            'mean': float(np.mean(max_probs)),
            'std': float(np.std(max_probs)),
            'min': float(np.min(max_probs)),
            'max': float(np.max(max_probs)),
            'median': float(np.median(max_probs)),
            'q25': float(np.percentile(max_probs, 25)),
            'q75': float(np.percentile(max_probs, 75))
        }
        
        # Confidence bins
        bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(max_probs, bins=bins)
        confidence_stats['distribution'] = {
            f'{bins[i]:.1f}-{bins[i+1]:.1f}': int(hist[i])
            for i in range(len(hist))
        }
        
        return confidence_stats
    
    def save_predictions(self):
        """Save predictions to CSV"""
        predictions_data = {
            'true_label': [self.class_labels[i] for i in self.results['labels']],
            'predicted_label': [self.class_labels[i] for i in self.results['predictions']],
            'confidence': np.max(self.results['probabilities'], axis=1)
        }
        
        # Add probabilities for each class
        for i, label in enumerate(self.class_labels):
            predictions_data[f'prob_{label}'] = self.results['probabilities'][:, i]
        
        # Add texts if available
        if 'texts' in self.results:
            predictions_data['text'] = self.results['texts']
        
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(PREDICTIONS_PATH, index=False)
        
        logger.info(f"Predictions saved to {PREDICTIONS_PATH}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=" * 80)
        report.append("BioBERT Health Risk Detection - Evaluation Report")
        report.append("=" * 80)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL METRICS:")
        report.append(f"  Accuracy: {self.results['accuracy']:.4f}")
        report.append(f"  Weighted Precision: {self.results['weighted_precision']:.4f}")
        report.append(f"  Weighted Recall: {self.results['weighted_recall']:.4f}")
        report.append(f"  Weighted F1-Score: {self.results['weighted_f1']:.4f}")
        report.append(f"  Macro F1-Score: {self.results['macro_f1']:.4f}")
        report.append(f"  ROC-AUC (OvR): {self.results['roc_auc_ovr']:.4f}")
        report.append("")
        
        # Per-class metrics
        report.append("PER-CLASS METRICS:")
        report.append("-" * 80)
        report.append(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        report.append("-" * 80)
        
        for label in self.class_labels:
            metrics = self.results['per_class'][label]
            report.append(
                f"{label:<20} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} "
                f"{metrics['f1']:<12.4f} "
                f"{metrics['support']:<10d}"
            )
        
        report.append("-" * 80)
        report.append("")
        
        # Confidence statistics
        conf_stats = self.get_prediction_confidence_distribution()
        report.append("CONFIDENCE STATISTICS:")
        report.append(f"  Mean: {conf_stats['mean']:.4f}")
        report.append(f"  Std: {conf_stats['std']:.4f}")
        report.append(f"  Median: {conf_stats['median']:.4f}")
        report.append(f"  Q25-Q75: [{conf_stats['q25']:.4f}, {conf_stats['q75']:.4f}]")
        report.append("")
        
        report.append("Confidence Distribution:")
        for range_str, count in conf_stats['distribution'].items():
            report.append(f"  {range_str}: {count} samples")
        
        # Save report
        report_text = '\n'.join(report)
        with open(CLASSIFICATION_REPORT_PATH, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        logger.info(f"Report saved to {CLASSIFICATION_REPORT_PATH}")
        
        return report_text


if __name__ == "__main__":
    logger.info("Evaluation module ready for use")
    print("Use this module to evaluate trained models")
