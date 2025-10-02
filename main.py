import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import torch
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Import all modules
from config import *
from data_preprocessing import DataLoader, TextPreprocessor
from emotion_mapping import EmotionMapper, process_emotions, augment_emotion_data
from dataset_creation import DatasetBuilder
from model_training import create_model, ModelTrainer
from evaluation import ModelEvaluator
from inference import HealthRiskPredictor, interactive_prediction
from utils import (
    set_seed,
    create_directories,
    save_json,
    load_json,
    plot_training_history,
    plot_emotion_distribution,
    generate_summary_report,
    calculate_model_size,
    monitor_gpu_usage
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f'training_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BioBERTHealthRiskPipeline:
    """
    Complete pipeline for health risk detection using BioBERT
    """
    
    def __init__(self):
        """Initialize pipeline components"""
        self.start_time = time.time()
        
        # Set random seed for reproducibility
        set_seed(RANDOM_SEED)
        
        # Create necessary directories
        create_directories()
        
        # Log system information
        self._log_system_info()
        
        # Initialize components
        self.data_loader = None
        self.emotion_mapper = None
        self.dataset_builder = None
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.predictor = None
        
        # Data holders
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def _log_system_info(self):
        """Log system and configuration information"""
        logger.info("=" * 80)
        logger.info("BioBERT HEALTH RISK DETECTION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Start Time: {datetime.now()}")
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Device: {torch.cuda.get_device_name()}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Random Seed: {RANDOM_SEED}")
        logger.info("=" * 80)
    
    def prepare_data(self, data_path: Optional[Path] = None):
        """
        Step 1: Load and prepare data
        
        Args:
            data_path: Optional path to data file
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*60)
        
        # Initialize data loader
        self.data_loader = DataLoader(data_path or RAW_DATA_PATH)
        
        # Load raw data
        df = self.data_loader.load_data()
        logger.info(f"Loaded {len(df)} samples")
        
        # Preprocess text
        preprocessor = TextPreprocessor()
        df = preprocessor.preprocess_dataframe(df)
        logger.info(f"After preprocessing: {len(df)} samples")
        
        # Initialize emotion mapper
        self.emotion_mapper = EmotionMapper()
        
        # Map emotions to 12 categories
        df = self.emotion_mapper.map_dataframe(df)
        
        # Validate mapping
        stats = self.emotion_mapper.validate_mapping(df)
        logger.info(f"Emotion mapping statistics:")
        for category in CLASS_LABELS:
            count = stats['class_balance'][category]['count']
            pct = stats['class_balance'][category]['percentage']
            logger.info(f"  {category}: {count} ({pct:.1f}%)")
        
        # Apply augmentation if enabled
        if AUGMENTATION_ENABLED:
            original_size = len(df)
            df = augment_emotion_data(df, AUGMENTATION_FACTOR)
            logger.info(f"Data augmented from {original_size} to {len(df)} samples")
        
        # Balance dataset
        df = self.data_loader.balance_dataset(df, SAMPLES_PER_CLASS)
        
        # Split data
        self.train_df, self.val_df, self.test_df = self.data_loader.split_data(df)
        
        # Save processed data
        self.train_df.to_csv(TRAIN_DATA_PATH, index=False)
        self.val_df.to_csv(VAL_DATA_PATH, index=False)
        self.test_df.to_csv(TEST_DATA_PATH, index=False)
        
        logger.info(f"Data split complete:")
        logger.info(f"  Train: {len(self.train_df)} samples")
        logger.info(f"  Val: {len(self.val_df)} samples")
        logger.info(f"  Test: {len(self.test_df)} samples")
        
        # Plot emotion distribution
        plot_emotion_distribution(df)
        
    def create_datasets(self):
        """
        Step 2: Create PyTorch datasets and dataloaders
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 2: DATASET CREATION")
        logger.info("="*60)
        
        # Initialize dataset builder
        self.dataset_builder = DatasetBuilder(MODEL_NAME)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = \
            self.dataset_builder.prepare_data_splits(
                self.train_df,
                self.val_df,
                self.test_df,
                use_weighted_sampling=True
            )
        
        logger.info(f"Dataloaders created:")
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Val batches: {len(self.val_loader)}")
        logger.info(f"  Test batches: {len(self.test_loader)}")
        
        # Get class weights
        self.class_weights = self.dataset_builder.class_weights
        
    def train_model(self, resume_from: Optional[Path] = None):
        """
        Step 3: Train BioBERT model
        
        Args:
            resume_from: Optional checkpoint to resume from
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("="*60)
        
        # Create model
        self.model = create_model(
            model_name=MODEL_NAME,
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE
        )
        
        # Log model info
        model_info = calculate_model_size(self.model)
        logger.info(f"Model Information:")
        logger.info(f"  Total Parameters: {model_info['total_parameters']:,}")
        logger.info(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
        logger.info(f"  Model Size: {model_info['model_size_mb']:.2f} MB")
        
        # Initialize trainer
        self.trainer = ModelTrainer(
            model=self.model,
            device=DEVICE,
            class_weights=self.class_weights
        )
        
        # Load checkpoint if provided
        if resume_from and resume_from.exists():
            self.trainer.load_checkpoint(resume_from)
            logger.info(f"Resumed from checkpoint: {resume_from}")
        
        # Train model
        training_start = time.time()
        
        history = self.trainer.train(
            self.train_loader,
            self.val_loader,
            num_epochs=NUM_EPOCHS,
            save_dir=CHECKPOINT_DIR
        )
        
        training_time = time.time() - training_start
        logger.info(f"Training completed in {training_time/60:.2f} minutes")
        
        # Save training history
        save_json(history, TRAINING_HISTORY_PATH)
        
        # Plot training curves
        plot_training_history(history)
        
        # Load best model
        if self.trainer.best_model_state:
            self.model.load_state_dict(self.trainer.best_model_state)
            torch.save(self.trainer.best_model_state, BEST_MODEL_PATH / "best_model.pt")
            logger.info(f"Best model saved with F1: {self.trainer.best_val_f1:.4f}")
        
        return history, training_time
    
    def evaluate_model(self):
        """
        Step 4: Comprehensive model evaluation
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("="*60)
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(
            model=self.model,
            device=DEVICE,
            class_labels=CLASS_LABELS
        )
        
        # Evaluate on test set
        results = self.evaluator.evaluate(
            self.test_loader,
            save_predictions=SAVE_PREDICTIONS,
            return_attention=SAVE_ATTENTION_WEIGHTS
        )
        
        # Generate visualizations
        logger.info("Generating evaluation visualizations...")
        
        # Confusion matrix
        self.evaluator.plot_confusion_matrix(
            save_path=CONFUSION_MATRIX_PATH,
            normalize=True
        )
        
        # Per-class metrics
        self.evaluator.plot_per_class_metrics()
        
        # ROC curves
        self.evaluator.plot_roc_curves()
        
        # Error analysis
        error_df = self.evaluator.analyze_errors(top_n=20)
        if not error_df.empty:
            logger.info(f"Top misclassified examples saved to {ERROR_ANALYSIS_PATH}")
        
        # Generate classification report
        report = self.evaluator.generate_report()
        
        return results
    
    def setup_inference(self):
        """
        Step 5: Setup inference pipeline
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 5: INFERENCE SETUP")
        logger.info("="*60)
        
        # Initialize predictor
        self.predictor = HealthRiskPredictor(
            model_path=BEST_MODEL_PATH / "best_model.pt",
            device=DEVICE
        )
        
        # Test inference on sample texts
        sample_texts = [
            "I've been feeling really down and hopeless lately",
            "So excited about the new opportunity!",
            "The stress from work is overwhelming me",
            "Feeling grateful and happy with life",
            "I can't stop worrying about everything"
        ]
        
        logger.info("Testing inference on sample texts:")
        for text in sample_texts:
            result = self.predictor.predict_single(text)
            logger.info(f"  Text: '{text[:50]}...'")
            logger.info(f"    Predicted: {result.predicted_emotion} ({result.confidence:.2%})")
            logger.info(f"    Risk Level: {result.risk_level}")
        
        logger.info("Inference pipeline ready!")
    
    def generate_final_report(self, training_time: float, eval_results: Dict):
        """
        Generate comprehensive final report
        
        Args:
            training_time: Time taken for training
            eval_results: Evaluation results dictionary
        """
        logger.info("\n" + "="*60)
        logger.info("GENERATING FINAL REPORT")
        logger.info("="*60)
        
        # Generate summary report
        report_text = generate_summary_report(
            model_metrics=eval_results,
            training_time=training_time
        )
        
        # Additional statistics
        total_time = time.time() - self.start_time
        
        logger.info(f"\nPipeline Summary:")
        logger.info(f"  Total Execution Time: {total_time/60:.2f} minutes")
        logger.info(f"  Training Time: {training_time/60:.2f} minutes")
        logger.info(f"  Best Validation F1: {self.trainer.best_val_f1:.4f}")
        logger.info(f"  Test Accuracy: {eval_results['accuracy']:.4f}")
        logger.info(f"  Test F1 (Weighted): {eval_results['weighted_f1']:.4f}")
        
        # GPU usage
        gpu_info = monitor_gpu_usage()
        if gpu_info['gpu_available']:
            logger.info(f"\nGPU Usage:")
            logger.info(f"  Memory Allocated: {gpu_info['memory_allocated_gb']:.2f} GB")
            logger.info(f"  Max Memory Used: {gpu_info['max_memory_allocated_gb']:.2f} GB")
    
    def run_full_pipeline(self):
        """
        Run the complete training and evaluation pipeline
        """
        try:
            logger.info("Starting BioBERT Health Risk Detection Pipeline...")
            
            # Step 1: Data Preparation
            self.prepare_data()
            
            # Step 2: Dataset Creation
            self.create_datasets()
            
            # Step 3: Model Training
            history, training_time = self.train_model()
            
            # Step 4: Model Evaluation
            eval_results = self.evaluate_model()
            
            # Step 5: Inference Setup
            self.setup_inference()
            
            # Generate Final Report
            self.generate_final_report(training_time, eval_results)
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            return False
    
    def run_inference_only(self, model_path: Optional[Path] = None):
        """
        Run inference only (no training)
        
        Args:
            model_path: Path to trained model
        """
        logger.info("Running inference mode...")
        
        # Initialize predictor
        self.predictor = HealthRiskPredictor(
            model_path=model_path or BEST_MODEL_PATH / "best_model.pt",
            device=DEVICE
        )
        
        # Run interactive mode
        interactive_prediction()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="BioBERT Health Risk Detection Pipeline"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval', 'inference', 'full'],
        default='full',
        help='Pipeline mode: train, eval, inference, or full'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to input data CSV file'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = BioBERTHealthRiskPipeline()
    
    if args.mode == 'full':
        # Run complete pipeline
        success = pipeline.run_full_pipeline()
        
    elif args.mode == 'train':
        # Training only
        if args.data_path:
            pipeline.prepare_data(Path(args.data_path))
        else:
            # Load existing processed data
            pipeline.train_df = pd.read_csv(TRAIN_DATA_PATH)
            pipeline.val_df = pd.read_csv(VAL_DATA_PATH)
            pipeline.test_df = pd.read_csv(TEST_DATA_PATH)
        
        pipeline.create_datasets()
        
        resume_path = Path(args.resume) if args.resume else None
        pipeline.train_model(resume_from=resume_path)
        
    elif args.mode == 'eval':
        # Evaluation only
        if not args.model_path:
            logger.error("Model path required for evaluation mode")
            return
        
        # Load test data
        pipeline.test_df = pd.read_csv(TEST_DATA_PATH)
        
        # Create dataset
        pipeline.dataset_builder = DatasetBuilder(MODEL_NAME)
        test_dataset = pipeline.dataset_builder.create_dataset(pipeline.test_df)
        pipeline.test_loader = pipeline.dataset_builder.create_dataloader(
            test_dataset,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            return_text=True
        )
        
        # Load model
        pipeline.model = create_model()
        checkpoint = torch.load(Path(args.model_path), map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            pipeline.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            pipeline.model.load_state_dict(checkpoint)
        
        # Evaluate
        pipeline.evaluate_model()
        
    elif args.mode == 'inference':
        # Inference only
        model_path = Path(args.model_path) if args.model_path else None
        pipeline.run_inference_only(model_path)
    
    logger.info("\nPipeline execution complete!")


if __name__ == "__main__":
    main()
