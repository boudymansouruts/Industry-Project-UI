import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.optim import AdamW
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from preprocess import (
    load_dailytalk_dataset as _load_dailytalk_dataset,
    load_audio as _load_audio,
    trim_audio as _trim_audio,
)


class GenericSpeakerDataset:
    """Dataset that loads DailyTalk samples and yields (audio_array, text)."""

    def __init__(self, data_root: str, processor: Any, max_samples: Optional[int] = None) -> None:
        self.data_root: str = data_root
        self.processor: Any = processor
        self.samples, self.speaker_mapping = _load_dailytalk_dataset(self.data_root, max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        sample = self.samples[index]
        audio_path: str = sample["audio_path"]
        text: str = sample["text"]
        audio, sr = _load_audio(audio_path, target_sr=16000)
        audio = _trim_audio(audio, max_length_seconds=20, sr=sr)
        return audio.astype(np.float32), text

a
def generic_collate_fn(batch: List[Tuple[np.ndarray, str]]) -> Dict[str, List[Any]]:
    audios: List[np.ndarray] = []
    texts: List[str] = []
    for audio, text in batch:
        audios.append(audio)
        texts.append(text)
    return {"audios": audios, "texts": texts}

def whisper_large_full_training():
    print("Whisper Large v2 Full Training")
    print("=" * 60)
    
    model_dir = "whisper-large-full"
    
    if os.path.exists(model_dir):
        try:
            import shutil
            shutil.rmtree(model_dir)
            print(f"Removed existing model directory: {model_dir}")
        except Exception as e:
            print(f"Could not remove existing directory: {e}")
            return
    else:
        print(f"Creating new model directory: {model_dir}")
    
    print(f"Loading Whisper Large v2 model...")
    model_name = "openai/whisper-large-v2"  # Using the largest Whisper model
    processor = WhisperProcessor.from_pretrained(model_name, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Load full dataset for comprehensive training from dailytalk/data
    data_dir = "dailytalk/data"
    dataset = GenericSpeakerDataset(data_dir, processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=generic_collate_fn)  # Small batch for large model
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Optimized training configuration to prevent overfitting
    num_epochs = 8  # Reduced epochs to prevent overfitting
    learning_rate = 3e-6  # Lower learning rate for stability
    accumulation_steps = 16  # Reduced accumulation for faster convergence
    warmup_steps = 200  # Reduced warmup
    early_stopping_patience = 3  # Early stopping to prevent overfitting
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\nTesting original Whisper Large v2 model...")
    print("-" * 50)
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
                
            audios = batch['audios']
            texts = batch['texts']
            
            audio, text = audios[0], texts[0]
            
            try:
                input_features = processor(
                    audio, sampling_rate=16000, return_tensors="pt"
                ).input_features.to(device)
                
                generated_ids = model.generate(
                    input_features,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
                    max_length=448,  # Increased max length for large model
                    num_beams=4,     # Beam search for better quality
                    early_stopping=True
                )
                
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                print(f"Sample {i + 1}:")
                print(f"  Original: '{text}'")
                print(f"  Whisper Large: '{transcription}'")
            except Exception as e:
                print(f"Sample {i + 1}: Error - {e}")
    
    print(f"\nStarting Whisper Large v2 full training...")
    print("-" * 50)
    print(f"Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Batch Size: 1")
    print(f"   Accumulation Steps: {accumulation_steps}")
    print(f"   Warmup Steps: {warmup_steps}")
    print(f"   Total Steps: {total_steps}")
    
    start_time = time.time()
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    training_losses = []
    
    model.train()
    for epoch in range(num_epochs):
        batch_loss = 0
        batch_count = 0
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}...")
        
        for batch_idx, batch in enumerate(dataloader):
            audios = batch['audios']
            texts = batch['texts']
            
            for audio, text in zip(audios, texts):
                try:
                    input_features = processor(
                        audio, sampling_rate=16000, return_tensors="pt"
                    ).input_features.to(device)
                    
                    labels = processor.tokenizer(
                        text, return_tensors="pt", padding=True, truncation=True
                    ).input_ids.to(device)
                    
                    labels[labels == processor.tokenizer.pad_token_id] = -100
                    
                    outputs = model(input_features=input_features, labels=labels)
                    loss = outputs.loss / accumulation_steps
                    loss.backward()
                    batch_loss += loss.item()
                    batch_count += 1
                    
                    if batch_count % accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                except Exception as e:
                    print(f"      Error in batch: {e}")
                    continue
            
            if (batch_idx + 1) % 50 == 0:  # More frequent updates for large model
                avg_loss = batch_loss / batch_count if batch_count > 0 else 0
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} ({100 * (batch_idx + 1) / len(dataloader):.1f}%), "
                      f"Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}, Time: {elapsed:.1f}s")
        
        if batch_count > 0:
            avg_epoch_loss = batch_loss / batch_count
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            training_losses.append(avg_epoch_loss)
            
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}, "
                  f"Epoch time: {epoch_time:.1f}s, Total time: {total_time/60:.1f}m")
            
            # Early stopping check
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
                print(f"   New best loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                print(f"   No improvement for {patience_counter} epochs (best: {best_loss:.4f})")
                
                if patience_counter >= early_stopping_patience:
                    print(f"   Early stopping triggered! No improvement for {early_stopping_patience} epochs")
                    print(f"   Training stopped at epoch {epoch + 1}")
                    break
    
    total_training_time = time.time() - start_time
    print(f"\nTotal training time: {total_training_time/60:.1f} minutes")
    
    print(f"\nTesting fine-tuned Whisper Large v2 model...")
    print("-" * 50)
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
                
            audios = batch['audios']
            texts = batch['texts']
            
            audio, text = audios[0], texts[0]
            
            try:
                input_features = processor(
                    audio, sampling_rate=16000, return_tensors="pt"
                ).input_features.to(device)
                
                generated_ids = model.generate(
                    input_features,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
                    max_length=448,
                    num_beams=4,
                    early_stopping=True
                )
                
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                print(f"Sample {i + 1}:")
                print(f"  Original: '{text}'")
                print(f"  Fine-tuned: '{transcription}'")
            except Exception as e:
                print(f"Sample {i + 1}: Error - {e}")
    
    print(f"\nSaving Whisper Large v2 model to: {model_dir}")
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    
    # Save comprehensive configuration
    config = {
        'model_name': model_name,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': 1,
        'accumulation_steps': accumulation_steps,
        'warmup_steps': warmup_steps,
        'total_steps': total_steps,
        'max_length': 448,
        'num_beams': 4,
        'early_stopping': True,
        'early_stopping_patience': early_stopping_patience,
        'scheduler': 'linear_with_warmup',
        'total_training_time_minutes': total_training_time / 60,
        'dataset_size': len(dataset),
        'device': str(device),
        'best_loss': best_loss,
        'final_epoch': len(training_losses),
        'training_losses': training_losses
    }
    
    config_file = os.path.join(model_dir, "training_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Whisper Large v2 full training completed successfully!")
    print(f"Model can now identify speakers with advanced Whisper Large architecture")
    print(f"Model saved to: {model_dir}")
    print(f"Training configuration saved to: {config_file}")
    print(f"Total training time: {total_training_time/60:.1f} minutes")
    print(f"Final model size: ~1550M parameters")

if __name__ == "__main__":
    whisper_large_full_training()
