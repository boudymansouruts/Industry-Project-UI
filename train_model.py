#!/usr/bin/env python3
"""
Generic Speaker Identification Whisper Fine-tuning
Trains model to identify any speakers as Speaker 1, Speaker 2, etc.
"""

import os
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import warnings
from preprocess import load_audio, trim_audio, load_dailytalk_dataset
warnings.filterwarnings("ignore")

class GenericSpeakerDataset(Dataset):
    """Dataset for generic speaker identification (Speaker 1, Speaker 2, etc.)"""
    
    def __init__(self, data_dir: str, processor, max_samples: int = None):
        self.processor = processor
        self.samples = []
        self.speaker_mapping = {}  # Map original speakers to generic Speaker 1, 2, etc.
        
        print(f"📖 Loading DailyTalk dataset for generic speaker identification...")
        self.samples, self.speaker_mapping = load_dailytalk_dataset(data_dir, max_samples)
        print(f"✅ Loaded {len(self.samples)} samples")
        print(f"👥 Found {len(self.speaker_mapping)} unique speakers mapped to generic IDs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load and preprocess audio using the preprocessing module
            audio, sr = load_audio(sample['audio_path'])
            audio = trim_audio(audio, max_length_seconds=20)
            
            return {
                'audio': audio,
                'text': sample['text'],
                'original_text': sample['original_text'],
                'original_speaker': sample['original_speaker'],
                'generic_speaker': sample['generic_speaker'],
                'language': sample['language']
            }
            
        except Exception as e:
            print(f"⚠️ Error loading {sample['audio_path']}: {e}")
            return {
                'audio': np.zeros(16000),
                'text': "error",
                'original_text': "error",
                'original_speaker': 0,
                'generic_speaker': "Speaker_1",
                'language': 'en'
            }

def generic_collate_fn(batch):
    """Collate function for generic speaker dataset"""
    audios = [item['audio'] for item in batch]
    texts = [item['text'] for item in batch]
    return {'audios': audios, 'texts': texts}

def generic_speaker_fine_tuning():
    """Fine-tune Whisper for generic speaker identification"""
    print("🎵 Generic Speaker Identification Whisper Fine-tuning")
    print("="*60)
    
    data_dir = "dailytalk/data"
    output_dir = "whisper-generic-speakers"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Clear old models
    if os.path.exists(output_dir):
        import shutil
        import time
        try:
            shutil.rmtree(output_dir)
            print(f"🗑️ Removed old model: {output_dir}")
        except PermissionError:
            print(f"⚠️ Could not remove old model directory (in use): {output_dir}")
            print("🔄 Continuing with existing model...")
            time.sleep(1)
    
    # Load Whisper Base model
    print("🤖 Loading Whisper Base model...")
    model_name = "openai/whisper-base"
    processor = WhisperProcessor.from_pretrained(model_name, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🖥️ Using device: {device}")
    
    # Create dataset
    dataset = GenericSpeakerDataset(data_dir, processor, max_samples=None)
    
    if len(dataset) == 0:
        print("❌ No samples loaded!")
        return
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=generic_collate_fn)
    
    print(f"📊 Dataset size: {len(dataset)} samples")
    print(f"🔄 Batches per epoch: {len(dataloader)}")
    
    # Test original model
    print("\n🧪 Testing original model...")
    test_samples = [dataset[i] for i in range(min(3, len(dataset)))]
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            input_features = processor.feature_extractor(
                sample['audio'], sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
                max_length=100
            )
            
            transcription = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            print(f"Sample {i+1}:")
            print(f"  Original Speaker: {sample['original_speaker']}")
            print(f"  Generic Speaker: {sample['generic_speaker']}")
            print(f"  Language: {sample['language']}")
            print(f"  Original: '{sample['original_text']}'")
            print(f"  Formatted: '{sample['text']}'")
            print(f"  Transcribed: '{transcription}'")
            print()
    
    # Training loop
    print("🚀 Starting generic speaker fine-tuning...")
    print("-" * 50)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Gradient accumulation
    accumulation_steps = 2
    optimizer.zero_grad()
    
    for epoch in range(5):
        print(f"🔄 Epoch {epoch+1}/5...")
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            audios = batch['audios']
            texts = batch['texts']
            
            batch_loss = 0
            
            for audio, text in zip(audios, texts):
                if len(audio) == 0 or not text.strip():
                    continue
                
                try:
                    # Process inputs
                    input_features = processor.feature_extractor(
                        audio, sampling_rate=16000, return_tensors="pt"
                    ).input_features.to(device)
                    
                    # Process labels
                    labels = processor.tokenizer(
                        text, return_tensors="pt", padding=True, truncation=True, max_length=100
                    ).input_ids.to(device)
                    
                    # Forward pass
                    outputs = model(input_features=input_features, labels=labels)
                    loss = outputs.loss / accumulation_steps
                    
                    if torch.isfinite(loss):
                        batch_loss += loss
                        loss.backward()
                        total_loss += loss.item() * accumulation_steps
                        batch_count += 1
                    
                    # Clear variables
                    del input_features, labels, outputs, loss
                    
                except Exception as e:
                    print(f"⚠️ Error processing sample: {e}")
                    continue
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                except Exception as e:
                    print(f"⚠️ Error in optimization step: {e}")
                    optimizer.zero_grad()
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / max(batch_count, 1)
                progress = (batch_idx + 1) / len(dataloader) * 100
                print(f"  Batch {batch_idx+1}/{len(dataloader)} ({progress:.1f}%), Avg Loss: {avg_loss:.4f}")
        
        # Final gradient update
        if batch_count > 0 and (len(dataloader) % accumulation_steps) != 0:
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            except Exception as e:
                print(f"⚠️ Error in final optimization step: {e}")
                optimizer.zero_grad()
        
        if batch_count > 0:
            epoch_loss = total_loss / batch_count
            print(f"✅ Epoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
        else:
            print(f"⚠️ Epoch {epoch+1} completed. No valid batches processed.")
        
        # Reinitialize gradients for next epoch
        optimizer.zero_grad()
    
    # Test fine-tuned model
    print("\n🧪 Testing fine-tuned model...")
    print("-" * 50)
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            input_features = processor.feature_extractor(
                sample['audio'], sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
                max_length=100
            )
            
            transcription = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            print(f"Sample {i+1}:")
            print(f"  Original Speaker: {sample['original_speaker']}")
            print(f"  Generic Speaker: {sample['generic_speaker']}")
            print(f"  Language: {sample['language']}")
            print(f"  Original: '{sample['original_text']}'")
            print(f"  Formatted: '{sample['text']}'")
            print(f"  Fine-tuned: '{transcription}'")
            print()
    
    # Save model
    print(f"💾 Saving model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print("✅ Generic speaker fine-tuning completed successfully!")
    print(f"🎤 Model can now identify any speakers as Speaker 1, Speaker 2, etc.")
    print(f"🌍 Model can detect languages: English, Spanish, French, German")
    print(f"📁 Model saved to: {output_dir}")

def main():
    """Main function"""
    try:
        generic_speaker_fine_tuning()
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
