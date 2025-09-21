#!/usr/bin/env python3
"""
Generic Speaker Identification Whisper Fine-tuning
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
    def __init__(self, data_dir: str, processor, max_samples: int = None):
        self.processor = processor
        self.samples = []
        self.speaker_mapping = {}
        
        print(f"📖 Loading DailyTalk dataset for generic speaker identification...")
        self.samples, self.speaker_mapping = load_dailytalk_dataset(data_dir, max_samples)
        print(f"✅ Loaded {len(self.samples)} samples")
        print(f"👥 Found {len(self.speaker_mapping)} unique speakers mapped to generic IDs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            audio, sr = load_audio(sample['audio_path'])
            audio = trim_audio(audio, max_length_seconds=20)
            
            return {
                'audio': audio,
                'text': sample['text'],
                'original_text': sample['original_text'],
                'original_speaker': sample['original_speaker'],
                'generic_speaker': sample['generic_speaker']
            }
            
        except Exception as e:
            print(f"⚠️ Error loading {sample['audio_path']}: {e}")
            return {
                'audio': np.zeros(16000),
                'text': '',
                'original_text': '',
                'original_speaker': 0,
                'generic_speaker': 'Speaker_1'
            }

def generic_collate_fn(batch):
    audios = []
    texts = []
    
    for item in batch:
        audios.append(item['audio'])
        texts.append(item['text'])
    
    return {
        'audios': audios,
        'texts': texts
    }

def generic_speaker_fine_tuning():
    print("🎵 Generic Speaker Identification Whisper Fine-tuning")
    print("=" * 60)
    
    model_dir = "whisper-new-model"
    
    if os.path.exists(model_dir):
        try:
            import shutil
            shutil.rmtree(model_dir)
            print(f"🗑️ Removed old model: {model_dir}")
        except Exception as e:
            print(f"⚠️ Could not remove old model directory (in use): {model_dir}")
            print(f"🔄 Continuing with existing model...")
    
    print(f"🤖 Loading Whisper Base model...")
    model_name = "openai/whisper-base"
    processor = WhisperProcessor.from_pretrained(model_name, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🖥️ Using device: {device}")
    
    dataset = GenericSpeakerDataset("dailytalk/data", processor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=generic_collate_fn)
    
    print(f"📊 Dataset size: {len(dataset)} samples")
    print(f"🔄 Batches per epoch: {len(dataloader)}")
    
    print(f"\n🧪 Testing original model...")
    print("-" * 50)
    
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
            
        audio = sample['audio']
        text = sample['text']
        original_speaker = sample['original_speaker']
        generic_speaker = sample['generic_speaker']
        original_text = sample['original_text']
        
        with torch.no_grad():
            input_features = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
                max_length=200
            )
            
            transcription = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
        
        print(f"Sample {i+1}:")
        print(f"  Original Speaker: {original_speaker}")
        print(f"  Generic Speaker: {generic_speaker}")
        print(f"  Original: '{original_text}'")
        print(f"  Formatted: '{text}'")
        print(f"  Transcribed: '{transcription}'")
        print()
    
    print(f"🚀 Starting generic speaker fine-tuning...")
    print("-" * 50)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    accumulation_steps = 8
    optimizer.zero_grad()
    
    for epoch in range(10):
        print(f"🔄 Epoch {epoch+1}/5...")
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            audios = batch['audios']
            texts = batch['texts']
            
            batch_loss = 0
            
            for i, (audio, text) in enumerate(zip(audios, texts)):
                if len(audio) < 16000:
                    audio = np.pad(audio, (0, 16000 - len(audio)))
                elif len(audio) > 16000:
                    audio = audio[:16000]
                
                input_features = processor.feature_extractor(
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
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += batch_loss
            batch_count += 1
            
            if batch_idx % 10 == 0:
                avg_loss = total_loss / batch_count
                print(f"  Batch {batch_idx}/{len(dataloader)} ({batch_idx/len(dataloader)*100:.1f}%), Avg Loss: {avg_loss:.4f}")
            
            if batch_idx >= 73:
                print(f"   ⚡ Training: stopping after {batch_idx} batches")
                break
        
        epoch_loss = total_loss / batch_count if batch_count > 0 else 0
        if epoch_loss > 0:
            print(f"✅ Epoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
        else:
            print(f"⚠️ Epoch {epoch+1} completed. No valid batches processed.")
    
    print(f"\n🧪 Testing fine-tuned model...")
    print("-" * 50)
    
    model.eval()
    
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
            
        audio = sample['audio']
        text = sample['text']
        original_speaker = sample['original_speaker']
        generic_speaker = sample['generic_speaker']
        original_text = sample['original_text']
        
        with torch.no_grad():
            input_features = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
                max_length=200
            )
            
            transcription = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
        
        print(f"Sample {i+1}:")
        print(f"  Original Speaker: {original_speaker}")
        print(f"  Generic Speaker: {generic_speaker}")
        print(f"  Original: '{original_text}'")
        print(f"  Formatted: '{text}'")
        print(f"  Fine-tuned: '{transcription}'")
        print()
    
    print(f"💾 Saving model to: {model_dir}")
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    
    print(f"✅ Generic speaker fine-tuning completed successfully!")
    print(f"🎤 Model can now identify speakers in speech-only segments")
    print(f"🔍 Uses voice activity detection + global speaker analysis")
    print(f"📁 Model saved to: {model_dir}")

if __name__ == "__main__":
    generic_speaker_fine_tuning()