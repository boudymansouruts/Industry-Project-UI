#!/usr/bin/env python3
"""
Working Whisper fine-tuning on DailyTalk dataset
"""

import os
import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

class DailyTalkDataset(Dataset):
    """DailyTalk dataset for Whisper fine-tuning"""
    
    def __init__(self, data_dir: str, processor, max_samples: int = 3000):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.samples = []
        
        print(f"📖 Loading DailyTalk dataset from: {data_dir}")
        self._load_samples(max_samples)
        print(f"✅ Loaded {len(self.samples)} samples")
    
    def _load_samples(self, max_samples: int):
        """Load audio-text pairs from the dataset"""
        metadata_path = self.data_dir.parent / "metadata.json"
        
        if not metadata_path.exists():
            print(f"❌ Metadata file not found: {metadata_path}")
            return
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        count = 0
        # Process more dialogs to get 3000 samples
        for dialog_id in sorted(metadata.keys(), key=int)[:200]:  # First 200 dialogs
            if count >= max_samples:
                break
            
            dialog_data = metadata[dialog_id]
            dialog_dir = self.data_dir / dialog_id
            
            if not dialog_dir.exists():
                continue
            
            for utterance_id in sorted(dialog_data.keys(), key=int):
                if count >= max_samples:
                    break
                
                utterance_data = dialog_data[utterance_id]
                speaker = utterance_data['speaker']
                text = utterance_data['text'].strip()
                
                # Skip empty or very short text
                if not text or len(text) < 3:
                    continue
                
                audio_file = f"{utterance_id}_{speaker}_d{dialog_id}.wav"
                audio_path = dialog_dir / audio_file
                
                if audio_path.exists():
                    self.samples.append({
                        'audio_path': str(audio_path),
                        'text': text
                    })
                    count += 1
                    
                    if count % 50 == 0:
                        print(f"   Loaded {count} samples...")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load audio
            audio, sr = sf.read(sample['audio_path'])
            
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            audio = librosa.util.normalize(audio)
            
            # Limit length
            max_length = 20 * 16000  # 20 seconds
            if len(audio) > max_length:
                audio = audio[:max_length]
            
            return {
                'audio': audio,
                'text': sample['text'],
                'path': sample['audio_path']
            }
            
        except Exception as e:
            print(f"⚠️ Error loading {sample['audio_path']}: {e}")
            return {
                'audio': np.zeros(16000),
                'text': "error",
                'path': sample['audio_path']
            }

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    audios = [item['audio'] for item in batch]
    texts = [item['text'] for item in batch]
    paths = [item['path'] for item in batch]
    
    return {
        'audios': audios,
        'texts': texts,
        'paths': paths
    }

def simple_fine_tuning():
    """Simple fine-tuning approach"""
    print("🎵 Simple Whisper Fine-tuning on DailyTalk")
    print("="*60)
    
    data_dir = "dailytalk/data"
    output_dir = "whisper-dailytalk-3k"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Load model and processor
    print("🤖 Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    
    # Configure model
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    
    # Create dataset
    dataset = DailyTalkDataset(data_dir, processor, max_samples=3000)
    
    if len(dataset) == 0:
        print("❌ No samples loaded!")
        return
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # Setup training
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    print(f"📊 Dataset size: {len(dataset)} samples")
    print(f"🔄 Batches per epoch: {len(dataloader)}")
    
    # Test original model first
    print("\n🧪 Testing original model...")
    test_samples = [dataset[i] for i in range(min(3, len(dataset)))]
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            input_features = processor.feature_extractor(
                sample['audio'], sampling_rate=16000, return_tensors="pt"
            ).input_features
            
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
                max_length=50
            )
            
            transcription = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            print(f"Sample {i+1}:")
            print(f"  Original: '{sample['text']}'")
            print(f"  Transcribed: '{transcription}'")
            print()
    
    # Training loop
    print("🚀 Starting fine-tuning...")
    print("-" * 40)
    
    model.train()
    epoch_losses = []
    
    for epoch in range(3):  # 3 epochs for better training
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            audios = batch['audios']
            texts = batch['texts']
            
            batch_loss = 0
            
            for audio, text in zip(audios, texts):
                if len(audio) == 0 or not text.strip():
                    continue
                
                # Process inputs
                input_features = processor.feature_extractor(
                    audio, sampling_rate=16000, return_tensors="pt"
                ).input_features
                
                # Process labels with proper format
                labels = processor.tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True, max_length=50
                ).input_ids
                
                # Forward pass
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss
                
                if torch.isfinite(loss):
                    batch_loss += loss
                
            if batch_loss > 0:
                # Backward pass
                optimizer.zero_grad()
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                total_loss += batch_loss.item()
                batch_count += 1
                
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = batch_loss.item() / len(audios)
                    print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
        
        if batch_count > 0:
            epoch_loss = total_loss / batch_count
            epoch_losses.append(epoch_loss)
            print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} completed. No valid batches processed.")
    
    # Test fine-tuned model
    print("\n🧪 Testing fine-tuned model...")
    print("-" * 40)
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            input_features = processor.feature_extractor(
                sample['audio'], sampling_rate=16000, return_tensors="pt"
            ).input_features
            
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
                max_length=50
            )
            
            transcription = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            print(f"Sample {i+1}:")
            print(f"  Original: '{sample['text']}'")
            print(f"  Fine-tuned: '{transcription}'")
            print()
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"💾 Model saved to: {output_dir}")
    print("✅ Fine-tuning completed!")
    
    # Test on external audio files
    test_on_external_files(output_dir)

def test_on_external_files(model_dir):
    """Test fine-tuned model on external audio files"""
    print("\n🎤 Testing on external audio files...")
    print("-" * 40)
    
    try:
        processor = WhisperProcessor.from_pretrained(model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        model.eval()
        
        test_files = ["091419-o-832-853.wav", "091452-i-834-836.wav"]
        
        for audio_file in test_files:
            if os.path.exists(audio_file):
                print(f"\n📁 File: {audio_file}")
                
                # Load audio
                audio, sr = sf.read(audio_file)
                if len(audio.shape) > 1:
                    audio = audio[:, 0]
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                
                # Transcribe
                input_features = processor.feature_extractor(
                    audio, sampling_rate=16000, return_tensors="pt"
                ).input_features
                
                with torch.no_grad():
                    predicted_ids = model.generate(
                        input_features,
                        forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
                        max_length=200
                    )
                
                transcription = processor.tokenizer.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]
                
                print(f"📝 Fine-tuned Transcription: {transcription}")
                
    except Exception as e:
        print(f"❌ Error testing external files: {e}")

def main():
    """Main function"""
    try:
        simple_fine_tuning()
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
