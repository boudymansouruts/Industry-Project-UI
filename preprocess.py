#!/usr/bin/env python3
"""
Data preprocessing for DailyTalk dataset - Whisper fine-tuning
"""

import os
import json
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
import argparse

class DailyTalkPreprocessor:
    """
    Preprocessor for DailyTalk dataset for Whisper fine-tuning
    """
    
    def __init__(self, data_dir: str, output_dir: str, target_sr: int = 16000):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.metadata_path = self.data_dir.parent / "metadata.json"
        self.metadata = self._load_metadata()
        
        print(f"📁 Input directory: {self.data_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎵 Target sample rate: {self.target_sr} Hz")
    
    def _load_metadata(self):
        """Load metadata from JSON file"""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"📖 Loaded metadata for {len(metadata)} dialogs")
        return metadata
    
    def _preprocess_audio(self, audio_path: str):
        """
        Preprocess a single audio file
        """
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            
            # Convert to float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Remove silence from beginning and end
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            return audio
            
        except Exception as e:
            print(f"⚠️ Error processing {audio_path}: {e}")
            return None
    
    def _is_valid_sample(self, text: str, audio: np.ndarray):
        """
        Check if sample is valid for training
        """
        # Check text
        if not text or len(text.strip()) < 3:
            return False
        
        # Check audio
        if audio is None or len(audio) < 0.1 * self.target_sr:  # At least 100ms
            return False
        
        # Check audio duration (not too long)
        if len(audio) > 30 * self.target_sr:  # Max 30 seconds
            return False
        
        return True
    
    def preprocess_dataset(self, max_samples: int = None, split_ratios: tuple = (0.8, 0.1, 0.1)):
        """
        Preprocess the entire dataset
        """
        print("🔄 Starting dataset preprocessing...")
        
        # Collect all valid samples
        samples = []
        total_processed = 0
        total_valid = 0
        
        for dialog_id in tqdm(sorted(self.metadata.keys(), key=int), desc="Processing dialogs"):
            if max_samples and total_valid >= max_samples:
                break
            
            dialog_data = self.metadata[dialog_id]
            dialog_dir = self.data_dir / dialog_id
            
            if not dialog_dir.exists():
                continue
            
            for utterance_id in sorted(dialog_data.keys(), key=int):
                if max_samples and total_valid >= max_samples:
                    break
                
                utterance_data = dialog_data[utterance_id]
                speaker = utterance_data['speaker']
                text = utterance_data['text'].strip()
                
                # Construct audio file path
                audio_file = f"{utterance_id}_{speaker}_d{dialog_id}.wav"
                audio_path = dialog_dir / audio_file
                
                if not audio_path.exists():
                    continue
                
                # Preprocess audio
                audio = self._preprocess_audio(str(audio_path))
                total_processed += 1
                
                # Validate sample
                if self._is_valid_sample(text, audio):
                    # Save preprocessed audio
                    output_audio_path = self.output_dir / f"{dialog_id}_{utterance_id}_{speaker}.wav"
                    sf.write(output_audio_path, audio, self.target_sr)
                    
                    samples.append({
                        'audio_path': str(output_audio_path),
                        'text': text,
                        'speaker': speaker,
                        'dialog_id': dialog_id,
                        'utterance_id': utterance_id,
                        'duration': len(audio) / self.target_sr,
                        'original_path': str(audio_path)
                    })
                    total_valid += 1
        
        print(f"📊 Preprocessing complete:")
        print(f"   Total processed: {total_processed}")
        print(f"   Valid samples: {total_valid}")
        print(f"   Success rate: {total_valid/total_processed*100:.1f}%")
        
        # Split dataset
        train_samples, val_samples, test_samples = self._split_dataset(samples, split_ratios)
        
        # Save splits
        self._save_split(train_samples, "train")
        self._save_split(val_samples, "val") 
        self._save_split(test_samples, "test")
        
        # Save complete metadata
        self._save_metadata(samples)
        
        return samples
    
    def _split_dataset(self, samples: list, split_ratios: tuple):
        """
        Split dataset into train/val/test
        """
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(samples)
        
        n_samples = len(samples)
        train_size = int(n_samples * split_ratios[0])
        val_size = int(n_samples * split_ratios[1])
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        print(f"📈 Dataset split:")
        print(f"   Train: {len(train_samples)} samples ({len(train_samples)/n_samples*100:.1f}%)")
        print(f"   Val: {len(val_samples)} samples ({len(val_samples)/n_samples*100:.1f}%)")
        print(f"   Test: {len(test_samples)} samples ({len(test_samples)/n_samples*100:.1f}%)")
        
        return train_samples, val_samples, test_samples
    
    def _save_split(self, samples: list, split_name: str):
        """
        Save a dataset split
        """
        split_file = self.output_dir / f"{split_name}.json"
        
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {split_name} split: {split_file}")
    
    def _save_metadata(self, samples: list):
        """
        Save complete metadata
        """
        # Calculate statistics
        durations = [sample['duration'] for sample in samples]
        text_lengths = [len(sample['text']) for sample in samples]
        
        metadata = {
            'total_samples': len(samples),
            'total_duration': sum(durations),
            'avg_duration': np.mean(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_text_length': np.mean(text_lengths),
            'min_text_length': min(text_lengths),
            'max_text_length': max(text_lengths),
            'sample_rate': self.target_sr,
            'samples': samples
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Dataset statistics:")
        print(f"   Total duration: {metadata['total_duration']:.1f}s ({metadata['total_duration']/60:.1f}m)")
        print(f"   Avg duration: {metadata['avg_duration']:.2f}s")
        print(f"   Avg text length: {metadata['avg_text_length']:.1f} chars")
        print(f"💾 Saved metadata: {metadata_file}")

def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description="Preprocess DailyTalk dataset for Whisper fine-tuning")
    parser.add_argument("--data_dir", default="dailytalk/data", help="Path to DailyTalk data directory")
    parser.add_argument("--output_dir", default="preprocessed_whisper", help="Output directory for preprocessed data")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate")
    
    args = parser.parse_args()
    
    print("🎵 DailyTalk Dataset Preprocessing for Whisper")
    print("="*60)
    
    # Check input directory
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    # Initialize preprocessor
    preprocessor = DailyTalkPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_sr=args.sample_rate
    )
    
    # Process dataset
    try:
        samples = preprocessor.preprocess_dataset(max_samples=args.max_samples)
        print(f"✅ Preprocessing completed successfully!")
        print(f"📁 Preprocessed data saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
