#!/usr/bin/env python3
"""
Fine-tune Whisper Large (openai/whisper-large-v2) on local DailyTalk-style data.

Expects dataset under dailytalk/data/** with matching .wav and .txt files.
Saves finetuned model and processor to models/whisper-large-finetuned.
"""

import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
import numpy as np
import soundfile as sf
import librosa

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Trainer,
    TrainingArguments,
)


def find_audio_text_pairs(root: Path) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for wav_path in root.rglob("*.wav"):
        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists():
            try:
                text = txt_path.read_text(encoding="utf-8").strip()
                if len(text) < 2:
                    continue
                pairs.append({"audio_path": wav_path, "text": text})
            except Exception:
                continue
    return pairs


def load_audio_16k(audio_path: Path) -> np.ndarray:
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = librosa.util.normalize(audio)
    return audio.astype(np.float32)


@dataclass
class WhisperCollator:
    processor: WhisperProcessor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [b["input_features"] for b in batch]
        labels = [b["labels"] for b in batch]
        # Pad inputs
        input_features = torch.stack(input_features)
        # Pad labels with -100
        max_len = max(l.size(0) for l in labels)
        padded = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            padded[i, : l.size(0)] = l
        return {"input_features": input_features, "labels": padded}


class WhisperLocalDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Dict[str, Any]], processor: WhisperProcessor, language: str = "en"):
        self.items = items
        self.processor = processor
        self.language = language

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        audio = load_audio_16k(item["audio_path"])
        # Input features
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt",
            padding="max_length", max_length=self.processor.feature_extractor.n_samples
        ).input_features.squeeze(0)

        # Labels
        with self.processor.as_target_processor():
            labels = self.processor.tokenizer(
                item["text"], return_tensors="pt"
            ).input_ids.squeeze(0)

        return {"input_features": input_features, "labels": labels}


def main() -> None:
    data_root = Path("dailytalk/data")
    assert data_root.exists(), f"Dataset not found at {data_root}"

    base_model = "openai/whisper-large-v2"
    output_dir = Path("models/whisper-large-finetuned")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning dataset for audio-text pairs...")
    pairs = find_audio_text_pairs(data_root)
    if len(pairs) < 20:
        raise RuntimeError("Not enough pairs found to train. Expected >= 20.")
    random.shuffle(pairs)

    # Split train/val
    split = int(0.9 * len(pairs))
    train_items = pairs[:split]
    val_items = pairs[split:]
    print(f"Found {len(pairs)} pairs (train={len(train_items)}, val={len(val_items)})")

    print("Loading processor and model...")
    processor = WhisperProcessor.from_pretrained(base_model, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(base_model)

    # Set generation config defaults (language/task) for the fine-tuned model
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.config.suppress_tokens = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = WhisperLocalDataset(train_items, processor)
    val_ds = WhisperLocalDataset(val_items, processor)

    collator = WhisperCollator(processor)

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=2,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving fine-tuned model and processor...")
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    print("Done. To use this model, update model_config.json to:\n"
          "  {\n    \"current_model\": \"whisper-large-finetuned\",\n    \"models\": { ... , \n      \"whisper-large-finetuned\": { \n        \"path\": \"models/whisper-large-finetuned\"\n      }\n    }\n  }")


if __name__ == "__main__":
    main()


