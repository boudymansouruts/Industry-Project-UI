#!/usr/bin/env python3
"""
Train a company-specific sentiment model on data/company_sentiment_examples.csv
Labels include: frustration, client_wants_to_leave, risk_issue, anger, urgency, escalation
"""

import os
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np


DATA_PATH = Path("data/company_sentiment_examples.csv")
OUTPUT_DIR = Path("models/company-sentiment")
MODEL_NAME = "roberta-large"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Basic sanity
    df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
    return df


def make_label_mapping(labels: List[str]) -> Dict[str, int]:
    classes = sorted(set(labels))
    return {cls: i for i, cls in enumerate(classes)}


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
    )


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}


def main():
    df = load_data()
    label2id = make_label_mapping(df.label.tolist())
    id2label = {v: k for k, v in label2id.items()}

    # Map labels to ids
    df["labels"] = df["label"].map(label2id)

    # Stratified split
    from sklearn.model_selection import train_test_split
    df_train, df_val = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df["labels"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ds_train = Dataset.from_pandas(df_train[["text", "labels"]].reset_index(drop=True))
    ds_val = Dataset.from_pandas(df_val[["text", "labels"]].reset_index(drop=True))

    ds_train = ds_train.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    ds_val = ds_val.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        evaluation_strategy="epoch" if len(df_val) > 0 else "no",
        save_strategy="epoch" if len(df_val) > 0 else "no",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=25,
        load_best_model_at_end=len(df_val) > 0,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=42,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    # Class weights
    import numpy as np
    class_counts = df_train["labels"].value_counts().sort_index().values.astype(float)
    inv_freq = 1.0 / np.maximum(class_counts, 1.0)
    weights = inv_freq / inv_freq.sum() * len(class_counts)

    from transformers import Trainer
    import torch
    import torch.nn as nn

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val if len(df_val) > 0 else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if len(df_val) > 0 else None,
    )

    trainer.train()

    # Save model and tokenizer
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save label mapping
    with open(OUTPUT_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print("Model saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()


