#!/usr/bin/env python3
"""
Train a sentiment analysis model using mental health data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import torch
import json
import os
from datetime import datetime

def load_and_preprocess_data(csv_path):
    """Load and preprocess the mental health data for sentiment analysis"""
    print("📊 Loading mental health data...")
    
    # Load the CSV data
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} records")
    
    # Create sentiment labels based on mental health condition and stress level
    def create_sentiment_label(row):
        """Create sentiment labels based on mental health and stress indicators"""
        # Combine mental health condition and stress level for sentiment
        mental_health = row['Mental_Health_Condition']
        stress_level = row['Stress_Level']
        severity = row['Severity']
        
        # Positive: No mental health issues + Low stress
        if mental_health == 'No' and stress_level == 'Low':
            return 'positive'
        
        # Negative: Mental health issues + High stress + High severity
        elif mental_health == 'Yes' and stress_level == 'High' and severity in ['High', 'Medium']:
            return 'negative'
        
        # Neutral: Everything else
        else:
            return 'neutral'
    
    # Apply sentiment labeling
    df['sentiment'] = df.apply(create_sentiment_label, axis=1)
    
    # Create text features for sentiment analysis
    def create_text_description(row):
        """Create text descriptions from the data for sentiment analysis"""
        text_parts = []
        
        # Basic demographics
        text_parts.append(f"I am a {row['Age']} year old {row['Gender']}.")
        text_parts.append(f"I work in {row['Occupation']}.")
        
        # Mental health and stress
        if row['Mental_Health_Condition'] == 'Yes':
            text_parts.append(f"I have mental health conditions with {row['Severity']} severity.")
        else:
            text_parts.append("I do not have mental health conditions.")
        
        text_parts.append(f"My stress level is {row['Stress_Level']}.")
        
        # Lifestyle factors
        text_parts.append(f"I sleep {row['Sleep_Hours']} hours per night.")
        text_parts.append(f"I work {row['Work_Hours']} hours per week.")
        text_parts.append(f"I exercise {row['Physical_Activity_Hours']} hours per week.")
        
        # Health habits
        text_parts.append(f"My diet quality is {row['Diet_Quality']}.")
        text_parts.append(f"I am a {row['Smoking_Habit']}.")
        text_parts.append(f"I am a {row['Alcohol_Consumption']}.")
        
        if row['Medication_Usage'] == 'Yes':
            text_parts.append("I take medication.")
        else:
            text_parts.append("I do not take medication.")
        
        return " ".join(text_parts)
    
    # Create text descriptions
    df['text'] = df.apply(create_text_description, axis=1)
    
    # Filter out very short texts
    df = df[df['text'].str.len() > 50]
    
    print(f"📈 Sentiment distribution:")
    print(df['sentiment'].value_counts())
    
    return df[['text', 'sentiment']]

def prepare_dataset(df, tokenizer, max_length=512):
    """Prepare the dataset for training"""
    print("🔧 Preparing dataset for training...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['sentiment'])
    
    # Create label mapping
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # Tokenize texts
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    
    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    return train_dataset, val_dataset, label_mapping

def train_sentiment_model(train_dataset, val_dataset, label_mapping, model_name="distilbert-base-uncased"):
    """Train the sentiment analysis model"""
    print(f"🚀 Training sentiment model using {model_name}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_mapping)
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/sentiment-model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs/sentiment",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = (predictions == labels).astype(np.float32).mean().item()
        
        return {"accuracy": accuracy}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("🏋️ Starting training...")
    trainer.train()
    
    # Save the model
    model_save_path = "./models/sentiment-model"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Save label mapping
    with open(os.path.join(model_save_path, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=2)
    
    print(f"✅ Model saved to {model_save_path}")
    
    # Evaluate the model
    print("📊 Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    return model_save_path, eval_results

def main():
    """Main training function"""
    print("🎯 Mental Health Sentiment Model Training")
    print("=" * 50)
    
    # Check if data file exists
    csv_path = "data/mental_health_data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Data file not found: {csv_path}")
        return
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(csv_path)
        
        # Prepare dataset
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        train_dataset, val_dataset, label_mapping = prepare_dataset(df, tokenizer)
        
        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(val_dataset)}")
        print(f"📊 Labels: {label_mapping}")
        
        # Train the model
        model_path, eval_results = train_sentiment_model(
            train_dataset, val_dataset, label_mapping
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📊 Final accuracy: {eval_results.get('eval_accuracy', 'N/A')}")
        
        # Create a simple test script
        test_script = f'''#!/usr/bin/env python3
"""
Test script for the trained sentiment model
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import torch

def load_sentiment_model(model_path="./models/sentiment-model"):
    """Load the trained sentiment model"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    with open(f"{{model_path}}/label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    
    return tokenizer, model, label_mapping

def analyze_sentiment(text, tokenizer, model, label_mapping):
    """Analyze sentiment of a given text"""
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get predicted label
    predicted_id = torch.argmax(predictions, dim=-1).item()
    predicted_label = label_mapping[str(predicted_id)]
    confidence = predictions[0][predicted_id].item()
    
    return {{
        'predicted_emotion': predicted_label,
        'confidence': confidence,
        'all_scores': {{label_mapping[str(i)]: predictions[0][i].item() for i in range(len(label_mapping))}}
    }}

if __name__ == "__main__":
    # Load model
    tokenizer, model, label_mapping = load_sentiment_model()
    
    # Test with sample text
    test_text = "I am a 30 year old engineer. I work 40 hours per week. I have no mental health conditions. My stress level is low. I sleep 8 hours per night."
    
    result = analyze_sentiment(test_text, tokenizer, model, label_mapping)
    print(f"Test result: {{result}}")
'''
        
        with open("test_sentiment_model.py", "w") as f:
            f.write(test_script)
        
        print("📝 Test script created: test_sentiment_model.py")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
