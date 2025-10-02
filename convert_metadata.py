"""
Convert metadata.json to CSV format for pipeline processing
"""

import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_metadata_to_csv(metadata_path: str, output_path: str):
    """
    Convert nested JSON metadata to flat CSV with text and emotion columns
    
    Args:
        metadata_path: Path to metadata.json file
        output_path: Path to output CSV file
    """
    logger.info(f"Loading metadata from: {metadata_path}")
    
    # Load JSON data
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all utterances
    utterances = []
    
    for dialog_id, dialog in data.items():
        if isinstance(dialog, dict):
            for utterance_id, utterance in dialog.items():
                if isinstance(utterance, dict) and 'text' in utterance and 'emotion' in utterance:
                    text = utterance.get('text', '').strip()
                    emotion = utterance.get('emotion', '').strip()
                    
                    # Only include if text is not empty
                    if text and len(text) > 5:  # Minimum text length
                        utterances.append({
                            'text': text,
                            'emotion': emotion,
                            'dialog_idx': utterance.get('dialog_idx', dialog_id),
                            'utterance_idx': utterance.get('utterance_idx', utterance_id),
                            'speaker': utterance.get('speaker', ''),
                            'act': utterance.get('act', '')
                        })
    
    # Create DataFrame
    df = pd.DataFrame(utterances)
    
    logger.info(f"Extracted {len(df)} utterances from {len(data)} dialogs")
    logger.info(f"\nEmotion distribution:")
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        logger.info(f"  {emotion}: {count}")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved dataset to: {output_path}")
    
    return df


def main():
    """Main conversion function"""
    # Paths
    project_root = Path(__file__).parent
    metadata_path = project_root / "metadata.json"
    output_path = project_root / "data" / "dailytalk.csv"
    
    # Ensure data directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert
    df = convert_metadata_to_csv(str(metadata_path), str(output_path))
    
    # Display sample
    logger.info("\nSample data:")
    logger.info(df[['text', 'emotion']].head(10).to_string())
    
    logger.info("\n" + "="*80)
    logger.info("Conversion complete! Ready to run pipeline.")
    logger.info("="*80)


if __name__ == "__main__":
    main()
