#!/usr/bin/env python
"""
Main script to process data and prepare it for model fine-tuning.

Usage:
    python scripts/prepare_data.py --config config/sources.yaml --output data/processed/train.jsonl
"""

import json
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm

from src.data_processing.loader import DataLoader
from src.data_processing.cleaner import TextCleaner, CleaningConfig


def main(args):
    """Process and prepare training data."""
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data from all sources
    loader = DataLoader(args.config)
    documents = loader.load_all_sources()
    print(f"Total documents loaded: {len(documents)}")
    
    if not documents:
        print("No documents found. Please configure data sources in config/sources.yaml")
        return
    
    # Extract text and metadata
    texts = [doc.content for doc in documents]
    
    # Clean texts
    cleaning_config = CleaningConfig(
        remove_urls=config['processing'].get('remove_urls', True),
        remove_emails=config['processing'].get('remove_emails', False),
        normalize_whitespace=config['processing'].get('normalize_whitespace', True),
        remove_duplicates=config['processing'].get('remove_duplicates', True),
        min_length=config['processing'].get('min_length', 50),
        max_length=config['processing'].get('max_length', 2048)
    )
    
    cleaner = TextCleaner(cleaning_config)
    cleaned_texts = cleaner.clean(texts)
    print(f"Texts after cleaning: {len(cleaned_texts)}")
    
    # Prepare output format
    output_format = config['output'].get('format', 'jsonl')
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to JSONL format
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in tqdm(cleaned_texts, desc="Writing training data"):
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Training data saved to {output_path}")
    print(f"Total samples: {len(cleaned_texts)}")
    
    # Calculate splits
    total = len(cleaned_texts)
    train_split = config['output'].get('train_split', 0.8)
    val_split = config['output'].get('val_split', 0.1)
    
    train_size = int(total * train_split)
    val_size = int(total * val_split)
    
    print(f"Recommended split: train={train_size}, val={val_size}, test={total - train_size - val_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for model fine-tuning")
    parser.add_argument(
        "--config",
        default="config/sources.yaml",
        help="Path to sources configuration file"
    )
    parser.add_argument(
        "--output",
        default="data/processed/train.jsonl",
        help="Output path for processed training data"
    )
    
    args = parser.parse_args()
    main(args)
