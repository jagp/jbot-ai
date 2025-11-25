#!/usr/bin/env python
"""
Fine-tuning script for custom language models.

Usage:
    python scripts/finetune.py --model gpt2 --data data/processed/train.jsonl --output models/finetuned
"""

import argparse
import torch
from pathlib import Path

from src.training.finetuner import FineTuner


def main(args):
    """Run the fine-tuning pipeline."""
    
    # Initialize fine-tuner
    finetuner = FineTuner(
        model_name=args.model,
        output_dir=args.output,
        use_lora=args.use_lora
    )
    
    # Load model
    finetuner.load_model()
    
    # Check if training data exists
    if not Path(args.data).exists():
        print(f"Error: Training data not found at {args.data}")
        print("Please run: python scripts/prepare_data.py")
        return
    
    # Prepare dataset
    train_dataset = finetuner.prepare_dataset(args.data, max_length=args.max_length)
    
    # Split into train/eval if needed
    split_dataset = train_dataset['train'].train_test_split(test_size=0.1)
    
    # Train the model
    finetuner.train(
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save the model
    finetuner.save_model()
    print("Fine-tuning complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model on custom data")
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Model name or path to fine-tune"
    )
    parser.add_argument(
        "--data",
        default="data/processed/train.jsonl",
        help="Path to training data in JSONL format"
    )
    parser.add_argument(
        "--output",
        default="models/finetuned",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training"
    )
    
    args = parser.parse_args()
    main(args)
