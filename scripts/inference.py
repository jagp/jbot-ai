#!/usr/bin/env python
"""
Inference script to test the fine-tuned model.

Usage:
    python scripts/inference.py --model models/finetuned --prompt "Hello, I"
"""

import argparse
import torch
from src.training.finetuner import FineTuner


def main(args):
    """Run inference with fine-tuned model."""
    
    # Initialize fine-tuner
    finetuner = FineTuner(
        model_name=args.model,
        use_lora=False  # Disable LoRA for inference
    )
    
    # Load the fine-tuned model
    finetuner.load_finetuned_model(args.model)
    
    # Generate text
    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    
    generated = finetuner.generate(args.prompt, max_length=args.max_length)
    print(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned model")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--prompt",
        default="Hello, I",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum length of generated text"
    )
    
    args = parser.parse_args()
    main(args)
