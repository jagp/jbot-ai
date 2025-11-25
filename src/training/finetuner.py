"""Fine-tuning pipeline for language models."""

import torch
from pathlib import Path
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset


class FineTuner:
    """Fine-tune language models on custom data."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        output_dir: str = "./models/finetuned",
        use_lora: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize fine-tuner."""
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load base model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Apply LoRA if configured
        if self.use_lora:
            self.model = self._apply_lora()
        
        print(f"Model loaded on device: {self.device}")
    
    def _apply_lora(self):
        """Apply LoRA adaptation to reduce trainable parameters."""
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],  # For GPT-2 style models
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        return get_peft_model(self.model, lora_config)
    
    def prepare_dataset(self, data_path: str, max_length: int = 512):
        """Prepare dataset from JSONL file."""
        print(f"Loading dataset from: {data_path}")
        
        dataset = load_dataset('json', data_files=data_path)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        return tokenized_dataset
    
    def train(
        self,
        train_dataset,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        eval_dataset=None
    ):
        """Train the model."""
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            learning_rate=learning_rate,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            fp16=self.device == "cuda",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        print("Starting training...")
        trainer.train()
        print(f"Training complete. Model saved to {self.output_dir}")
    
    def save_model(self, path: Optional[str] = None):
        """Save the fine-tuned model."""
        save_path = path or self.output_dir
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        if self.use_lora:
            self.model.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)
        
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def load_finetuned_model(self, path: str):
        """Load a fine-tuned model."""
        print(f"Loading fine-tuned model from {path}")
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the fine-tuned model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
