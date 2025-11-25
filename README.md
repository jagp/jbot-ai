# jbot-ai: Personal Language Model Fine-Tuning

Fine-tune open-source language models using your personal writings, emails, and text messages as training data.

## Overview

This project provides a complete pipeline for:
- **Data Collection**: Load emails, messages, writings, and documents from various sources
- **Data Processing**: Clean, normalize, and prepare text data for training
- **Model Fine-Tuning**: Fine-tune models like GPT-2 using your personal data with optional LoRA
- **Inference**: Generate text using your custom-trained model

## Features

- 📧 Support for multiple data sources (emails, messages, writings)
- 🔄 Configurable data processing pipeline
- 🚀 Parameter-efficient training with LoRA support
- 💾 Model checkpoint management
- 🧪 Easy inference for testing outputs
- 🛠️ Extensible architecture for custom models

## Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration, optional but recommended)
- 8GB+ RAM (16GB+ recommended)

## Installation

1. **Clone and setup the workspace:**
   ```bash
   cd c:\Users\jared\Projects\jbot-ai
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Step 1: Configure Your Data Sources

Edit `config/sources.yaml` to point to your data:

```yaml
sources:
  emails:
    enabled: true
    path: ./data/raw/emails
    format: "mbox"
  
  messages:
    enabled: true
    path: ./data/raw/messages
    format: "json"
  
  writings:
    enabled: true
    path: ./data/raw/writings
    format: "txt"
```

### Step 2: Prepare Training Data

```bash
python scripts/prepare_data.py --config config/sources.yaml --output data/processed/train.jsonl
```

This will:
- Load documents from all enabled sources
- Clean and normalize text
- Remove duplicates and filter by length
- Output JSONL format suitable for training

### Step 3: Fine-Tune the Model

```bash
python scripts/finetune.py \
  --model gpt2 \
  --data data/processed/train.jsonl \
  --output models/finetuned \
  --epochs 3 \
  --batch-size 8 \
  --use-lora
```

Options:
- `--model`: Base model name (default: gpt2)
- `--data`: Training data path
- `--output`: Model output directory
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--use-lora`: Enable LoRA for efficient training
- `--learning-rate`: Learning rate (default: 5e-5)

### Step 4: Test the Model

```bash
python scripts/inference.py \
  --model models/finetuned \
  --prompt "Hello, I think that"
```

## Project Structure

```
jbot-ai/
├── config/                  # Configuration files
│   └── sources.yaml        # Data source configuration
├── data/
│   ├── raw/               # Raw input data (emails, messages, etc)
│   └── processed/         # Processed training data (JSONL)
├── scripts/               # Main executables
│   ├── prepare_data.py   # Data preparation script
│   ├── finetune.py       # Fine-tuning script
│   └── inference.py      # Inference script
├── src/                  # Source code modules
│   ├── data_processing/  # Data loading and cleaning
│   │   ├── loader.py    # Load data from various sources
│   │   └── cleaner.py   # Text cleaning utilities
│   └── training/        # Model training
│       └── finetuner.py # Fine-tuning implementation
├── models/              # Saved models directory
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Data Source Formats

### Text Files (.txt)
- Plain text files, one document per file
- Location: `data/raw/writings/`

### JSON Files (.json)
- Array of objects with `content` field
- Example: `[{"content": "My message..."}, ...]`
- Location: `data/raw/messages/`

### CSV Files (.csv)
- Must have `text` or `content` column
- Location: `data/raw/documents/`

### Email (mbox format)
- Standard mbox format for email exports
- Location: `data/raw/emails/`

## Export Your Data

### Gmail Emails
1. Use Google Takeout to export emails
2. Convert to mbox format using tools like `goog-mail-backup`
3. Place in `data/raw/emails/`

### WhatsApp/iMessage
1. Export conversations as JSON or text files
2. Place in `data/raw/messages/`

### Personal Writings
1. Export notes/journals as text files
2. Place in `data/raw/writings/`

## Advanced Usage

### Using Different Base Models

```bash
python scripts/finetune.py --model EleutherAI/gpt-neo-125M --data data/processed/train.jsonl
```

Popular models:
- `gpt2` - Small, fast
- `EleutherAI/gpt-neo-125M` - Medium quality
- `EleutherAI/gpt-j-6B` - Larger (requires more VRAM)

### Adjusting Training Parameters

For faster training with less data:
```bash
python scripts/finetune.py --epochs 1 --batch-size 16 --learning-rate 1e-4
```

For better quality with more data:
```bash
python scripts/finetune.py --epochs 5 --batch-size 4 --learning-rate 3e-5
```

### GPU Acceleration

Models automatically use CUDA if available. To force CPU:
```python
# In scripts/finetune.py or inference.py
finetuner = FineTuner(model_name=args.model, device="cpu")
```

## Troubleshooting

### Out of Memory (OOM) Error
- Reduce `batch-size`: `--batch-size 4`
- Enable LoRA: `--use-lora`
- Use a smaller model: `--model gpt2`

### No Data Found
- Check that `config/sources.yaml` has correct paths
- Verify data files exist in `data/raw/`
- Ensure paths are relative to project root

### Slow Training
- Reduce `--max-length` to 256 or 384
- Use GPU: ensure CUDA is installed
- Reduce data size for testing

## Tips for Best Results

1. **Data Quality**: Clean, well-formatted text produces better models
2. **Data Volume**: Minimum ~1000 documents recommended
3. **Training Time**: Start with 1-2 epochs on a small subset to test
4. **Diversity**: Mix different types of writing for more robust outputs

## License

This project is for personal use. Respect data privacy when using personal information.

## Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Fine-tuning Guide](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## Contributing

Feel free to extend this project with:
- Additional data source formats
- New model architectures
- Custom training strategies
- Evaluation metrics

## Future Enhancements

- [ ] Web UI for data management
- [ ] Support for larger models (LLaMA, Mistral)
- [ ] Quantization for inference
- [ ] Distributed training
- [ ] Model evaluation metrics
