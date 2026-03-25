# Fine-Tuning Guide for MiniSim

Fine-tune a language model on resolved prediction market data to improve
MiniSim's forecasting accuracy. This pipeline uses LoRA (Low-Rank Adaptation)
to efficiently train on your accumulated prediction history.

## Prerequisites

### Apple Silicon (MLX — recommended for local training)

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 14+ (Sonoma or later)
- 16GB+ RAM (32GB+ recommended for 14B parameter models)
- Python 3.10+

```bash
pip install mlx-lm
```

### NVIDIA GPU (Unsloth — recommended for cloud training)

- NVIDIA GPU with 16GB+ VRAM (A100, RTX 4090, etc.)
- CUDA 12.1+
- Python 3.10+

```bash
pip install unsloth
```

## Pipeline Overview

The fine-tuning pipeline consists of three scripts:

1. **`scripts/finetune_prep.py`** — Prepare training data from resolved predictions
2. **`scripts/finetune_train.py`** — Run LoRA fine-tuning (MLX or Unsloth)
3. **`scripts/finetune_eval.py`** — Evaluate fine-tuned model vs base model

## Step 1: Data Preparation

```bash
python scripts/finetune_prep.py
```

This script:
- Loads all resolved predictions from the SQLite database (`results/minisim.db`)
- Loads the 544-question eval dataset (`results/eval_dataset_500.json`)
- Deduplicates questions
- Splits 80/10/10 into train/val/test sets
- Saves JSONL files to `results/`

### Options

```bash
python scripts/finetune_prep.py --min-examples 100  # Require at least 100 examples
python scripts/finetune_prep.py --db path/to/other.db  # Use a different database
python scripts/finetune_prep.py --seed 123  # Different random seed for splits
```

### Training Data Format

Each example is formatted as a chat conversation:

```json
{
  "messages": [
    {"role": "system", "content": "You are a calibrated forecaster. Estimate P(YES) for the question."},
    {"role": "user", "content": "Question: Will the Fed cut rates?\nContext: Category: econ; Source: curated"},
    {"role": "assistant", "content": "{\"probability\": 0.0, \"reasoning\": \"Ground truth resolution.\"}"}
  ]
}
```

### Growing Your Training Set

More data generally means better fine-tuning results. Here are ways to grow
your training set:

1. **Run more predictions**: Use the MiniSim scanner and bot to predict on
   active markets, then resolve them as outcomes become known.

2. **Resolve historical predictions**: Run `python scripts/resolve_metaculus.py`
   and `python scripts/resolve_manual.py` to resolve past predictions.

3. **Import external datasets**: Add resolved questions from Metaculus,
   Manifold, or Kalshi to the eval dataset JSON.

4. **Minimum recommended**: 200+ examples for noticeable improvement,
   500+ for reliable gains, 1000+ for strong calibration improvement.

## Step 2: Training

### MLX (Apple Silicon)

```bash
python scripts/finetune_train.py \
    --model qwen2.5:14b \
    --epochs 3 \
    --lora-rank 8
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `qwen2.5:14b` | Model to fine-tune (Ollama-style name) |
| `--epochs` | `3` | Number of training epochs |
| `--lora-rank` | `8` | LoRA adapter rank (higher = more capacity) |
| `--learning-rate` | `2e-4` | Learning rate |
| `--batch-size` | `2` | Training batch size |

### Supported Models

| Ollama Name | MLX Model | RAM Needed |
|-------------|-----------|------------|
| `qwen2.5:3b` | Qwen2.5-3B-Instruct-4bit | 8GB |
| `qwen2.5:7b` | Qwen2.5-7B-Instruct-4bit | 16GB |
| `qwen2.5:14b` | Qwen2.5-14B-Instruct-4bit | 32GB |
| `llama3.1:8b` | Llama-3.1-8B-Instruct-4bit | 16GB |
| `mistral:7b` | Mistral-7B-Instruct-v0.3-4bit | 16GB |

### Unsloth (NVIDIA GPU)

If MLX is not available, the training script prints detailed Unsloth
instructions. In summary:

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=16)
# ... load JSONL data and train with SFTTrainer
```

### Training Output

Training produces:
- `results/finetune_output/` — LoRA adapter weights
- `results/finetune_config.json` — Training configuration

## Step 3: Evaluation

After training, import the fine-tuned model into Ollama and compare:

```bash
# Create Ollama model with the LoRA adapter
ollama create qwen2.5:14b-minisim -f results/finetune_output/Modelfile

# Run evaluation
python scripts/finetune_eval.py \
    --base-model qwen2.5:14b \
    --finetuned-model qwen2.5:14b-minisim
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | `qwen2.5:14b` | Base model in Ollama |
| `--finetuned-model` | `qwen2.5:14b-minisim` | Fine-tuned model in Ollama |
| `--test-data` | `results/finetune_test.jsonl` | Test set path |
| `--max-examples` | `0` (all) | Limit evaluation examples |

### Evaluation Metrics

The evaluation computes:
- **Brier Score** for each model on each test question (lower = better)
- **Average Brier Score** across all test questions
- **Improvement %** — how much better the fine-tuned model is
- **Win Rate** — what fraction of questions the fine-tuned model wins

Results are saved to `results/finetune_eval.json`.

## Expected Results

| Dataset Size | Expected Improvement | Notes |
|-------------|---------------------|-------|
| < 50 | Unreliable | Too little data; may overfit |
| 50-200 | 0-5% | Marginal gains, mostly calibration |
| 200-500 | 3-10% | Noticeable Brier improvement |
| 500-1000 | 5-15% | Strong calibration improvement |
| 1000+ | 10-20% | Best results, domain adaptation |

### When Fine-Tuning Is Worth It

Fine-tuning is worthwhile when:
- You have 200+ resolved predictions with ground truth
- The base model consistently miscalibrates in specific domains
- You want to encode domain-specific knowledge (e.g., political cycles, economic indicators)
- The Brier score improvement justifies the training compute

Fine-tuning is NOT worth it when:
- You have fewer than 50 examples (use prompt engineering instead)
- The base model is already well-calibrated for your domain
- You need the model to generalize to very different question types

## Cost Estimates

### Local (Apple Silicon with MLX)

| Model | RAM | Training Time (500 examples) | Cost |
|-------|-----|------------------------------|------|
| 3B | 8GB | ~10 min | Free |
| 7B | 16GB | ~30 min | Free |
| 14B | 32GB | ~60 min | Free |

### Cloud GPU (Unsloth)

| Provider | GPU | Training Time (500 examples) | Cost |
|----------|-----|------------------------------|------|
| Lambda Labs | A100 (80GB) | ~15 min | ~$0.50 |
| RunPod | A100 (40GB) | ~20 min | ~$0.40 |
| Vast.ai | RTX 4090 | ~25 min | ~$0.20 |
| Google Colab Pro | T4/A100 | ~30 min | ~$0.10 |

## Troubleshooting

### "Not enough examples" warning
Run more predictions through MiniSim and resolve them. See "Growing Your
Training Set" above.

### MLX out of memory
Try a smaller model (`--model qwen2.5:7b` or `--model qwen2.5:3b`) or
reduce batch size (`--batch-size 1`).

### Training loss not decreasing
- Increase epochs (`--epochs 5`)
- Lower learning rate (`--learning-rate 1e-4`)
- Check data quality — ensure resolutions are correct (0.0 or 1.0)

### Fine-tuned model worse than base
- More training data usually helps
- Try fewer epochs (overfitting)
- Ensure test set is representative of real forecasting questions
