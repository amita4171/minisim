"""
Fine-Tune Training — LoRA fine-tune a language model on MiniSim predictions.

Uses MLX (Apple Silicon native) for local training, with fallback instructions
for cloud GPU training via Unsloth.

Usage:
    python scripts/finetune_train.py [--model qwen2.5:14b] [--epochs 3] [--lora-rank 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def check_mlx_available() -> bool:
    """Check if mlx-lm is installed and available."""
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def check_training_data() -> tuple[bool, dict]:
    """Verify training data exists and return stats."""
    train_path = os.path.join(RESULTS_DIR, "finetune_train.jsonl")
    val_path = os.path.join(RESULTS_DIR, "finetune_val.jsonl")

    if not os.path.exists(train_path):
        return False, {"error": "Training data not found. Run finetune_prep.py first."}
    if not os.path.exists(val_path):
        return False, {"error": "Validation data not found. Run finetune_prep.py first."}

    train_count = sum(1 for _ in open(train_path))
    val_count = sum(1 for _ in open(val_path))

    return True, {"train_examples": train_count, "val_examples": val_count}


def resolve_model_name(model_arg: str) -> str:
    """Map Ollama-style model names to HuggingFace model IDs for MLX.

    MLX needs HuggingFace model paths. Common mappings:
    - qwen2.5:14b -> Qwen/Qwen2.5-14B-Instruct
    - qwen2.5:7b  -> Qwen/Qwen2.5-7B-Instruct
    - llama3.1:8b -> meta-llama/Llama-3.1-8B-Instruct
    - mistral:7b  -> mistralai/Mistral-7B-Instruct-v0.3
    """
    mappings = {
        "qwen2.5:14b": "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "qwen2.5:7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "qwen2.5:3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "llama3.1:8b": "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "mistral:7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    }
    return mappings.get(model_arg, model_arg)


def print_install_instructions():
    """Print instructions for installing MLX and alternative cloud approaches."""
    print("""
=== MLX-LM Not Installed ===

To fine-tune locally on Apple Silicon, install mlx-lm:

    pip install mlx-lm

Requirements:
  - Apple Silicon Mac (M1/M2/M3/M4)
  - macOS 14+ (Sonoma or later)
  - 16GB+ RAM (32GB+ recommended for 14B models)

=== Alternative: Cloud GPU with Unsloth ===

For NVIDIA GPU training (faster, supports larger models):

1. Set up a cloud GPU instance (Lambda Labs, RunPod, etc.)
2. Install Unsloth:

    pip install unsloth

3. Use this training script template:

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,            # LoRA rank
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # Load your JSONL data and train with SFTTrainer
    from trl import SFTTrainer
    from transformers import TrainingArguments

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,  # load from finetune_train.jsonl
        args=TrainingArguments(
            output_dir="results/finetune_output",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            learning_rate=2e-4,
        ),
    )
    trainer.train()
    model.save_pretrained("results/finetune_output")

See docs/fine-tuning.md for full instructions.
""")


def save_config(args: argparse.Namespace, data_stats: dict, output_dir: str) -> str:
    """Save training configuration to JSON."""
    config = {
        "model": args.model,
        "model_resolved": resolve_model_name(args.model),
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "output_dir": output_dir,
        "train_data": os.path.join(RESULTS_DIR, "finetune_train.jsonl"),
        "val_data": os.path.join(RESULTS_DIR, "finetune_val.jsonl"),
        "data_stats": data_stats,
    }
    config_path = os.path.join(RESULTS_DIR, "finetune_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def run_mlx_finetune(args: argparse.Namespace) -> None:
    """Run LoRA fine-tuning using MLX-LM."""
    from mlx_lm import fine_tune

    model_name = resolve_model_name(args.model)
    output_dir = os.path.join(RESULTS_DIR, "finetune_output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Model:      {model_name}")
    print(f"  LoRA rank:  {args.lora_rank}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output:     {output_dir}")
    print()

    fine_tune(
        model=model_name,
        train_data=os.path.join(RESULTS_DIR, "finetune_train.jsonl"),
        val_data=os.path.join(RESULTS_DIR, "finetune_val.jsonl"),
        lora_rank=args.lora_rank,
        epochs=args.epochs,
        output_dir=output_dir,
    )

    print(f"\n  Fine-tuning complete! Adapter saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on MiniSim prediction data"
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:14b",
        help="Model to fine-tune (default: qwen2.5:14b)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank for adapter (default: 8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size (default: 2)",
    )
    args = parser.parse_args()

    print("=== MiniSim Fine-Tune Training ===\n")

    # 1. Check training data
    print("Checking training data...")
    data_ok, data_stats = check_training_data()
    if not data_ok:
        print(f"  ERROR: {data_stats['error']}")
        sys.exit(1)
    print(f"  Train: {data_stats['train_examples']} examples")
    print(f"  Val:   {data_stats['val_examples']} examples")
    print()

    # 2. Save config regardless of backend
    output_dir = os.path.join(RESULTS_DIR, "finetune_output")
    config_path = save_config(args, data_stats, output_dir)
    print(f"  Config saved to: {config_path}\n")

    # 3. Check for MLX
    if check_mlx_available():
        print("MLX-LM detected. Starting LoRA fine-tuning...\n")
        run_mlx_finetune(args)
    else:
        print("MLX-LM not available.")
        print_install_instructions()
        print("Config has been saved — you can use it with your preferred training framework.")


if __name__ == "__main__":
    main()
