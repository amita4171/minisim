"""
Fine-Tune Data Preparation — Export resolved predictions to JSONL for fine-tuning.

Loads resolved predictions from SQLite DB and the 544-question eval dataset,
formats them as chat-style training examples, and splits 80/10/10 into
train/val/test JSONL files.

Usage:
    python scripts/finetune_prep.py [--min-examples 50] [--db results/minisim.db]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db.database import Database

SYSTEM_PROMPT = "You are a calibrated forecaster. Estimate P(YES) for the question."

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def format_training_example(
    question: str,
    resolution: float,
    context: str | None = None,
) -> dict:
    """Create a chat-format training example from a resolved question."""
    user_content = f"Question: {question}"
    if context:
        user_content += f"\nContext: {context}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "probability": resolution,
                        "reasoning": "Ground truth resolution.",
                    }
                ),
            },
        ]
    }


def load_db_examples(db_path: str) -> list[dict]:
    """Load resolved predictions from the SQLite database."""
    db = Database(path=db_path)
    rows = db.conn.execute(
        "SELECT question, resolution, category, source FROM predictions "
        "WHERE resolution IS NOT NULL"
    ).fetchall()
    db.close()

    examples = []
    for row in rows:
        context_parts = []
        if row["category"]:
            context_parts.append(f"Category: {row['category']}")
        if row["source"]:
            context_parts.append(f"Source: {row['source']}")
        context = "; ".join(context_parts) if context_parts else None

        examples.append(
            format_training_example(row["question"], row["resolution"], context)
        )
    return examples


def load_eval_dataset(eval_path: str) -> list[dict]:
    """Load the 544-question eval dataset and format as training examples."""
    if not os.path.exists(eval_path):
        print(f"  Warning: eval dataset not found at {eval_path}, skipping.")
        return []

    with open(eval_path) as f:
        data = json.load(f)

    questions = data.get("questions", [])
    examples = []
    for q in questions:
        if q.get("resolution") is None:
            continue
        context_parts = []
        if q.get("category"):
            context_parts.append(f"Category: {q['category']}")
        if q.get("source"):
            context_parts.append(f"Source: {q['source']}")
        if q.get("market_price") is not None:
            context_parts.append(f"Market price: {q['market_price']}")
        context = "; ".join(context_parts) if context_parts else None

        examples.append(
            format_training_example(q["question"], q["resolution"], context)
        )
    return examples


def estimate_tokens(examples: list[dict]) -> int:
    """Rough token estimate: ~4 chars per token across all messages."""
    total_chars = 0
    for ex in examples:
        for msg in ex["messages"]:
            total_chars += len(msg["content"])
    return total_chars // 4


def split_data(
    examples: list[dict], seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split examples 80/10/10 into train/val/test."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def save_jsonl(examples: list[dict], path: str) -> None:
    """Save examples to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare fine-tuning data from resolved predictions"
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=50,
        help="Minimum examples required for fine-tuning (default: 50)",
    )
    parser.add_argument(
        "--db",
        default=os.path.join(RESULTS_DIR, "minisim.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--eval-dataset",
        default=os.path.join(RESULTS_DIR, "eval_dataset_500.json"),
        help="Path to eval dataset JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    args = parser.parse_args()

    print("=== Fine-Tune Data Preparation ===\n")

    # 1. Load from DB
    print("Loading resolved predictions from database...")
    db_examples = load_db_examples(args.db)
    print(f"  Found {len(db_examples)} resolved predictions in DB")

    # 2. Load eval dataset
    print("Loading eval dataset...")
    eval_examples = load_eval_dataset(args.eval_dataset)
    print(f"  Found {len(eval_examples)} questions from eval dataset")

    # 3. Deduplicate by question text
    seen_questions: set[str] = set()
    all_examples: list[dict] = []
    for ex in db_examples + eval_examples:
        q_text = ex["messages"][1]["content"]
        if q_text not in seen_questions:
            seen_questions.add(q_text)
            all_examples.append(ex)

    print(f"\n  Total unique examples: {len(all_examples)}")

    # 4. Check minimum
    if len(all_examples) < args.min_examples:
        print(
            f"\n  WARNING: Only {len(all_examples)} examples found, "
            f"but {args.min_examples} recommended for fine-tuning."
        )
        print("  Consider running more predictions and resolving them first.")
        print("  See docs/fine-tuning.md for guidance on growing your dataset.\n")

    # 5. Split
    train, val, test = split_data(all_examples, seed=args.seed)

    # 6. Save
    train_path = os.path.join(RESULTS_DIR, "finetune_train.jsonl")
    val_path = os.path.join(RESULTS_DIR, "finetune_val.jsonl")
    test_path = os.path.join(RESULTS_DIR, "finetune_test.jsonl")

    save_jsonl(train, train_path)
    save_jsonl(val, val_path)
    save_jsonl(test, test_path)

    # 7. Stats
    total_tokens = estimate_tokens(all_examples)
    print(f"\n=== Dataset Statistics ===")
    print(f"  Total examples: {len(all_examples)}")
    print(f"  Train split:    {len(train)} ({len(train)/len(all_examples)*100:.0f}%)")
    print(f"  Val split:      {len(val)} ({len(val)/len(all_examples)*100:.0f}%)")
    print(f"  Test split:     {len(test)} ({len(test)/len(all_examples)*100:.0f}%)")
    print(f"  Est. tokens:    {total_tokens:,}")
    print(f"\n  Saved to:")
    print(f"    {train_path}")
    print(f"    {val_path}")
    print(f"    {test_path}")

    if len(all_examples) >= args.min_examples:
        print(f"\n  Dataset meets minimum threshold ({args.min_examples}). Ready for fine-tuning!")
    print()


if __name__ == "__main__":
    main()
