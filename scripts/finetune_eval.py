"""
Fine-Tune Evaluation — Compare fine-tuned model against base model.

Loads the test split, runs predictions through both models via Ollama,
computes Brier scores, and prints a comparison table.

Usage:
    python scripts/finetune_eval.py \
        --base-model qwen2.5:14b \
        --finetuned-model qwen2.5:14b-minisim
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_test_set(path: str) -> list[dict]:
    """Load test examples from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def extract_ground_truth(example: dict) -> float:
    """Extract the ground truth probability from a training example."""
    assistant_msg = example["messages"][2]["content"]
    try:
        data = json.loads(assistant_msg)
        return float(data["probability"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0.0


def extract_question(example: dict) -> str:
    """Extract the question text from a training example."""
    return example["messages"][1]["content"]


def query_ollama(model: str, question: str, system_prompt: str) -> float | None:
    """Query an Ollama model and extract a probability prediction."""
    try:
        import requests
    except ImportError:
        print("  ERROR: requests library required. pip install requests")
        sys.exit(1)

    prompt_text = (
        f"{question}\n\n"
        "Respond with a JSON object containing your probability estimate: "
        '{\"probability\": <float between 0 and 1>, \"reasoning\": \"<brief reasoning>\"}'
    )

    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")

        # Try to parse JSON from response
        try:
            data = json.loads(content)
            return float(data["probability"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback: look for a number after "probability"
        match = re.search(r'"probability"\s*:\s*([\d.]+)', content)
        if match:
            return float(match.group(1))

        # Fallback: look for any float between 0 and 1
        match = re.search(r'\b(0\.\d+|1\.0|0\.0)\b', content)
        if match:
            return float(match.group(1))

        return None
    except Exception as e:
        print(f"    Error querying {model}: {e}")
        return None


def brier_score(predicted: float, actual: float) -> float:
    """Compute Brier score (lower is better)."""
    return (predicted - actual) ** 2


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model vs base model"
    )
    parser.add_argument(
        "--base-model",
        default="qwen2.5:14b",
        help="Base model name in Ollama (default: qwen2.5:14b)",
    )
    parser.add_argument(
        "--finetuned-model",
        default="qwen2.5:14b-minisim",
        help="Fine-tuned model name in Ollama (default: qwen2.5:14b-minisim)",
    )
    parser.add_argument(
        "--test-data",
        default=os.path.join(RESULTS_DIR, "finetune_test.jsonl"),
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Max examples to evaluate (0 = all)",
    )
    args = parser.parse_args()

    print("=== Fine-Tune Evaluation ===\n")

    # 1. Load test set
    if not os.path.exists(args.test_data):
        print(f"  ERROR: Test data not found at {args.test_data}")
        print("  Run finetune_prep.py first.")
        sys.exit(1)

    test_examples = load_test_set(args.test_data)
    if args.max_examples > 0:
        test_examples = test_examples[: args.max_examples]

    print(f"  Test examples: {len(test_examples)}")
    print(f"  Base model:    {args.base_model}")
    print(f"  Finetuned:     {args.finetuned_model}")
    print()

    system_prompt = test_examples[0]["messages"][0]["content"] if test_examples else ""

    # 2. Run predictions
    results = []
    base_briers = []
    ft_briers = []
    base_errors = 0
    ft_errors = 0

    for i, example in enumerate(test_examples):
        question = extract_question(example)
        truth = extract_ground_truth(example)

        short_q = question[:60] + "..." if len(question) > 60 else question
        print(f"  [{i+1}/{len(test_examples)}] {short_q}")

        # Query base model
        base_pred = query_ollama(args.base_model, question, system_prompt)
        if base_pred is None:
            base_errors += 1
            base_pred = 0.5  # fallback

        # Query fine-tuned model
        ft_pred = query_ollama(args.finetuned_model, question, system_prompt)
        if ft_pred is None:
            ft_errors += 1
            ft_pred = 0.5  # fallback

        base_bs = brier_score(base_pred, truth)
        ft_bs = brier_score(ft_pred, truth)

        base_briers.append(base_bs)
        ft_briers.append(ft_bs)

        results.append({
            "question": question,
            "ground_truth": truth,
            "base_prediction": base_pred,
            "finetuned_prediction": ft_pred,
            "base_brier": round(base_bs, 6),
            "finetuned_brier": round(ft_bs, 6),
        })

        print(f"    Truth: {truth:.1f} | Base: {base_pred:.3f} (BS={base_bs:.4f}) | "
              f"FT: {ft_pred:.3f} (BS={ft_bs:.4f})")

    # 3. Compute summary
    if not results:
        print("\n  No results to evaluate.")
        sys.exit(1)

    avg_base = sum(base_briers) / len(base_briers)
    avg_ft = sum(ft_briers) / len(ft_briers)
    improvement = ((avg_base - avg_ft) / avg_base * 100) if avg_base > 0 else 0
    ft_wins = sum(1 for r in results if r["finetuned_brier"] < r["base_brier"])

    # 4. Print comparison table
    print("\n" + "=" * 60)
    print("  COMPARISON: Base vs Fine-Tuned")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Base':>12} {'Fine-Tuned':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Avg Brier Score':<30} {avg_base:>12.4f} {avg_ft:>12.4f}")
    print(f"  {'Questions Evaluated':<30} {len(results):>12} {len(results):>12}")
    print(f"  {'Prediction Errors':<30} {base_errors:>12} {ft_errors:>12}")
    print(f"  {'FT Wins':<30} {'':>12} {ft_wins:>12}")
    print(f"  {'Improvement':<30} {'':>12} {improvement:>+11.1f}%")
    print("=" * 60)

    if improvement > 0:
        print(f"\n  Fine-tuned model is {improvement:.1f}% better (lower Brier = better)")
    elif improvement < 0:
        print(f"\n  Base model is {-improvement:.1f}% better — fine-tuning may need more data")
    else:
        print("\n  Models perform identically on this test set")

    # 5. Save results
    eval_output = {
        "base_model": args.base_model,
        "finetuned_model": args.finetuned_model,
        "n_examples": len(results),
        "base_avg_brier": round(avg_base, 6),
        "finetuned_avg_brier": round(avg_ft, 6),
        "improvement_pct": round(improvement, 2),
        "finetuned_wins": ft_wins,
        "base_errors": base_errors,
        "finetuned_errors": ft_errors,
        "results": results,
    }

    output_path = os.path.join(RESULTS_DIR, "finetune_eval.json")
    with open(output_path, "w") as f:
        json.dump(eval_output, f, indent=2)

    print(f"\n  Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
