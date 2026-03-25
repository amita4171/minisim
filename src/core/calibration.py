"""
Calibration System — Auto-correct systematic biases in swarm predictions.

Implements:
1. Platt scaling (logistic calibration) — learns a transform from predicted -> actual
2. Calibration curve computation with ECE (Expected Calibration Error)
3. Isotonic regression (non-parametric calibration)
4. Auto-correction: apply learned transform to new predictions

Based on:
- MIT Thermometer (temperature scaling for LLMs)
- Metaculus forecasting-tools calibration transformer
- arxiv 2404.09127 (confidence calibration via deliberation)

Usage:
    from src.calibration import CalibrationTransformer
    cal = CalibrationTransformer()
    cal.fit(predictions, outcomes)  # learn from resolved data
    corrected = cal.transform(0.65)  # apply correction
"""
from __future__ import annotations

import json
import math
import os
import statistics
from typing import Optional


class CalibrationTransformer:
    """Learns and applies calibration corrections to probability estimates.

    If predictions of 0.30 actually resolve YES 45% of the time,
    this transformer learns to shift 0.30 -> 0.45.
    """

    def __init__(self, method: str = "platt", n_buckets: int = 10):
        """
        Args:
            method: "platt" (logistic), "isotonic" (non-parametric), or "temperature"
            n_buckets: number of calibration buckets
        """
        self.method = method
        self.n_buckets = n_buckets
        self.is_fitted = False

        # Platt scaling parameters: P_calibrated = 1 / (1 + exp(-(a*logit(p) + b)))
        self.platt_a = 1.0
        self.platt_b = 0.0

        # Temperature scaling: P_calibrated = P^(1/T) / (P^(1/T) + (1-P)^(1/T))
        self.temperature = 1.0

        # Isotonic: lookup table
        self.isotonic_map = {}

        # Raw calibration data
        self.calibration_curve = {}
        self.ece = None
        self.n_samples = 0

    def fit(self, predictions: list[float], outcomes: list[float]):
        """Learn calibration parameters from resolved predictions.

        Args:
            predictions: list of predicted probabilities (0-1)
            outcomes: list of actual outcomes (0 or 1)
        """
        if len(predictions) < 5:
            return  # not enough data to calibrate

        self.n_samples = len(predictions)

        # Compute calibration curve
        self.calibration_curve = self._compute_curve(predictions, outcomes)

        # Compute ECE
        self.ece = self._compute_ece(predictions, outcomes)

        if self.method == "platt":
            self._fit_platt(predictions, outcomes)
        elif self.method == "temperature":
            self._fit_temperature(predictions, outcomes)
        elif self.method == "isotonic":
            self._fit_isotonic(predictions, outcomes)

        self.is_fitted = True

    def transform(self, p: float) -> float:
        """Apply calibration correction to a probability.

        Returns the corrected probability, or the original if not fitted.
        """
        if not self.is_fitted:
            return p

        p = max(0.001, min(0.999, p))

        if self.method == "platt":
            logit = math.log(p / (1 - p))
            calibrated_logit = self.platt_a * logit + self.platt_b
            return 1 / (1 + math.exp(-calibrated_logit))

        elif self.method == "temperature":
            p_t = p ** (1 / self.temperature)
            q_t = (1 - p) ** (1 / self.temperature)
            return p_t / (p_t + q_t)

        elif self.method == "isotonic":
            # Find nearest bucket
            best_bucket = None
            best_dist = float("inf")
            for bucket_mid, actual_rate in self.isotonic_map.items():
                dist = abs(p - bucket_mid)
                if dist < best_dist:
                    best_dist = dist
                    best_bucket = actual_rate
            return best_bucket if best_bucket is not None else p

        return p

    def transform_batch(self, predictions: list[float]) -> list[float]:
        """Apply calibration to a list of predictions."""
        return [self.transform(p) for p in predictions]

    def _fit_platt(self, predictions: list[float], outcomes: list[float]):
        """Fit Platt scaling via simple gradient descent on log-loss."""
        a, b = 1.0, 0.0
        lr = 0.01

        for _ in range(1000):
            grad_a, grad_b = 0.0, 0.0
            for p, y in zip(predictions, outcomes):
                p = max(0.001, min(0.999, p))
                logit = math.log(p / (1 - p))
                q = 1 / (1 + math.exp(-(a * logit + b)))
                q = max(0.001, min(0.999, q))
                err = q - y
                grad_a += err * logit
                grad_b += err

            n = len(predictions)
            a -= lr * grad_a / n
            b -= lr * grad_b / n

        self.platt_a = a
        self.platt_b = b

    def _fit_temperature(self, predictions: list[float], outcomes: list[float]):
        """Fit temperature scaling by minimizing log-loss."""
        best_t = 1.0
        best_loss = float("inf")

        for t_int in range(20, 300, 5):  # search T from 0.2 to 3.0
            t = t_int / 100.0
            loss = 0.0
            for p, y in zip(predictions, outcomes):
                p = max(0.001, min(0.999, p))
                p_t = p ** (1 / t)
                q_t = (1 - p) ** (1 / t)
                cal = p_t / (p_t + q_t)
                cal = max(0.001, min(0.999, cal))
                loss -= y * math.log(cal) + (1 - y) * math.log(1 - cal)

            if loss < best_loss:
                best_loss = loss
                best_t = t

        self.temperature = best_t

    def _fit_isotonic(self, predictions: list[float], outcomes: list[float]):
        """Fit isotonic (bucket-based) calibration."""
        self.isotonic_map = {}
        for i in range(self.n_buckets):
            lo = i / self.n_buckets
            hi = (i + 1) / self.n_buckets
            mid = (lo + hi) / 2

            bucket_outcomes = [y for p, y in zip(predictions, outcomes) if lo <= p < hi]
            if i == self.n_buckets - 1:
                bucket_outcomes += [y for p, y in zip(predictions, outcomes) if p == 1.0]

            if bucket_outcomes:
                self.isotonic_map[mid] = statistics.mean(bucket_outcomes)

    def _compute_curve(self, predictions: list[float], outcomes: list[float]) -> dict:
        """Compute calibration curve: predicted vs actual resolution rate."""
        curve = {}
        for i in range(self.n_buckets):
            lo = i / self.n_buckets
            hi = (i + 1) / self.n_buckets
            label = f"{lo:.1f}-{hi:.1f}"

            bucket_preds = []
            bucket_outcomes = []
            for p, y in zip(predictions, outcomes):
                if lo <= p < hi or (i == self.n_buckets - 1 and p == 1.0):
                    bucket_preds.append(p)
                    bucket_outcomes.append(y)

            if bucket_preds:
                curve[label] = {
                    "count": len(bucket_preds),
                    "mean_predicted": round(statistics.mean(bucket_preds), 4),
                    "actual_rate": round(statistics.mean(bucket_outcomes), 4),
                    "gap": round(statistics.mean(bucket_outcomes) - statistics.mean(bucket_preds), 4),
                }
        return curve

    def _compute_ece(self, predictions: list[float], outcomes: list[float]) -> float:
        """Compute Expected Calibration Error."""
        total_ece = 0.0
        n = len(predictions)
        for i in range(self.n_buckets):
            lo = i / self.n_buckets
            hi = (i + 1) / self.n_buckets
            bucket = [(p, y) for p, y in zip(predictions, outcomes) if lo <= p < hi]
            if i == self.n_buckets - 1:
                bucket += [(p, y) for p, y in zip(predictions, outcomes) if p == 1.0]
            if bucket:
                avg_pred = statistics.mean([p for p, y in bucket])
                avg_outcome = statistics.mean([y for p, y in bucket])
                total_ece += len(bucket) / n * abs(avg_outcome - avg_pred)
        return round(total_ece, 4)

    def get_summary(self) -> dict:
        """Get calibration summary."""
        return {
            "method": self.method,
            "is_fitted": self.is_fitted,
            "n_samples": self.n_samples,
            "ece": self.ece,
            "platt_a": round(self.platt_a, 4) if self.method == "platt" else None,
            "platt_b": round(self.platt_b, 4) if self.method == "platt" else None,
            "temperature": round(self.temperature, 4) if self.method == "temperature" else None,
            "calibration_curve": self.calibration_curve,
        }

    def save(self, path: str = "results/calibration_model.json"):
        """Save calibration model to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)

    @classmethod
    def load(cls, path: str = "results/calibration_model.json") -> CalibrationTransformer:
        """Load a fitted calibration model."""
        with open(path) as f:
            data = json.load(f)
        ct = cls(method=data["method"])
        ct.is_fitted = data["is_fitted"]
        ct.n_samples = data["n_samples"]
        ct.ece = data["ece"]
        if data.get("platt_a"):
            ct.platt_a = data["platt_a"]
            ct.platt_b = data["platt_b"]
        if data.get("temperature"):
            ct.temperature = data["temperature"]
        ct.calibration_curve = data.get("calibration_curve", {})
        return ct

    def print_summary(self):
        """Print calibration summary."""
        print(f"\n{'=' * 50}")
        print(f"Calibration Model ({self.method})")
        print(f"{'=' * 50}")
        print(f"  Fitted: {self.is_fitted} | Samples: {self.n_samples}")
        print(f"  ECE: {self.ece}")

        if self.method == "platt":
            print(f"  Platt a={self.platt_a:.4f}, b={self.platt_b:.4f}")
        elif self.method == "temperature":
            print(f"  Temperature: {self.temperature:.4f}")

        if self.calibration_curve:
            print(f"\n  Calibration Curve:")
            print(f"  {'Bucket':<10} {'N':>4}  {'Predicted':>10}  {'Actual':>10}  {'Gap':>8}")
            for label, d in sorted(self.calibration_curve.items()):
                print(f"  {label:<10} {d['count']:>4}  {d['mean_predicted']:>10.3f}  "
                      f"{d['actual_rate']:>10.3f}  {d['gap']:>+8.3f}")


def fit_calibration_from_backtest(backtest_path: str = "results/backtest_results.json") -> CalibrationTransformer:
    """Convenience: fit calibration from existing backtest results."""
    with open(backtest_path) as f:
        data = json.load(f)

    results = data.get("all_results", [])
    predictions = [r["swarm_probability"] for r in results]
    outcomes = [r["resolution"] for r in results]

    ct = CalibrationTransformer(method="platt")
    ct.fit(predictions, outcomes)
    ct.save()
    ct.print_summary()
    return ct
