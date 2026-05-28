"""
Evaluate Autoencoder, Isolation Forest, and their ensemble on held-out test data.
Produces intrinsic metrics (F1, precision, recall) and simulated business impact.

Usage:
    python -m ml.evaluation.evaluate --csv ml/artifacts/dev/simulated_transactions.csv --model-dir ml/artifacts/dev
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import joblib
import torch
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from ml.training.train import load_and_prepare, Autoencoder, INPUT_DIM, HIDDEN_DIM, LATENT_DIM
from ml.features.feature_engineering import FEATURE_NAMES

# --- Simulated business costs (adjust to match your scenario) ---
COST_PER_FALSE_POSITIVE = 100    # manual review / customer friction per flag
SAVINGS_PER_TRUE_POSITIVE = 1000 # fraud loss prevented per catch
AVG_FRAUD_LOSS_UNDETECTED = 500  # average loss when fraud goes undetected


def _load_ae(model_dir: str):
    model = Autoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "autoencoder_model.pt"), map_location="cpu")
    )
    model.eval()
    scaler = joblib.load(os.path.join(model_dir, "minmax_scaler.joblib"))
    threshold = float(open(os.path.join(model_dir, "threshold.txt")).read().strip())
    return model, scaler, threshold


def _ae_predict(model, scaler, threshold, X: np.ndarray):
    X_s = scaler.transform(X)
    X_t = torch.tensor(X_s, dtype=torch.float32)
    with torch.no_grad():
        recon = model(X_t)
        errs = torch.mean((X_t - recon) ** 2, dim=1).numpy()
    return (errs > threshold).astype(int), errs


def _if_predict(if_model, X: np.ndarray) -> np.ndarray:
    return np.where(if_model.predict(X) == -1, 1, 0)


def _print_metrics(y_true, y_pred, model_name: str) -> dict:
    SEP = "=" * 58
    print(f"\n{SEP}")
    print(f"  {model_name}")
    print(SEP)
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"], zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    print(f"Confusion Matrix:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"F1={f1:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")

    # Simulated business impact
    potential_loss = (tp + fn) * AVG_FRAUD_LOSS_UNDETECTED
    savings = tp * SAVINGS_PER_TRUE_POSITIVE
    review_cost = fp * COST_PER_FALSE_POSITIVE
    net = savings - review_cost

    print(f"\n  Simulated Business Impact")
    print(f"  {'Potential fraud losses (undetected):':<38} ${potential_loss:>10,.0f}")
    print(f"  {'Savings from caught fraud (TP):':<38} ${savings:>10,.0f}")
    print(f"  {'Review cost from false alarms (FP):':<38} -${review_cost:>9,.0f}")
    print(f"  {'Net savings:':<38} ${net:>10,.0f}")

    return {
        "model": model_name,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "net_savings": net,
    }


def evaluate(csv_path: str, model_dir: str):
    print(f"Loading data from {csv_path}...")
    df = load_and_prepare(csv_path)

    normal_X = df[df["is_fraud"] == False][FEATURE_NAMES].values
    fraud_X = df[df["is_fraud"] == True][FEATURE_NAMES].values

    # Use same random_state as training so val set is truly held-out
    _, X_val_normal = train_test_split(normal_X, test_size=0.2, random_state=42)

    X_test = np.vstack([X_val_normal, fraud_X])
    y_test = np.array([0] * len(X_val_normal) + [1] * len(fraud_X))
    print(f"Test set: {len(X_val_normal)} normal + {len(fraud_X)} fraud")

    ae_model, scaler, threshold = _load_ae(model_dir)
    if_model = joblib.load(os.path.join(model_dir, "isolation_forest.joblib"))

    ae_preds, ae_errors = _ae_predict(ae_model, scaler, threshold, X_test)
    if_preds = _if_predict(if_model, X_test)
    ensemble_preds = np.where((ae_preds == 1) | (if_preds == 1), 1, 0)

    results = [
        _print_metrics(y_test, ae_preds, "Autoencoder"),
        _print_metrics(y_test, if_preds, "Isolation Forest"),
        _print_metrics(y_test, ensemble_preds, "Ensemble  (AE OR IF flags)"),
    ]

    print(f"\n{'=' * 58}")
    print("  Summary Comparison")
    print(f"{'=' * 58}")
    print(f"{'Model':<28} {'F1':>6} {'Prec':>7} {'Recall':>8} {'Net Savings':>13}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['model']:<28} {r['f1']:>6.4f} {r['precision']:>7.4f} "
            f"{r['recall']:>8.4f} ${r['net_savings']:>11,.0f}"
        )
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="ml/artifacts/dev/simulated_transactions.csv")
    p.add_argument("--model-dir", default="ml/artifacts/dev")
    args = p.parse_args()
    evaluate(args.csv, args.model_dir)
