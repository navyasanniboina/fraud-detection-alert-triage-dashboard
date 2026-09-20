"""
Evaluation & Visualization Module for Credit Card Fraud Detection.
Computes PR-AUC, ROC-AUC, threshold performance, and generates diagnostic plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier_at_threshold(y_true, y_proba, threshold: float = 0.5) -> dict:
    """
    Evaluates classifier predictions using a specific decision threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    return {
        "threshold": threshold,
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "cm_array": cm.tolist(),
    }


def evaluate_anomaly_detector(y_true, anomaly_scores, raw_preds) -> dict:
    """
    Evaluates Isolation Forest predictions (-1 for anomaly/fraud, 1 for normal).
    Converts to 1 (fraud) and 0 (normal).
    """
    y_pred = (raw_preds == -1).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Isolation forest anomaly score (inverted so higher means more anomalous)
    proba_proxy = -anomaly_scores
    pr_auc = average_precision_score(y_true, proba_proxy)
    roc_auc = roc_auc_score(y_true, proba_proxy)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "cm_array": cm.tolist(),
    }


def plot_precision_recall_curves(results: dict, y_test, output_path: str = "credit_card_fraud_detection/reports/precision_recall_curves.png"):
    """
    Plots Precision-Recall curves comparing all models and random baseline.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(9, 7))

    # Baseline ratio (P(Fraud) = n_fraud / total)
    base_rate = float((y_test == 1).sum() / len(y_test))
    plt.axhline(y=base_rate, color="gray", linestyle="--", label=f"Random Chance (PR = {base_rate:.4f})")

    for name, res in results.items():
        if "y_proba" in res and res["y_proba"]:
            prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
            plt.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {res['pr_auc']:.3f})")

    plt.xlabel("Recall (Fraud Detected %)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (True Fraud %)", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall Curves (Extreme Imbalance Benchmark)", fontsize=13, fontweight="bold")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved Precision-Recall curves to {output_path}")


def plot_threshold_tuning(tuning_df: pd.DataFrame, optimal_thresh: float, output_path: str = "credit_card_fraud_detection/reports/threshold_tuning_curve.png"):
    """
    Plots Precision, Recall, and F1-score across all decision thresholds.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))

    plt.plot(tuning_df["threshold"], tuning_df["precision"], label="Precision", color="#3b82f6", lw=2)
    plt.plot(tuning_df["threshold"], tuning_df["recall"], label="Recall", color="#10b981", lw=2)
    plt.plot(tuning_df["threshold"], tuning_df["f1_score"], label="F1-Score", color="#ef4444", lw=2.5)

    plt.axvline(x=optimal_thresh, color="purple", linestyle="--", lw=1.8, label=f"Optimal F1 Threshold ({optimal_thresh:.2f})")
    plt.axvline(x=0.50, color="gray", linestyle=":", lw=1.5, label="Default Threshold (0.50)")

    plt.title("Decision Threshold Tuning Curve (Precision vs Recall Trade-off)", fontsize=13, fontweight="bold")
    plt.xlabel("Probability Threshold", fontsize=11)
    plt.ylabel("Metric Score", fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved threshold tuning curve to {output_path}")


def plot_confusion_matrices_grid(panels: list, output_path: str = "credit_card_fraud_detection/reports/confusion_matrices_comparison.png"):
    """
    Plots 2x2 grid comparing confusion matrices (Default vs Tuned vs SMOTE vs Anomaly).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, p in enumerate(panels[:4]):
        cm = np.array(p["cm_array"])
        tn, fp, fn, tp = cm.ravel()
        labels = [
            [f"TN (Legit)\n{tn}", f"FP (False Alarm)\n{fp}"],
            [f"FN (Missed Fraud)\n{fn}", f"TP (Caught Fraud)\n{tp}"],
        ]

        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Reds",
            cbar=False,
            ax=axes[idx],
            annot_kws={"size": 11, "weight": "bold"},
        )
        axes[idx].set_title(
            f"{p['title']}\nPrec: {p['precision']:.3f} | Rec: {p['recall']:.3f} | F1: {p['f1_score']:.3f}",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("Actual Label", fontsize=10)
        axes[idx].set_xticklabels(["Normal (0)", "Fraud (1)"])
        axes[idx].set_yticklabels(["Normal (0)", "Fraud (1)"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrices comparison to {output_path}")
