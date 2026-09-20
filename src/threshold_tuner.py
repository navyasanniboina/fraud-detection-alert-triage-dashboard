"""
Threshold Tuning Module for Credit Card Fraud Detection.
Optimizes decision thresholds across Precision-Recall curves, F1-scores, and cost-benefit trade-offs.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def scan_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = None,
    cost_fn: float = 250.0,
    cost_fp: float = 10.0,
) -> pd.DataFrame:
    """
    Scans probability thresholds from 0.01 to 0.99 and calculates
    Precision, Recall, F1-Score, and Financial Cost for each.
    
    Args:
        cost_fn: Financial damage of a False Negative (stolen fraud amount).
        cost_fp: Operational cost of a False Positive (SMS alert, customer verification, friction).
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    records = []
    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        # Financial business cost = (Missed Frauds * $250) + (False Alarms * $10)
        total_cost = (fn * cost_fn) + (fp * cost_fp)

        records.append(
            {
                "threshold": round(float(thresh), 3),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "total_cost": round(float(total_cost), 2),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )

    return pd.DataFrame(records)


def find_optimal_thresholds(tuning_df: pd.DataFrame) -> dict:
    """
    Identifies key decision thresholds:
    1. Max F1 Threshold (Balanced precision/recall)
    2. Min Cost Threshold (Minimal financial damage)
    3. High-Recall Threshold (At least 90% fraud caught)
    """
    # Best F1
    max_f1_row = tuning_df.loc[tuning_df["f1_score"].idxmax()]
    
    # Min Cost
    min_cost_row = tuning_df.loc[tuning_df["total_cost"].idxmin()]

    # High recall >= 90% with highest precision
    high_recall_candidates = tuning_df[tuning_df["recall"] >= 0.90]
    if not high_recall_candidates.empty:
        high_recall_row = high_recall_candidates.loc[high_recall_candidates["precision"].idxmax()]
    else:
        high_recall_row = max_f1_row

    # Default 0.5 threshold
    default_row = tuning_df.iloc[(tuning_df["threshold"] - 0.50).abs().argsort()[:1]].iloc[0]

    return {
        "best_f1": max_f1_row.to_dict(),
        "min_cost": min_cost_row.to_dict(),
        "high_recall_90": high_recall_row.to_dict(),
        "default_05": default_row.to_dict(),
    }
