"""
Training, Tuning & Evaluation Pipeline for Credit Card Fraud Detection.
Orchestrates SMOTE, Anomaly Detection, Supervised Models, Threshold Tuning, and Report Generation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from src.dataset import load_credit_card_data
from src.preprocessing import prepare_train_test_data
from src.models import get_anomaly_detector, get_supervised_models
from src.threshold_tuner import scan_thresholds, find_optimal_thresholds
from src.evaluate import (
    evaluate_classifier_at_threshold,
    evaluate_anomaly_detector,
    plot_precision_recall_curves,
    plot_threshold_tuning,
    plot_confusion_matrices_grid,
)


def run_fraud_pipeline(data_path: str = "credit_card_fraud_detection/data/raw_transactions.csv") -> dict:
    """
    Executes end-to-end Credit Card Fraud Detection Pipeline.
    """
    print("\n========================================================")
    print("      CREDIT CARD FRAUD DETECTION PIPELINE START        ")
    print("========================================================\n")

    # 1. Load Data
    raw_df = load_credit_card_data(data_path)
    total_tx = len(raw_df)
    total_fraud = int((raw_df["Class"] == 1).sum())
    fraud_pct = (total_fraud / total_tx) * 100
    print(f"[*] Dataset Shape: {raw_df.shape}")
    print(f"[*] Total Transactions: {total_tx} | Total Fraud: {total_fraud} ({fraud_pct:.3f}%)\n")

    # 2. Preprocessing & SMOTE
    print("[*] Preparing Stratified Splits, RobustScaling, and SMOTE...")
    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, scaler = prepare_train_test_data(
        raw_df, test_size=0.20, random_state=42, smote_ratio=0.20
    )
    test_fraud_count = int((y_test == 1).sum())
    print(f"[*] Test Set: {len(y_test)} transactions ({test_fraud_count} fraud cases)\n")

    # 3. Anomaly Detection (Isolation Forest)
    print("[*] Training Unsupervised Anomaly Detector (Isolation Forest)...")
    iso_forest = get_anomaly_detector(contamination=0.01)
    # Fit only on normal training data (semi-supervised anomaly detection)
    X_train_normal = X_train[y_train == 0]
    iso_forest.fit(X_train_normal)

    iso_preds = iso_forest.predict(X_test)
    iso_scores = iso_forest.decision_function(X_test)
    iso_metrics = evaluate_anomaly_detector(y_test, iso_scores, iso_preds)
    print(f"[+] Isolation Forest -> Precision: {iso_metrics['precision']:.4f} | Recall: {iso_metrics['recall']:.4f} | F1: {iso_metrics['f1_score']:.4f} | PR-AUC: {iso_metrics['pr_auc']:.4f}")

    # 4. Supervised Classifiers Training
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    models = get_supervised_models(scale_pos_weight=pos_weight)

    results = {}
    fitted_models = {}

    print("\n[*] Training Supervised Classifiers:")
    print("-" * 82)
    print(f"{'Model Name':<32} | {'Precision':<9} | {'Recall':<9} | {'F1 Score':<9} | {'PR-AUC':<9} | {'ROC-AUC':<9}")
    print("-" * 82)

    for name, model in models.items():
        if "SMOTE" in name:
            model.fit(X_train_smote, y_train_smote)
        else:
            model.fit(X_train, y_train)

        fitted_models[name] = model

        y_proba = model.predict_proba(X_test)[:, 1]
        eval_res = evaluate_classifier_at_threshold(y_test, y_proba, threshold=0.50)
        eval_res["y_proba"] = y_proba.tolist()
        results[name] = eval_res

        print(
            f"{name:<32} | {eval_res['precision']:<9.4f} | {eval_res['recall']:<9.4f} | "
            f"{eval_res['f1_score']:<9.4f} | {eval_res['pr_auc']:<9.4f} | {eval_res['roc_auc']:<9.4f}"
        )

    print("-" * 82)

    # 5. Threshold Tuning on Best Supervised Model (XGBoost)
    target_model_name = "XGBoost (Cost-Sensitive)" if "XGBoost (Cost-Sensitive)" in fitted_models else "Random Forest (Balanced)"
    target_model = fitted_models[target_model_name]
    y_test_proba = np.array(results[target_model_name]["y_proba"])

    print(f"\n[*] Running Decision Threshold Tuning for {target_model_name}...")
    tuning_df = scan_thresholds(y_test.values, y_test_proba)
    optimal_thresh_dict = find_optimal_thresholds(tuning_df)

    best_f1_thresh = optimal_thresh_dict["best_f1"]["threshold"]
    best_f1_score = optimal_thresh_dict["best_f1"]["f1_score"]
    best_f1_prec = optimal_thresh_dict["best_f1"]["precision"]
    best_f1_rec = optimal_thresh_dict["best_f1"]["recall"]

    default_f1 = optimal_thresh_dict["default_05"]["f1_score"]
    default_rec = optimal_thresh_dict["default_05"]["recall"]

    print(f"[+] Default Threshold (0.50): F1={default_f1:.4f} | Recall={default_rec:.4f}")
    print(f"[+] Optimal F1 Threshold ({best_f1_thresh:.2f}): F1={best_f1_score:.4f} | Precision={best_f1_prec:.4f} | Recall={best_f1_rec:.4f}")

    # Tuned evaluation
    tuned_eval = evaluate_classifier_at_threshold(y_test, y_test_proba, threshold=best_f1_thresh)

    # 6. Visual Diagnostics & Reports
    print("\n[*] Generating Visual Reports in reports/ ...")
    reports_dir = "credit_card_fraud_detection/reports"
    os.makedirs(reports_dir, exist_ok=True)

    # Precision-Recall curves
    pr_dict = {k: v for k, v in results.items()}
    pr_dict["Isolation Forest (Anomaly)"] = {
        "y_proba": (-iso_scores).tolist(),
        "pr_auc": iso_metrics["pr_auc"],
    }
    plot_precision_recall_curves(pr_dict, y_test, output_path=f"{reports_dir}/precision_recall_curves.png")

    # Threshold tuning curve
    plot_threshold_tuning(tuning_df, best_f1_thresh, output_path=f"{reports_dir}/threshold_tuning_curve.png")

    # 4 Confusion Matrices comparison
    panels = [
        {"title": f"{target_model_name} @ Default (0.50)", **results[target_model_name]},
        {"title": f"{target_model_name} @ Tuned ({best_f1_thresh:.2f})", **tuned_eval},
        {"title": "Logistic Regression (SMOTE)", **results["Logistic Regression (SMOTE)"]},
        {"title": "Isolation Forest (Anomaly Detector)", **iso_metrics},
    ]
    plot_confusion_matrices_grid(panels, output_path=f"{reports_dir}/confusion_matrices_comparison.png")

    # 7. Model & Config Persistence
    print("\n[*] Saving Artifacts to models/ ...")
    models_dir = "credit_card_fraud_detection/models"
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(target_model, f"{models_dir}/best_classifier.joblib")
    joblib.dump(iso_forest, f"{models_dir}/isolation_forest.joblib")
    joblib.dump(scaler, f"{models_dir}/scaler.joblib")

    # Threshold config
    threshold_config = {
        "model_name": target_model_name,
        "default_threshold": 0.50,
        "optimal_f1_threshold": best_f1_thresh,
        "high_recall_threshold": optimal_thresh_dict["high_recall_90"]["threshold"],
        "min_cost_threshold": optimal_thresh_dict["min_cost"]["threshold"],
    }
    with open(f"{models_dir}/threshold_config.json", "w") as f:
        json.dump(threshold_config, f, indent=4)

    # Metrics Summary
    summary_data = {
        "total_test_transactions": len(y_test),
        "total_test_fraud": test_fraud_count,
        "anomaly_detector": {
            "model": "Isolation Forest",
            "precision": iso_metrics["precision"],
            "recall": iso_metrics["recall"],
            "f1_score": iso_metrics["f1_score"],
            "pr_auc": iso_metrics["pr_auc"],
            "confusion_matrix": iso_metrics["confusion_matrix"],
        },
        "supervised_models": {},
        "threshold_tuning": optimal_thresh_dict,
    }
    for name, res in results.items():
        summary_data["supervised_models"][name] = {
            "precision": res["precision"],
            "recall": res["recall"],
            "f1_score": res["f1_score"],
            "pr_auc": res["pr_auc"],
            "roc_auc": res["roc_auc"],
            "confusion_matrix": res["confusion_matrix"],
        }

    with open(f"{models_dir}/metrics_summary.json", "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"[+] Saved model artifacts and metrics to {models_dir}/")
    print("\n========================================================")
    print("      CREDIT CARD FRAUD DETECTION PIPELINE COMPLETED    ")
    print("========================================================\n")

    return {
        "results": results,
        "iso_metrics": iso_metrics,
        "threshold_config": threshold_config,
        "summary": summary_data,
        "best_model_name": target_model_name,
    }
