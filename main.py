"""
Main CLI Entrypoint for Credit Card Fraud Detection Project.
Runs dataset loading, SMOTE, Anomaly Detection, Classification, Threshold Tuning, and Test Inference.
"""

import sys
import os

# Ensure package root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_fraud_pipeline
from src.predict import FraudDetector


def main():
    print("=" * 70)
    print("       AI-POWERED CREDIT CARD FRAUD DETECTION SYSTEM         ")
    print("       SMOTE | Anomaly Detection | PR-AUC | Threshold Tuning  ")
    print("=" * 70)

    # 1. Run Complete Pipeline
    pipeline_res = run_fraud_pipeline()

    # 2. Detailed Threshold Tuning Output
    summary = pipeline_res["summary"]
    tuning_info = summary["threshold_tuning"]

    print("\n" + "=" * 70)
    print("       DECISION THRESHOLD TUNING & COST OPTIMIZATION         ")
    print("=" * 70)
    print(f"{'Threshold Profile':<22} | {'Thresh':<7} | {'Precision':<9} | {'Recall':<9} | {'F1 Score':<9} | {'Financial Cost':<14}")
    print("-" * 75)

    for profile_name, data in tuning_info.items():
        label = {
            "default_05": "Default Threshold",
            "best_f1": "Optimal F1 Threshold",
            "high_recall_90": "High-Recall (90%+)",
            "min_cost": "Minimum Cost",
        }.get(profile_name, profile_name)

        print(
            f"{label:<22} | {data['threshold']:<7.2f} | {data['precision']:<9.4f} | "
            f"{data['recall']:<9.4f} | {data['f1_score']:<9.4f} | ${data['total_cost']:<13.2f}"
        )

    print("-" * 75)

    # 3. Test Live Inference with FraudDetector
    print("\n" + "=" * 70)
    print("            SAMPLE TRANSACTION REAL-TIME INFERENCE           ")
    print("=" * 70)

    detector = FraudDetector()

    # Sample 1: Normal everyday transaction ($35 grocery)
    normal_tx = {"Time": 45000.0, "Amount": 35.50}
    for i in range(1, 29):
        normal_tx[f"V{i}"] = 0.05 * (i % 3)

    # Sample 2: Sophisticated Fraud Transaction (abnormal V14, V12, V4 shifts + $1,250 spike at 3 AM)
    fraud_tx = {"Time": 12500.0, "Amount": 1250.00}
    for i in range(1, 29):
        fraud_tx[f"V{i}"] = 0.0
    fraud_tx["V14"] = -6.2  # Strong fraud indicator
    fraud_tx["V12"] = -4.8
    fraud_tx["V4"] = 4.5
    fraud_tx["V11"] = 3.9

    for desc, tx in [
        ("Legitimate Transaction ($35.50 at grocery store)", normal_tx),
        ("High-Risk Fraudulent Transaction ($1,250 spike with anomalous PCA vectors)", fraud_tx),
    ]:
        res = detector.evaluate_transaction(tx)
        print(f"\n[Scenario]: {desc}")
        print(f"  -> Decision      : {res['is_fraud_decision']} (Probability: {res['fraud_probability_percentage']})")
        print(f"  -> Risk Tier     : {res['risk_tier']}")
        print(f"  -> Anomaly Flag  : {res['anomaly_detected']} (Score: {res['anomaly_score']})")
        print(f"  -> Banking Action: {res['recommended_banking_action']}")

    print("\n" + "=" * 70)
    print("  Artifacts Successfully Generated:")
    print("    - Supervised Model : credit_card_fraud_detection/models/best_classifier.joblib")
    print("    - Anomaly Detector : credit_card_fraud_detection/models/isolation_forest.joblib")
    print("    - Threshold Config : credit_card_fraud_detection/models/threshold_config.json")
    print("    - PR Curves Plot   : credit_card_fraud_detection/reports/precision_recall_curves.png")
    print("    - Threshold Curve  : credit_card_fraud_detection/reports/threshold_tuning_curve.png")
    print("    - Confusion Matrix : credit_card_fraud_detection/reports/confusion_matrices_comparison.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
