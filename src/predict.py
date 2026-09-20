"""
Inference & Fraud Risk Scoring Module for Credit Card Transactions.
Loads trained classifier, isolation forest, and scaler to evaluate transactions in real-time.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np


class FraudDetector:
    def __init__(
        self,
        classifier_path: str = "credit_card_fraud_detection/models/best_classifier.joblib",
        iso_path: str = "credit_card_fraud_detection/models/isolation_forest.joblib",
        scaler_path: str = "credit_card_fraud_detection/models/scaler.joblib",
        config_path: str = "credit_card_fraud_detection/models/threshold_config.json",
    ):
        if not os.path.exists(classifier_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Fraud model artifacts not found. Please run main.py first!")

        self.classifier = joblib.load(classifier_path)
        self.iso_forest = joblib.load(iso_path) if os.path.exists(iso_path) else None
        self.scaler = joblib.load(scaler_path)

        self.threshold = 0.50
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
                self.threshold = cfg.get("optimal_f1_threshold", 0.50)

    def evaluate_transaction(self, transaction: dict, custom_threshold: float = None) -> dict:
        """
        Scores a single credit card transaction.
        """
        thresh = custom_threshold if custom_threshold is not None else self.threshold

        df = pd.DataFrame([transaction])
        
        # RobustScale Amount and Time
        scaled_at = self.scaler.transform(df[["Amount", "Time"]])
        df["Scaled_Amount"] = scaled_at[:, 0]
        df["Scaled_Time"] = scaled_at[:, 1]
        
        # Order columns: V1..V28, Scaled_Amount, Scaled_Time
        v_cols = [f"V{i}" for i in range(1, 29)]
        ordered_cols = v_cols + ["Scaled_Amount", "Scaled_Time"]
        X_eval = df[ordered_cols]

        # 1. Supervised Classifier Probability
        prob = float(self.classifier.predict_proba(X_eval)[0][1])
        is_fraud = bool(prob >= thresh)

        # 2. Anomaly Score from Isolation Forest
        anomaly_flag = False
        raw_anomaly_score = 0.0
        if self.iso_forest is not None:
            raw_anomaly_score = float(self.iso_forest.decision_function(X_eval)[0])
            anomaly_flag = bool(self.iso_forest.predict(X_eval)[0] == -1)

        # 3. Risk Tier & Banking Actions
        if prob >= 0.70 or (prob >= thresh and anomaly_flag):
            risk_tier = "CRITICAL FRAUD RISK"
            color = "#ef4444"
            action = "AUTO-BLOCK TRANSACTION: Immediate card freeze and automated SMS fraud confirmation required."
        elif prob >= thresh:
            risk_tier = "SUSPICIOUS ACTIVITY"
            color = "#f59e0b"
            action = "STEP-UP AUTHENTICATION: Trigger 3D-Secure 2FA / Biometric verification prompt."
        elif anomaly_flag:
            risk_tier = "UNUSUAL BEHAVIOR (OUTLIER)"
            color = "#3b82f6"
            action = "FLAG FOR MANUAL AUDIT: Transaction deviates from customer baseline profile."
        else:
            risk_tier = "LOW RISK (NORMAL)"
            color = "#10b981"
            action = "AUTO-APPROVE: Transaction parameters match normal verified spending behavior."

        return {
            "fraud_probability": round(prob, 4),
            "fraud_probability_percentage": f"{prob * 100:.2f}%",
            "decision_threshold_applied": round(thresh, 3),
            "is_fraud_decision": "FRAUD DETECTED" if is_fraud else "LEGITIMATE",
            "anomaly_detected": anomaly_flag,
            "anomaly_score": round(raw_anomaly_score, 4),
            "risk_tier": risk_tier,
            "risk_color": color,
            "recommended_banking_action": action,
        }
