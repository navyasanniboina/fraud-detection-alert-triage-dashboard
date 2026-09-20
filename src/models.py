"""
Models Module for Credit Card Fraud Detection.
Includes Anomaly Detection (Isolation Forest) and Supervised Classifiers (Logistic Regression, Random Forest, XGBoost).
"""

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_anomaly_detector(contamination: float = 0.01, random_state: int = 42) -> IsolationForest:
    """
    Returns an unsupervised Isolation Forest anomaly detector.
    Contamination is the expected proportion of outliers (fraud) in the data.
    """
    model = IsolationForest(
        n_estimators=150,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    return model


def get_supervised_models(scale_pos_weight: float = 1.0) -> dict:
    """
    Returns configured supervised classifiers for fraud detection.
    """
    models = {
        "Logistic Regression (Balanced)": LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        "Logistic Regression (SMOTE)": LogisticRegression(
            C=0.1,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost (Cost-Sensitive)": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        ),
    }
    return models
