"""
Preprocessing & SMOTE Module for Credit Card Fraud Detection.
Applies RobustScaler to Amount & Time, stratified splitting, and SMOTE oversampling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def scale_features(df: pd.DataFrame, scaler: RobustScaler = None):
    """
    Applies RobustScaler to 'Amount' and 'Time' features.
    Returns transformed DataFrame and the fitted scaler.
    """
    df = df.copy()
    if scaler is None:
        scaler = RobustScaler()
        df[["Scaled_Amount", "Scaled_Time"]] = scaler.fit_transform(df[["Amount", "Time"]])
    else:
        df[["Scaled_Amount", "Scaled_Time"]] = scaler.transform(df[["Amount", "Time"]])

    # Drop original unscaled Time and Amount
    df = df.drop(columns=["Time", "Amount"])
    return df, scaler


def prepare_train_test_data(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42, smote_ratio: float = 0.20
):
    """
    Prepares train and test splits with strict leakage prevention:
    1. Stratified train/test split.
    2. RobustScaler fitted strictly on X_train, then transforming X_test.
    3. SMOTE over-sampling applied strictly to X_train.
    
    Returns:
        X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, scaler
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratified split to ensure rare fraud cases are evenly distributed
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit scaler strictly on training set
    scaler = RobustScaler()
    
    X_train = X_train_raw.copy()
    X_train[["Scaled_Amount", "Scaled_Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_train = X_train.drop(columns=["Time", "Amount"])

    X_test = X_test_raw.copy()
    X_test[["Scaled_Amount", "Scaled_Time"]] = scaler.transform(X_test[["Amount", "Time"]])
    X_test = X_test.drop(columns=["Time", "Amount"])

    # Apply SMOTE strictly on training set
    print(f"[*] Original Train class distribution: Normal={int((y_train == 0).sum())}, Fraud={int((y_train == 1).sum())}")
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"[+] SMOTE Resampled Train class distribution: Normal={int((y_train_smote == 0).sum())}, Fraud={int((y_train_smote == 1).sum())}")

    return X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, scaler
