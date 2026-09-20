"""
Dataset module for Credit Card Fraud Detection.
Loads existing Kaggle creditcard.csv or generates a statistically authentic imbalanced benchmark dataset.
"""

import os
import numpy as np
import pandas as pd


def generate_benchmark_credit_card_data(
    n_samples: int = 15000, fraud_ratio: float = 0.005, random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a statistically authentic synthetic credit card transaction dataset
    matching the real Kaggle benchmark schema (Time, V1-V28 PCA features, Amount, Class).
    Fraud transactions (~0.5%) exhibit realistic shifts in key PCA components (V4, V11, V12, V14).
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # 1. Time (0 to 172800 seconds = 48 hours)
    time_normal = np.sort(np.random.uniform(0, 172800, size=n_normal))
    # Fraud often clusters at night or in rapid bursts
    time_fraud = np.sort(np.random.uniform(20000, 150000, size=n_fraud))

    # 2. V1 to V28 PCA components (standard normal for legitimate transactions)
    v_normal = np.random.normal(0, 1, size=(n_normal, 28))

    # Fraudulent transactions exhibit key statistical anomalies:
    # In real credit card fraud datasets:
    # V14, V12, V10, V17 are heavily negatively correlated with fraud
    # V4, V11, V2, V19 are positively correlated with fraud
    v_fraud = np.random.normal(0, 1.2, size=(n_fraud, 28))
    v_fraud[:, 13] -= 5.5  # V14 (Index 13) strong negative anomaly
    v_fraud[:, 11] -= 4.2  # V12 (Index 11) negative anomaly
    v_fraud[:, 9] -= 3.8   # V10 (Index 9) negative anomaly
    v_fraud[:, 16] -= 3.2  # V17 (Index 16) negative anomaly
    v_fraud[:, 3] += 4.0   # V4 (Index 3) strong positive shift
    v_fraud[:, 10] += 3.5  # V11 (Index 10) positive shift
    v_fraud[:, 1] += 2.8   # V2 (Index 1) positive shift

    # 3. Transaction Amount ($)
    # Legitimate spending is right-skewed log-normal (coffee, groceries, bills: $5 - $250)
    amount_normal = np.round(np.random.lognormal(mean=3.2, sigma=1.2, size=n_normal), 2)
    amount_normal = np.clip(amount_normal, 0.5, 2500.0)

    # Fraudulent spending either tests cards with micro-charges ($1-$3) or takes large amounts ($300-$2000)
    amount_fraud = []
    for _ in range(n_fraud):
        if np.random.rand() < 0.25:
            amount_fraud.append(round(np.random.uniform(1.0, 5.0), 2))
        else:
            amount_fraud.append(round(np.random.uniform(180.0, 1850.0), 2))
    amount_fraud = np.array(amount_fraud)

    # 4. Classes
    class_normal = np.zeros(n_normal, dtype=int)
    class_fraud = np.ones(n_fraud, dtype=int)

    # Combine into dataframes
    v_cols = [f"V{i}" for i in range(1, 29)]

    df_normal = pd.DataFrame(v_normal, columns=v_cols)
    df_normal.insert(0, "Time", time_normal)
    df_normal["Amount"] = amount_normal
    df_normal["Class"] = class_normal

    df_fraud = pd.DataFrame(v_fraud, columns=v_cols)
    df_fraud.insert(0, "Time", time_fraud)
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = class_fraud

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    # Shuffle
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def load_credit_card_data(file_path: str = "credit_card_fraud_detection/data/raw_transactions.csv") -> pd.DataFrame:
    """
    Loads credit card transaction dataset. If file doesn't exist, generates
    realistic benchmark imbalanced transactions and saves to file_path.
    """
    if os.path.exists(file_path):
        print(f"[*] Loading transactions from {file_path}")
        df = pd.read_csv(file_path)
    else:
        print(f"[!] {file_path} not found. Generating benchmark imbalanced dataset...")
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        df = generate_benchmark_credit_card_data(n_samples=15000, fraud_ratio=0.005)
        df.to_csv(file_path, index=False)
        print(f"[+] Saved generated benchmark dataset to {file_path} (Shape: {df.shape})")

    return df
