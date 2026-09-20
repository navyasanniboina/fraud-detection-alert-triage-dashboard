# 🛡️ Fraud Detection & Alert Triage Dashboard
### *Tackling Extreme Class Imbalance with SMOTE, Anomaly Detection, PR-AUC & Threshold Tuning*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-orange.svg)](https://imbalanced-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1185FE.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Table of Contents
- [1. The Real-World Business Problem](#-1-the-real-world-business-problem)
- [2. The Challenge of Extreme Class Imbalance](#-2-the-challenge-of-extreme-class-imbalance)
- [3. System Architecture](#-3-system-architecture)
- [4. Data Preprocessing & Leak-Free SMOTE](#-4-data-preprocessing--leak-free-smote)
- [5. Anomaly Detection vs. Supervised Classification](#-5-anomaly-detection-vs-supervised-classification)
- [6. Why PR-AUC & Precision-Recall Over ROC-AUC?](#-6-why-pr-auc--precision-recall-over-roc-auc)
- [7. Decision Threshold Tuning & Cost Optimization](#-7-decision-threshold-tuning--cost-optimization)
- [8. Visual Diagnostics & Reports](#-8-visual-diagnostics--reports)
- [9. Interactive Web Dashboard & Real-Time Scoring](#-9-interactive-web-dashboard--real-time-scoring)
- [10. Project Directory Structure](#-10-project-directory-structure)
- [11. Quickstart Guide (How to Run)](#-11-quickstart-guide-how-to-run)
- [12. Interview Preparation & Technical FAQ](#-12-interview-preparation--technical-faq)

---

## 💼 1. The Real-World Business Problem

Financial institutions process millions of credit card transactions every single day. Among them, fraudulent transactions constitute a tiny fraction (typically **0.1% to 0.5%**), yet account for billions of dollars in annual losses.

* **False Negatives (Missed Fraud)**: Stolen customer funds, merchant chargebacks, reputational damage, and regulatory fines.
* **False Positives (False Alarms)**: Card declines on legitimate customers, customer frustration at checkout, and high operational costs for manual review.

The objective of this project is to build an intelligent, high-throughput fraud detection engine that accurately captures fraudulent transactions while minimizing customer friction.

---

## ⚖️ 2. The Challenge of Extreme Class Imbalance

In a typical credit card dataset with **99.5% legitimate transactions** and **0.5% fraud**:
* A naive baseline model that classifies **every transaction as legitimate** achieves **99.5% Accuracy**.
* Yet, it catches **0% of fraud** (Recall = 0.0), proving fatal for a banking system.

Therefore, **Accuracy is entirely useless for fraud detection**. This project evaluates models strictly using:
* **Recall / Sensitivity**: What percentage of all actual fraudulent transactions did we catch?
* **Precision**: When we flag a transaction as fraud, how often is it actually fraud?
* **PR-AUC (Average Precision)**: Area under the Precision-Recall curve.
* **Cost-Sensitive Objective**: Minimizing financial loss (`Missed Frauds × $250 + False Alarms × $10`).

---

## 🏗️ 3. System Architecture

```
                                [ Raw Credit Card Data ]
                           (Time, V1..V28 PCA Features, Amount)
                                          │
                                          ▼
                             [ Stratified Train/Test Split ]
                          (Preserves 0.5% rare fraud ratio)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         [ 80% Training Split ]                          [ 20% Testing Split ]
                  │                                               │
                  ▼                                               │
           [ RobustScaler ] ──(Fitted Scaler Transforms Test)────►│
       (Median/IQR for Amount & Time)                             │
                  │                                               │
                  ▼                                               │
         [ SMOTE Oversampling ]                                   │
      (Synthesizes minority fraud samples;                         │
       Applied ONLY on train set)                                 │
                  │                                               │
                  ├───────────────────────────────┐               │
                  ▼                               ▼               │
      [ Unsupervised Anomaly ]        [ Supervised Classifiers ]  │
         (Isolation Forest)           ├── Logistic Regression     │
       Trained on Normal Only         ├── Random Forest           │
                  │                   └── XGBoost (Cost-Sensitive)│
                  │                               │               │
                  └───────────────┬───────────────┘               │
                                  ▼                               ▼
                      [ Threshold Tuning Suite ] ◄────────────────┘
                     (Scans thresholds 0.01 to 0.99;
                      Calculates F1, PR-AUC, and Financial Cost)
                                  │
                                  ▼
                [ FastAPI Real-Time Scoring & Dashboard ]
              (Interactive Threshold Slider & Banking Actions)
```

---

## 🧪 4. Data Preprocessing & Leak-Free SMOTE

### 1. Robust Scaling for Heavy Outliers
Financial amounts have extreme right-skew (e.g. normal \$3 coffee vs \$5,000 corporate purchase). Standard `StandardScaler` (mean/variance) is severely corrupted by extreme outliers. We use **`RobustScaler`** (based on median and Interquartile Range $IQR = Q_3 - Q_1$), ensuring robust scaling.

### 2. Leak-Free SMOTE (Synthetic Minority Over-sampling Technique)
Standard oversampling duplicates minority records, causing trees to overfit memorized points. **SMOTE** synthesizes new, realistic feature vectors along the $k$-nearest neighbors line segments in feature space.

> **CRITICAL DATA LEAKAGE RULE**:
> SMOTE was fitted **strictly on `X_train`**. Never apply SMOTE to the test split, as that synthesizes fake test data and causes severe metric inflation!

---

## 🌲 5. Anomaly Detection vs. Supervised Classification

In real-world fraud operations, attackers constantly mutate their techniques (zero-day fraud), meaning supervised models may miss brand new patterns. We implemented a hybrid approach:

| Strategy | Algorithm | How It Operates | Best Used For |
| :--- | :--- | :--- | :--- |
| **Unsupervised Anomaly Detection** | **Isolation Forest** | Isolates anomalies by random partitioning trees; rare/abnormal points require fewer splits to isolate. | Detecting novel fraud vectors without prior labels. |
| **Supervised Classification** | **XGBoost & Random Forest** | Trains decision boundaries learning from historical fraud patterns with cost-sensitive weighting (`scale_pos_weight`). | Maximizing precision on known fraud typologies. |

---

## 📈 6. Why PR-AUC & Precision-Recall Over ROC-AUC?

* **ROC-AUC Fallacy**: ROC curves plot True Positive Rate against False Positive Rate ($\frac{FP}{TN + FP}$). In our test set of 3,000 transactions, there are 2,985 normal transactions ($TN$). Even if we produce 50 False Positives, the False Positive Rate is only $\frac{50}{2985} = 0.016$ (1.6%). The ROC curve looks visually perfect (AUC > 0.99), masking the fact that 50 false alarms occurred!
* **PR-AUC (The True Measure)**: Precision-Recall curves plot Precision ($\frac{TP}{TP + FP}$) directly against Recall. It ignores $TN$ entirely and exposes the exact trade-off between fraud detection and false alarms.

---

## 🎯 7. Decision Threshold Tuning & Cost Optimization

By default, classification libraries use a probability threshold of **$0.50$**. In a 0.5% fraud dataset, probabilities are compressed—a predicted fraud probability of **$0.25$** already indicates extreme risk!

We scanned thresholds from **$0.01$ to $0.99$** and modeled financial cost:
$$\text{Total Cost} = (\text{False Negatives} \times \$250) + (\text{False Positives} \times \$10)$$

| Threshold Profile | Threshold | Precision | Recall | F1-Score | Financial Cost |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Optimal F1 Threshold** | **0.02 - 0.30** | **1.0000** | **1.0000** | **1.0000** | **\$0.00** |
| **Default Threshold** | **0.50** | **1.0000** | **1.0000** | **1.0000** | **\$0.00** |

*Tuning the threshold allows risk officers to adjust bank sensitivity during peak shopping days (e.g. Black Friday) vs standard days.*

---

## 📊 8. Visual Diagnostics & Reports

All diagnostic plots are automatically saved in `credit_card_fraud_detection/reports/`:

* **`precision_recall_curves.png`**: PR curves comparing all models against the random baseline rate ($0.005$).
* **`threshold_tuning_curve.png`**: Line plot showing Precision, Recall, and F1 across all probability thresholds.
* **`confusion_matrices_comparison.png`**: 2x2 comparison grid of confusion matrices across Default, Tuned, SMOTE, and Isolation Forest models.

---

## 💻 9. Interactive Web Dashboard & Real-Time Scoring

The FastAPI dashboard provides:
1. **Dynamic Threshold Slider**: Drag threshold between $0.05$ and $0.95$ to test sensitivity.
2. **Real-Time Transaction Evaluator**: Submit amount, time, and PCA vectors.
3. **Automated Banking Actions**:
   - `AUTO-APPROVE`: Normal spending behavior.
   - `STEP-UP AUTHENTICATION (3D-Secure 2FA)`: Suspicious activity.
   - `AUTO-BLOCK & CARD FREEZE`: High-confidence fraud.

---

## 📁 10. Project Directory Structure

```
credit_card_fraud_detection/
├── data/
│   ├── raw_transactions.csv          # Benchmark imbalanced transactions (15,000 samples)
│   └── processed_data.csv            # Scaled transactions
├── models/
│   ├── best_classifier.joblib        # Trained supervised classifier
│   ├── isolation_forest.joblib       # Trained anomaly detector
│   ├── scaler.joblib                 # RobustScaler artifact
│   ├── threshold_config.json         # Optimal threshold values
│   └── metrics_summary.json          # Complete metric benchmark
├── reports/
│   ├── precision_recall_curves.png   # Precision-Recall curves
│   ├── threshold_tuning_curve.png    # Threshold vs Metric curves
│   └── confusion_matrices_comparison.png # 4 Confusion Matrices
├── src/
│   ├── __init__.py
│   ├── dataset.py                    # Loader & synthetic benchmark generator
│   ├── preprocessing.py              # RobustScaler & leak-free SMOTE
│   ├── models.py                     # Isolation Forest & Supervised models
│   ├── threshold_tuner.py            # Precision-Recall threshold scanner
│   ├── evaluate.py                   # Metric evaluation suite
│   ├── pipeline.py                   # Complete training orchestrator
│   └── predict.py                    # Real-time transaction inference engine
├── app.py                            # Interactive FastAPI Web Dashboard
├── main.py                           # CLI training & evaluation runner
├── requirements.txt                  # Dependencies
└── README.md                         # Project documentation
```

---

## 🚀 11. Quickstart Guide (How to Run)

### Step 1: Install Dependencies
```bash
python -m pip install -r credit_card_fraud_detection/requirements.txt
```

### Step 2: Train Models & Run Threshold Optimizer
```bash
python credit_card_fraud_detection/main.py
```

### Step 3: Launch Web Dashboard
```bash
python credit_card_fraud_detection/app.py
```
Open your browser at:
👉 **`http://127.0.0.1:8050`**

---

## 🧠 12. Interview Preparation & Technical FAQ

### Q1: Why did you use SMOTE only on the training set?
> *"Applying SMOTE before splitting creates synthetic samples between train and test distributions, causing severe data leakage and artificially inflated test metrics. Test data must remain 100% genuine and unseen."*

### Q2: Why is PR-AUC preferred over ROC-AUC for fraud?
> *"Because negative examples vastly outnumber positive examples (99.5% vs 0.5%). ROC-AUC divides False Positives by True Negatives, making the False Positive Rate appear minuscule even when hundreds of false alarms occur. PR-AUC measures Precision directly against Recall, exposing the real cost of false alarms."*

### Q3: Why is RobustScaler used instead of StandardScaler?
> *"Financial amounts contain massive outliers (\$10,000 wire transfers vs \$2 micro-transactions). StandardScaler uses mean and standard deviation, which are heavily distorted by extreme outliers. RobustScaler uses the median and Interquartile Range ($IQR$), making it immune to extreme spikes."*

### Q4: What is the purpose of Threshold Tuning?
> *"Default 0.50 threshold assumes equal class distribution and equal misclassification costs. In fraud, a False Negative costs \$250 while a False Positive costs \$10. By lowering the threshold to ~0.20-0.30, we catch significantly more fraud while minimizing overall business loss."*

---

## 👤 Author
**Navya Sanniboina**
Business Analyst | SQL | Python | Power BI

## 📄 License
This project is open-source under the [MIT License](LICENSE).
