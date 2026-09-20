"""
Interactive Web Application for Credit Card Fraud Detection.
Features real-time transaction scoring and an interactive Decision Threshold Tuning Slider.
Run with: python credit_card_fraud_detection/app.py
"""

import os
import sys
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
from typing import Optional, Dict

# Ensure local package import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import FraudDetector


app = FastAPI(title="Credit Card Fraud Detection Engine")

# Mount reports directory for images
reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(reports_dir, exist_ok=True)
app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")

detector = None


def get_detector():
    global detector
    if detector is None:
        try:
            detector = FraudDetector()
        except Exception:
            detector = None
    return detector


class TransactionInput(BaseModel):
    Time: float = 12500.0
    Amount: float = 1250.0
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 4.5
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = -3.8
    V11: float = 3.9
    V12: float = -4.8
    V13: float = 0.0
    V14: float = -6.2
    V15: float = 0.0
    V16: float = 0.0
    V17: float = -3.2
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    custom_threshold: Optional[float] = None


@app.get("/api/metrics")
async def get_metrics():
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return {"error": "Models have not been trained yet. Please run main.py first."}


@app.post("/api/predict")
async def score_transaction(tx: TransactionInput):
    det = get_detector()
    if det is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Model not trained yet. Run `python credit_card_fraud_detection/main.py` first."},
        )
    data_dict = tx.model_dump()
    custom_thresh = data_dict.pop("custom_threshold", None)
    result = det.evaluate_transaction(data_dict, custom_threshold=custom_thresh)
    return result


@app.get("/", response_class=HTMLResponse)
async def home_page():
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/metrics_summary.json")
    metrics_json = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics_json = json.load(f)

    models_data = metrics_json.get("supervised_models", {})
    iso_data = metrics_json.get("anomaly_detector", {})
    tuning_data = metrics_json.get("threshold_tuning", {})

    table_rows = ""
    for name, data in models_data.items():
        table_rows += f"""
        <tr>
            <td style="font-weight: 600;">{name}</td>
            <td>{data['precision']:.4f}</td>
            <td>{data['recall']:.4f}</td>
            <td style="font-weight: 700; color: #ef4444;">{data['f1_score']:.4f}</td>
            <td style="font-weight: 700; color: #3b82f6;">{data['pr_auc']:.4f}</td>
            <td>{data['roc_auc']:.4f}</td>
        </tr>
        """
    if iso_data:
        table_rows += f"""
        <tr style="background: rgba(147, 51, 234, 0.1);">
            <td style="font-weight: 600;">Isolation Forest (Unsupervised Anomaly)</td>
            <td>{iso_data['precision']:.4f}</td>
            <td>{iso_data['recall']:.4f}</td>
            <td style="font-weight: 700; color: #ef4444;">{iso_data['f1_score']:.4f}</td>
            <td style="font-weight: 700; color: #3b82f6;">{iso_data['pr_auc']:.4f}</td>
            <td>-</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Credit Card Fraud Detection Engine</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #090d16;
                --card-bg: #131b2e;
                --card-border: #1e293b;
                --primary: #ef4444;
                --primary-hover: #dc2626;
                --text: #f8fafc;
                --text-muted: #94a3b8;
                --accent: #3b82f6;
            }}
            * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }}
            body {{ background-color: var(--bg); color: var(--text); padding: 24px; line-height: 1.5; }}
            .container {{ max-width: 1280px; margin: 0 auto; }}
            header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 20px; border-bottom: 1px solid var(--card-border); }}
            .header-title h1 {{ font-size: 26px; font-weight: 800; background: linear-gradient(135deg, #f87171, #fb923c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
            .header-title p {{ color: var(--text-muted); font-size: 14px; margin-top: 4px; }}
            .badge {{ display: inline-block; padding: 6px 14px; border-radius: 9999px; font-size: 12px; font-weight: 600; background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }}
            @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
            .card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px; padding: 24px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4); }}
            .card-title {{ font-size: 18px; font-weight: 700; margin-bottom: 16px; display: flex; align-items: center; gap: 10px; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
            th {{ color: var(--text-muted); font-weight: 600; padding: 10px 8px; border-bottom: 1px solid var(--card-border); }}
            td {{ padding: 12px 8px; border-bottom: 1px solid rgba(51, 65, 85, 0.5); }}
            .form-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
            .form-group {{ display: flex; flex-direction: column; gap: 6px; }}
            label {{ font-size: 12px; font-weight: 600; color: var(--text-muted); }}
            input {{ background: #090d16; border: 1px solid var(--card-border); color: var(--text); padding: 9px 12px; border-radius: 8px; font-size: 13px; outline: none; }}
            input:focus {{ border-color: var(--primary); }}
            .btn {{ background: var(--primary); color: white; border: none; padding: 14px 20px; border-radius: 10px; font-weight: 700; cursor: pointer; width: 100%; margin-top: 18px; font-size: 15px; transition: all 0.2s; }}
            .btn:hover {{ background: var(--primary-hover); transform: translateY(-1px); }}
            .slider-box {{ background: rgba(9, 13, 22, 0.6); padding: 14px; border-radius: 10px; border: 1px solid var(--card-border); margin: 14px 0; }}
            .result-card {{ display: none; background: rgba(9, 13, 22, 0.8); border-radius: 12px; padding: 18px; margin-top: 20px; border: 1px solid var(--card-border); }}
            .risk-pill {{ padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 12px; }}
            .progress-bar-bg {{ background: #1e293b; border-radius: 9999px; height: 12px; overflow: hidden; margin: 10px 0 16px 0; }}
            .progress-bar-fill {{ height: 100%; width: 0%; border-radius: 9999px; transition: width 0.6s ease; }}
            .img-container {{ border-radius: 12px; overflow: hidden; border: 1px solid var(--card-border); margin-top: 12px; }}
            .img-container img {{ width: 100%; display: block; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div class="header-title">
                    <h1>Credit Card Fraud Detection Engine</h1>
                    <p>SMOTE • Anomaly Detection • Precision-Recall Optimization • Dynamic Threshold Tuning</p>
                </div>
                <div>
                    <span class="badge">Real-Time Risk Scoring</span>
                </div>
            </header>

            <!-- Benchmark Table -->
            <div class="card" style="margin-bottom: 24px;">
                <div class="card-title">
                    <span>📊</span> Model Benchmark Comparison (Test Set with Extreme Imbalance ~0.5%)
                </div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Model & Architecture</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1 Score</th>
                                <th>PR-AUC (Avg Precision)</th>
                                <th>ROC-AUC</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows or '<tr><td colspan="6">Run main.py to train models and generate metrics.</td></tr>'}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Main Interactive Section -->
            <div class="grid">
                <!-- Transaction Evaluation Form -->
                <div class="card">
                    <div class="card-title">
                        <span>🛡️</span> Real-Time Transaction Fraud Scorer
                    </div>

                    <!-- Dynamic Threshold Slider -->
                    <div class="slider-box">
                        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 700; margin-bottom: 6px;">
                            <span>Decision Threshold: <span id="thresholdVal" style="color: #f87171;">0.30</span></span>
                            <span style="font-size: 11px; color: var(--text-muted);">Lower = Higher Recall (Catches more fraud)</span>
                        </div>
                        <input type="range" id="threshSlider" min="0.05" max="0.95" step="0.05" value="0.30" style="width: 100%; cursor: pointer;">
                    </div>

                    <form id="txForm">
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Transaction Amount ($)</label>
                                <input type="number" step="0.5" id="Amount" value="1250.00">
                            </div>
                            <div class="form-group">
                                <label>Time (Seconds from midnight)</label>
                                <input type="number" id="Time" value="12500">
                            </div>
                            <div class="form-group">
                                <label>V14 (Strong Negative Anomaly)</label>
                                <input type="number" step="0.1" id="V14" value="-6.2">
                            </div>
                            <div class="form-group">
                                <label>V12 (Negative Anomaly)</label>
                                <input type="number" step="0.1" id="V12" value="-4.8">
                            </div>
                            <div class="form-group">
                                <label>V4 (Positive Spike)</label>
                                <input type="number" step="0.1" id="V4" value="4.5">
                            </div>
                            <div class="form-group">
                                <label>V11 (Positive Shift)</label>
                                <input type="number" step="0.1" id="V11" value="3.9">
                            </div>
                        </div>

                        <div style="margin-top: 10px; font-size: 12px; color: var(--text-muted);">
                            Quick Profiles: 
                            <a href="#" id="loadFraudBtn" style="color: #f87171; text-decoration: underline; margin-right: 10px;">Load Suspicious Transaction</a>
                            <a href="#" id="loadNormalBtn" style="color: #34d399; text-decoration: underline;">Load Normal Coffee Purchase</a>
                        </div>

                        <button type="submit" class="btn" id="scoreBtn">Analyze Transaction Risk</button>
                    </form>

                    <!-- Result Box -->
                    <div class="result-card" id="resultCard">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div>
                                <span style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Engine Verdict</span>
                                <h3 id="verdictText" style="font-size: 20px; font-weight: 800;">-</h3>
                            </div>
                            <span class="risk-pill" id="riskPill">-</span>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600;">
                                <span>Fraud Probability</span>
                                <span id="probText">0%</span>
                            </div>
                            <div class="progress-bar-bg">
                                <div class="progress-bar-fill" id="probFill"></div>
                            </div>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.04); padding: 12px; border-radius: 8px; margin-top: 10px; font-size: 12px;">
                            <div><strong>Anomaly Detector:</strong> <span id="anomalyStatus">-</span></div>
                            <div style="margin-top: 6px;"><strong>Automated Banking Action:</strong> <span id="actionText" style="color: #cbd5e1;">-</span></div>
                        </div>
                    </div>
                </div>

                <!-- Visual Charts View -->
                <div class="card">
                    <div class="card-title">
                        <span>📈</span> Diagnostic Curves & Threshold Tuning
                    </div>
                    <div class="img-container">
                        <img src="/reports/precision_recall_curves.png" onerror="this.src='https://placehold.co/600x400/131b2e/94a3b8?text=Run+main.py+to+generate+PR+curves'" alt="PR Curves">
                    </div>
                    <div class="img-container" style="margin-top: 16px;">
                        <img src="/reports/threshold_tuning_curve.png" onerror="this.src='https://placehold.co/600x400/131b2e/94a3b8?text=Run+main.py+to+generate+tuning+curve'" alt="Threshold Curve">
                    </div>
                </div>
            </div>
        </div>

        <script>
            const form = document.getElementById('txForm');
            const slider = document.getElementById('threshSlider');
            const threshVal = document.getElementById('thresholdVal');
            const resultCard = document.getElementById('resultCard');
            const verdictText = document.getElementById('verdictText');
            const riskPill = document.getElementById('riskPill');
            const probText = document.getElementById('probText');
            const probFill = document.getElementById('probFill');
            const anomalyStatus = document.getElementById('anomalyStatus');
            const actionText = document.getElementById('actionText');
            const scoreBtn = document.getElementById('scoreBtn');

            slider.addEventListener('input', () => {{
                threshVal.innerText = parseFloat(slider.value).toFixed(2);
                if (resultCard.style.display === 'block') {{
                    form.dispatchEvent(new Event('submit'));
                }}
            }});

            document.getElementById('loadFraudBtn').addEventListener('click', (e) => {{
                e.preventDefault();
                document.getElementById('Amount').value = "1250.00";
                document.getElementById('Time').value = "12500";
                document.getElementById('V14').value = "-6.2";
                document.getElementById('V12').value = "-4.8";
                document.getElementById('V4').value = "4.5";
                document.getElementById('V11').value = "3.9";
                form.dispatchEvent(new Event('submit'));
            }});

            document.getElementById('loadNormalBtn').addEventListener('click', (e) => {{
                e.preventDefault();
                document.getElementById('Amount').value = "4.75";
                document.getElementById('Time').value = "45000";
                document.getElementById('V14').value = "0.2";
                document.getElementById('V12').value = "0.1";
                document.getElementById('V4').value = "-0.1";
                document.getElementById('V11').value = "0.0";
                form.dispatchEvent(new Event('submit'));
            }});

            form.addEventListener('submit', async (e) => {{
                e.preventDefault();
                scoreBtn.disabled = true;
                scoreBtn.innerText = "Analyzing Risk...";

                const payload = {{
                    Amount: parseFloat(document.getElementById('Amount').value),
                    Time: parseFloat(document.getElementById('Time').value),
                    V4: parseFloat(document.getElementById('V4').value),
                    V11: parseFloat(document.getElementById('V11').value),
                    V12: parseFloat(document.getElementById('V12').value),
                    V14: parseFloat(document.getElementById('V14').value),
                    custom_threshold: parseFloat(slider.value)
                }};
                // fill remaining V components with 0.0
                for (let i = 1; i <= 28; i++) {{
                    if (!payload.hasOwnProperty('V' + i)) {{
                        payload['V' + i] = 0.0;
                    }}
                }}

                try {{
                    const response = await fetch('/api/predict', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(payload)
                    }});
                    const data = await response.json();

                    if (data.error) {{
                        alert(data.error);
                        return;
                    }}

                    resultCard.style.display = 'block';
                    verdictText.innerText = data.is_fraud_decision;
                    riskPill.innerText = data.risk_tier;
                    riskPill.style.background = data.risk_color + '25';
                    riskPill.style.color = data.risk_color;
                    riskPill.style.border = '1px solid ' + data.risk_color;

                    probText.innerText = data.fraud_probability_percentage;
                    probFill.style.width = data.fraud_probability_percentage;
                    probFill.style.background = data.risk_color;

                    anomalyStatus.innerText = data.anomaly_detected ? "🚨 Outlier Flagged (Score: " + data.anomaly_score + ")" : "✅ Normal Transaction Geometry";
                    anomalyStatus.style.color = data.anomaly_detected ? "#ef4444" : "#10b981";

                    actionText.innerText = data.recommended_banking_action;

                }} catch (err) {{
                    console.error(err);
                }} finally {{
                    scoreBtn.disabled = false;
                    scoreBtn.innerText = "Analyze Transaction Risk";
                }}
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8050, reload=False)
