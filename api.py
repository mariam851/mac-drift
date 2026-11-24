"""
Mondrian Adaptive Conformal Prediction (MAC-Drift)
Module: Production Real-Time Streaming Inference, Calibration API & Interactive UI
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from typing import List, Dict, Any
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.data.dataset_loader import TemporalPELoader
from src.models.ensemble_manager import LOOEnsembleManager
from src.conformal.nonconformity_scores import compute_aps_scores
from src.conformal.mac_drift import MACDriftClassifier


# Global service state maintaining the streaming model, dataset cache, and adaptive buffers
state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the LOO ensemble and calibrates MAC-Drift on startup using warmup stream data.
    """
    dataset_path = "data/processed/ember_stream.parquet"
    warmup_samples = 5000
    alpha = 0.10

    loader = TemporalPELoader(filepath=dataset_path)
    X_raw, y_raw, _ = loader.load_stream()

    X = np.ascontiguousarray(np.asarray(X_raw, dtype=np.float32))
    y = np.asarray(y_raw, dtype=np.int64).ravel()

    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    ensemble_mgr = LOOEnsembleManager(n_estimators=20, subsample_ratio=0.8, seed=42)
    loo_probs, _ = ensemble_mgr.fit_and_get_loo_probs(X[:warmup_samples], y[:warmup_samples])
    warmup_scores = compute_aps_scores(loo_probs, y[:warmup_samples], randomized=False)

    mac_model = MACDriftClassifier(num_classes=n_classes, alpha=alpha, delta=0.05)
    mac_model.update_calibration_stream(warmup_scores, y[:warmup_samples])
    mac_model.recalibrate_adaptive_quantiles()

    state["ensemble"] = ensemble_mgr
    state["mac_drift"] = mac_model
    state["X_stream"] = X
    state["y_stream"] = y
    state["warmup_samples"] = warmup_samples
    state["n_features"] = n_features
    state["n_classes"] = n_classes
    state["alpha"] = alpha
    state["stream_counter"] = warmup_samples

    print(f"[✓] MAC-Drift API & UI Ready | Features: {n_features} | Classes: {n_classes}")
    yield
    state.clear()


app = FastAPI(
    title="MAC-Drift Malware Triage API",
    description="Mondrian Adaptive Conformal Inference for Drifting Malware Streams",
    version="1.0.0",
    lifespan=lifespan
)


# --- Pydantic Request/Response Schemas ---

class PredictRequest(BaseModel):
    features: List[List[float]] = Field(
        ...,
        description="2D array of static PE feature vectors with shape (N, D)."
    )


class PredictResponse(BaseModel):
    prediction_sets: List[List[int]]
    set_sizes: List[int]
    probabilities: List[List[float]]
    is_ambiguous: List[bool]


class CalibrateRequest(BaseModel):
    features: List[List[float]] = Field(
        ...,
        description="2D array of newly verified PE feature vectors with shape (N, D)."
    )
    true_labels: List[int] = Field(
        ...,
        description="Ground-truth malware family indices corresponding to the features."
    )


# --- API Endpoints ---

@app.get("/status")
def get_system_status():
    """
    Returns the current adaptive calibration state and per-class Mondrian quantiles.
    """
    mac: MACDriftClassifier = state["mac_drift"]
    quantiles = np.asarray(mac.current_class_quantiles).tolist()
    return {
        "status": "online",
        "target_coverage": round(1.0 - state["alpha"], 2),
        "expected_features": state["n_features"],
        "num_classes": state["n_classes"],
        "processed_stream_samples": state["stream_counter"],
        "per_class_quantiles": {f"class_{c}": round(q, 4) for c, q in enumerate(quantiles)}
    }


@app.get("/sample/{index}")
def get_stream_sample(index: int):
    """
    Fetches a real PE feature vector and its true label from the test stream by index.
    """
    X = state["X_stream"]
    y = state["y_stream"]
    if index < 0 or index >= len(X):
        raise HTTPException(status_code=404, detail=f"Index must be between 0 and {len(X) - 1}.")
    return {
        "index": index,
        "true_label": int(y[index]),
        "features": X[index].tolist()
    }


@app.post("/predict", response_model=PredictResponse)
def predict_malware_sets(req: PredictRequest):
    """
    Evaluates incoming PE feature vectors and outputs guaranteed conformal prediction sets.
    """
    X_in = np.ascontiguousarray(np.asarray(req.features, dtype=np.float32))
    if X_in.ndim != 2 or X_in.shape[1] != state["n_features"]:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 2D array with {state['n_features']} features, got {X_in.shape}."
        )

    ensemble: LOOEnsembleManager = state["ensemble"]
    mac: MACDriftClassifier = state["mac_drift"]

    probs = ensemble.predict_test_proba(X_in)
    pred_sets = [[int(c) for c in s] for s in mac.predict_sets(probs)]
    set_sizes = [len(s) for s in pred_sets]
    is_ambiguous = [sz > 1 for sz in set_sizes]

    return PredictResponse(
        prediction_sets=pred_sets,
        set_sizes=set_sizes,
        probabilities=np.round(probs, 4).tolist(),
        is_ambiguous=is_ambiguous
    )


@app.post("/calibrate")
def online_recalibrate(req: CalibrateRequest):
    """
    Updates the MAC-Drift adaptive memory buffers and recomputes Lepski-selected quantiles.
    """
    X_cal = np.ascontiguousarray(np.asarray(req.features, dtype=np.float32))
    y_cal = np.asarray(req.true_labels, dtype=np.int64).ravel()

    if len(X_cal) != len(y_cal):
        raise HTTPException(
            status_code=400,
            detail="Mismatch between number of feature rows and true_labels length."
        )

    if X_cal.ndim != 2 or X_cal.shape[1] != state["n_features"]:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 2D array with {state['n_features']} features, got {X_cal.shape}."
        )

    ensemble: LOOEnsembleManager = state["ensemble"]
    mac: MACDriftClassifier = state["mac_drift"]

    probs = ensemble.predict_test_proba(X_cal)
    new_scores = compute_aps_scores(probs, y_cal, randomized=False)

    mac.update_calibration_stream(new_scores, y_cal)
    mac.recalibrate_adaptive_quantiles()
    state["stream_counter"] += len(y_cal)

    return {
        "message": "Online calibration updated successfully.",
        "updated_samples": len(y_cal),
        "total_stream_samples": state["stream_counter"],
        "updated_quantiles": np.round(np.asarray(mac.current_class_quantiles), 4).tolist()
    }


# --- Interactive Visual Dashboard (/ui) ---

@app.get("/ui", response_class=HTMLResponse)
def interactive_dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAC-Drift | Real-Time Malware Triage Console</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }
        .container { max-width: 960px; margin: 0 auto; }
        .header { border-bottom: 1px solid #334155; padding-bottom: 16px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { margin: 0; font-size: 22px; color: #38bdf8; }
        .badge { background: #065f46; color: #34d399; padding: 6px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }
        .card h2 { margin-top: 0; font-size: 16px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
        .control-row { display: flex; gap: 10px; align-items: center; margin-bottom: 16px; }
        input[type="number"] { background: #0f172a; border: 1px solid #475569; color: #fff; padding: 10px; border-radius: 8px; width: 130px; font-size: 15px; }
        button { background: #2563eb; color: white; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; transition: 0.2s; }
        button:hover { background: #1d4ed8; }
        button.secondary { background: #059669; }
        button.secondary:hover { background: #047857; }
        .metric-box { background: #0f172a; padding: 14px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid #38bdf8; }
        .metric-label { font-size: 12px; color: #94a3b8; }
        .metric-value { font-size: 20px; font-weight: bold; margin-top: 4px; }
        .bar-container { margin-bottom: 10px; }
        .bar-header { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
        .bar-bg { background: #0f172a; height: 10px; border-radius: 6px; overflow: hidden; }
        .bar-fill { background: #38bdf8; height: 100%; transition: width 0.3s ease; }
        .alert-safe { color: #34d399; }
        .alert-warn { color: #fbbf24; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }
        th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #334155; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1>MAC-Drift Malware Triage Console</h1>
            <div style="font-size: 13px; color: #94a3b8; margin-top: 4px;">Mondrian Adaptive Conformal Inference for Drifting PE Streams</div>
        </div>
        <span class="badge" id="statusBadge">System Online</span>
    </div>

    <div class="grid">
        <div class="card">
            <h2>1. Live PE Stream Inspector</h2>
            <div class="control-row">
                <input type="number" id="sampleIdx" value="5000" min="5000" max="24999" />
                <button onclick="analyzeSample()">Analyze Sample</button>
                <button class="secondary" onclick="randomSample()">Random</button>
            </div>

            <div class="metric-box">
                <div class="metric-label">Conformal Prediction Set (Target 90% Coverage)</div>
                <div class="metric-value" id="predSet">-</div>
            </div>

            <div class="metric-box" id="triageBox">
                <div class="metric-label">Triage Decision & Ground Truth</div>
                <div class="metric-value" id="triageStatus">-</div>
            </div>

            <button class="secondary" style="width: 100%; margin-top: 8px;" onclick="calibrateCurrent()">
                Feed Ground-Truth Label to Online Calibration (/calibrate)
            </button>
            <div id="calMsg" style="font-size: 13px; color: #34d399; margin-top: 8px;"></div>
        </div>

        <div class="card">
            <h2>2. Posterior Class Probabilities</h2>
            <div id="probBars"></div>

            <h2 style="margin-top: 24px;">3. Mondrian Adaptive Quantiles</h2>
            <table>
                <thead><tr><th>Malware Family</th><th>Adaptive Quantile (q̂)</th></tr></thead>
                <tbody id="quantileTable"></tbody>
            </table>
            <div style="font-size: 12px; color: #94a3b8; margin-top: 10px;" id="streamCount"></div>
        </div>
    </div>
</div>

<script>
    const FAMILY_NAMES = ["Class 0 (Benign/Packer)", "Class 1 (Trojan)", "Class 2 (Ransomware)", "Class 3 (Worm/Dropper)", "Class 4 (Spyware/Adware)"];
    let currentFeatures = null;
    let currentTrueLabel = null;

    async function refreshStatus() {
        const res = await fetch('/status');
        const data = await res.json();
        document.getElementById('streamCount').innerText = `Processed Stream Samples: ${data.processed_stream_samples}`;
        const tbody = document.getElementById('quantileTable');
        tbody.innerHTML = '';
        Object.entries(data.per_class_quantiles).forEach(([cls, val], idx) => {
            tbody.innerHTML += `<tr><td>${FAMILY_NAMES[idx] || cls}</td><td><strong>${val}</strong></td></tr>`;
        });
    }

    function randomSample() {
        const rnd = Math.floor(Math.random() * (24999 - 5000 + 1)) + 5000;
        document.getElementById('sampleIdx').value = rnd;
        analyzeSample();
    }

    async function analyzeSample() {
        document.getElementById('calMsg').innerText = '';
        const idx = document.getElementById('sampleIdx').value;
        const sampleRes = await fetch(`/sample/${idx}`);
        const sampleData = await sampleRes.json();
        currentFeatures = sampleData.features;
        currentTrueLabel = sampleData.true_label;

        const predRes = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: [currentFeatures] })
        });
        const pred = await predRes.json();

        const pSet = pred.prediction_sets[0];
        const isAmb = pred.is_ambiguous[0];
        const probs = pred.probabilities[0];
        const covered = pSet.includes(currentTrueLabel);

        document.getElementById('predSet').innerText = `{ ${pSet.map(c => 'Family ' + c).join(', ')} }  (Size: ${pSet.length})`;
        
        const triageEl = document.getElementById('triageStatus');
        const decisionText = isAmb ? '⚠️ AMBIGUOUS (Escalate to Analyst)' : '✅ CONFIDENT SINGLETON';
        triageEl.innerHTML = `<span class="${isAmb ? 'alert-warn' : 'alert-safe'}">${decisionText}</span><br><span style="font-size:14px; color:#cbd5e1;">True Label: Family ${currentTrueLabel} (${covered ? 'Covered ✓' : 'Missed ✗'})</span>`;

        const barsDiv = document.getElementById('probBars');
        barsDiv.innerHTML = '';
        probs.forEach((p, i) => {
            const pct = (p * 100).toFixed(2);
            barsDiv.innerHTML += `
                <div class="bar-container">
                    <div class="bar-header"><span>${FAMILY_NAMES[i]}</span><span>${pct}%</span></div>
                    <div class="bar-bg"><div class="bar-fill" style="width: ${pct}%"></div></div>
                </div>
            `;
        });
    }

    async function calibrateCurrent() {
        if (!currentFeatures) return;
        const res = await fetch('/calibrate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: [currentFeatures], true_labels: [currentTrueLabel] })
        });
        const data = await res.json();
        document.getElementById('calMsg').innerText = `✓ ${data.message} (Total samples: ${data.total_stream_samples})`;
        refreshStatus();
    }

    refreshStatus();
    analyzeSample();
</script>
</body>
</html>
"""
