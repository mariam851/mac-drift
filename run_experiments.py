import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data.dataset_loader import TemporalPELoader
from src.data.temporal_splitter import TemporalStreamSplitter
from src.models.ensemble_manager import LOOEnsembleManager
from src.conformal.nonconformity_scores import compute_aps_scores
from src.conformal.fixed_window_cp import FixedWindowConformalClassifier
from src.conformal.aci_baseline import ACIConformalClassifier
from src.conformal.eraps_baseline import ERAPSClassifier
from src.conformal.mac_drift import MACDriftClassifier
from src.evaluation.coverage_metrics import (
    compute_marginal_coverage,
    compute_class_conditional_coverage,
    compute_coverage_violations
)
from src.evaluation.efficiency_metrics import compute_efficiency_summary
def run_benchmark(
    dataset_path: str,
    output_dir: str = "results/",
    warmup_samples: int = 5000,
    batch_size: int = 500,
    alpha: float = 0.10,
    seed: int = 42
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"[*] Loading streaming dataset from: {dataset_path}")
    loader = TemporalPELoader(filepath=dataset_path)
    X_raw, y_raw, timestamps = loader.load_stream()
    X = np.ascontiguousarray(np.asarray(X_raw, dtype=np.float32))
    y = np.asarray(y_raw, dtype=np.int64).ravel()
    timestamps = np.asarray(timestamps)
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    print(f"[+] Loaded {len(X)} samples across {n_classes} classes: {unique_classes.tolist()}.")
    print(f"[*] Initializing warmup models on first {warmup_samples} samples...")
    ensemble_mgr = LOOEnsembleManager(n_estimators=20, subsample_ratio=0.8, seed=seed)
    loo_probs, _ = ensemble_mgr.fit_and_get_loo_probs(X[:warmup_samples], y[:warmup_samples])
    warmup_scores = compute_aps_scores(loo_probs, y[:warmup_samples], randomized=False)
    methods: Dict[str, Any] = {
        "Fixed-Window": FixedWindowConformalClassifier(window_size=1500, alpha=alpha),
        "ACI": ACIConformalClassifier(alpha=alpha, gamma=0.01, window_size=1500),
        "ERAPS": ERAPSClassifier(buffer_size=1500, alpha=alpha),
        "MAC-Drift (Ours)": MACDriftClassifier(num_classes=n_classes, alpha=alpha, delta=0.05)
    }
    methods["Fixed-Window"].update_calibration(warmup_scores)
    methods["ACI"].update_calibration(warmup_scores)
    methods["ERAPS"].slide_buffer(warmup_scores)
    methods["MAC-Drift (Ours)"].update_calibration_stream(warmup_scores, y[:warmup_samples])
    methods["MAC-Drift (Ours)"].recalibrate_adaptive_quantiles()
    splitter = TemporalStreamSplitter(
        timestamps=timestamps,
        warmup_size=warmup_samples,
        step_batch_size=batch_size
    )
    records = []
    print("[*] Starting sequential evaluation stream...")
    for history_idx, test_idx, meta in tqdm(splitter.generate_splits()):
        X_test_batch = X[test_idx]
        y_test_batch = y[test_idx]
        test_probs = ensemble_mgr.predict_test_proba(X_test_batch)
        for name, model in methods.items():
            pred_sets = model.predict_sets(test_probs)
            pred_sets = [[int(c) for c in s] for s in pred_sets]
            m_cov = compute_marginal_coverage(pred_sets, y_test_batch)
            c_cov = compute_class_conditional_coverage(pred_sets, y_test_batch, n_classes)
            viol = compute_coverage_violations(c_cov, target_coverage=(1.0 - alpha))
            eff = compute_efficiency_summary(pred_sets, n_classes)
            record = {
                "step": meta["step"],
                "method": name,
                "marginal_coverage": m_cov,
                "max_class_violation": viol["max_violation"],
                "min_class_coverage": viol["min_class_coverage"],
                "avg_set_size": eff["avg_size"],
                "empty_rate": eff["empty_rate"],
                "singleton_rate": eff["singleton_rate"]
            }
            records.append(record)
            test_scores = compute_aps_scores(test_probs, y_test_batch, randomized=False)
            if name == "Fixed-Window":
                model.update_calibration(test_scores)
            elif name == "ACI":
                covered = np.array([int(y_test_batch[i]) in pred_sets[i] for i in range(len(y_test_batch))])
                model.adapt_alpha(covered)
                model.update_calibration(test_scores)
            elif name == "ERAPS":
                model.slide_buffer(test_scores)
            elif name == "MAC-Drift (Ours)":
                model.update_calibration_stream(test_scores, y_test_batch)
                model.recalibrate_adaptive_quantiles()
    df_results = pd.DataFrame(records)
    csv_file = out_path / "streaming_benchmark_results.csv"
    df_results.to_csv(csv_file, index=False)
    print(f"[✓] Benchmark completed. Results saved to: {csv_file}")
    summary = df_results.groupby("method").agg({
        "marginal_coverage": "mean",
        "max_class_violation": "mean",
        "min_class_coverage": "mean",
        "avg_set_size": "mean",
        "empty_rate": "mean"
    }).reset_index()
    print("\n--- Summary Performance Table ---")
    print(summary.to_string(index=False))
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/ember_stream.parquet")
    parser.add_argument("--out", type=str, default="results/")
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=500)
    args = parser.parse_args()
    run_benchmark(
        dataset_path=args.data,
        output_dir=args.out,
        warmup_samples=args.warmup,
        batch_size=args.batch
    )
