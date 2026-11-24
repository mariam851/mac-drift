"""
Unit Test: Verification of Within-Class Coverage (Mondrian Validity) under Imbalance and Drift
"""

import numpy as np
import pytest
from src.conformal.mac_drift import MACDriftClassifier
from src.conformal.nonconformity_scores import compute_aps_scores


def test_mac_drift_within_class_coverage():
    rng = np.random.default_rng(2026)
    n_samples = 4000
    alpha = 0.10

    # Highly imbalanced: 90% Class 0 (Benign), 10% Class 1 (Rare Malware)
    labels = rng.choice([0, 1], size=n_samples, p=[0.90, 0.10])
    
    # Generate synthetic probabilities with drift:
    # Class 1 gets harder to classify over time (p drops from 0.8 to 0.4)
    probs = np.zeros((n_samples, 2), dtype=np.float64)
    time_factor = np.linspace(0.0, 1.0, n_samples)

    for i in range(n_samples):
        y = labels[i]
        if y == 0:
            p_true = rng.uniform(0.70, 0.95)
        else:
            # Concept drift on rare malware
            p_true = np.clip(rng.uniform(0.75, 0.95) - 0.35 * time_factor[i], 0.1, 0.99)
            
        probs[i, y] = p_true
        probs[i, 1 - y] = 1.0 - p_true

    scores = compute_aps_scores(probs, labels, randomized=False)

    # Initial warmup stream
    warmup_n = 2000
    mac = MACDriftClassifier(
        num_classes=2,
        alpha=alpha,
        delta=0.05,
        window_candidates=[64, 128, 256, 512, 1024]
    )
    mac.update_calibration_stream(scores[:warmup_n], labels[:warmup_n])
    mac.recalibrate_adaptive_quantiles()

    # Streaming evaluation on drifted period
    test_probs = probs[warmup_n:]
    test_labels = labels[warmup_n:]
    
    pred_sets = mac.predict_sets(test_probs)

    covered_class_0 = [0 in pred_sets[i] for i in range(len(test_labels)) if test_labels[i] == 0]
    covered_class_1 = [1 in pred_sets[i] for i in range(len(test_labels)) if test_labels[i] == 1]

    cov_0 = np.mean(covered_class_0)
    cov_1 = np.mean(covered_class_1)

    # Both classes MUST maintain valid within-class coverage >= (1 - alpha - tolerance)
    assert cov_0 >= (1.0 - alpha - 0.05)
    assert cov_1 >= (1.0 - alpha - 0.05)
    