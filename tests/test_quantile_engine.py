"""
Unit Test: Mathematical Verification of Lepski Adaptive Quantile Engine
"""

import numpy as np
import pytest
from src.drift.lepski_window_selector import LepskiWindowSelector
from src.drift.kolmogorov_smirnov import compute_ks_distance


def test_ks_identity_and_shift():
    rng = np.random.default_rng(42)
    s1 = rng.normal(0, 1, 1000)
    s2 = rng.normal(0, 1, 1000)
    s3 = rng.normal(3, 1, 1000)
    
    d_null, p_null = compute_ks_distance(s1, s2)
    d_shift, p_shift = compute_ks_distance(s1, s3)
    
    assert d_null < 0.10
    assert p_null > 0.01
    assert d_shift > 0.80
    assert p_shift < 1e-10


def test_lepski_adapts_to_abrupt_drift():
    """
    If past 3 windows have old mean 0.0, but most recent window shifts to 5.0,
    Lepski must reject contaminated longer windows and choose the shortest window k=1.
    """
    rng = np.random.default_rng(2026)
    
    # Simulate scores: W1 (new/current, shifted), W2 (old), W3 (old)
    w1 = rng.exponential(scale=5.0, size=500)
    w2 = np.concatenate([w1, rng.exponential(scale=1.0, size=500)])
    w3 = np.concatenate([w2, rng.exponential(scale=1.0, size=1000)])
    
    candidate_scores = [w1, w2, w3]
    candidate_sizes = [len(w1), len(w2), len(w3)]
    
    selector = LepskiWindowSelector(alpha=0.10, delta=0.05)
    best_idx, opt_q, diag = selector.select_best_window(candidate_scores, candidate_sizes)
    
    # Must select k=0 (the shortest, uncontaminated window)
    assert best_idx == 0
    assert diag["bias_proxy"][2] > diag["bias_proxy"][0]
    