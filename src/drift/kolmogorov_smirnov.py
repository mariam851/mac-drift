"""
Mondrian Adaptive Conformal Prediction (MAC-Drift)
Module: Kolmogorov-Smirnov & Drift Distance Engine
"""

from typing import Union, Tuple, List, Dict
import numpy as np
from scipy.stats import ks_2samp


def compute_ks_distance(
    sample_ref: np.ndarray, 
    sample_curr: np.ndarray
) -> Tuple[float, float]:
    """
    Computes two-sample Kolmogorov-Smirnov statistic between reference and current windows.
    
    Parameters:
        sample_ref: 1D array of scores/features from reference period.
        sample_curr: 1D array of scores/features from current period.
        
    Returns:
        stat: KS test statistic (supremum difference between empirical CDFs).
        p_val: Two-tailed asymptotic p-value.
    """
    sample_ref = np.asarray(sample_ref).ravel()
    sample_curr = np.asarray(sample_curr).ravel()
    
    if len(sample_ref) == 0 or len(sample_curr) == 0:
        return 0.0, 1.0
        
    res = ks_2samp(sample_ref, sample_curr)
    return float(res.statistic), float(res.pvalue)


def empirical_total_variation_discrete(
    p_counts: np.ndarray, 
    q_counts: np.ndarray
) -> float:
    """
    Computes Total Variation distance between two categorical distributions (e.g. label proportions).
    TV(P, Q) = 0.5 * sum(|p_i - q_i|)
    """
    p_prob = p_counts / np.maximum(np.sum(p_counts), 1e-12)
    q_prob = q_counts / np.maximum(np.sum(q_counts), 1e-12)
    return float(0.5 * np.sum(np.abs(p_prob - q_prob)))


def compute_window_drift_matrix(
    window_scores: List[np.ndarray]
) -> np.ndarray:
    """
    Computes pairwise KS-drift distances across all candidate look-back periods.
    Output: symmetric matrix D where D[i, j] is KS(window_i, window_j).
    """
    n_windows = len(window_scores)
    drift_matrix = np.zeros((n_windows, n_windows), dtype=np.float64)
    
    for i in range(n_windows):
        for j in range(i + 1, n_windows):
            d, _ = compute_ks_distance(window_scores[i], window_scores[j])
            drift_matrix[i, j] = d
            drift_matrix[j, i] = d
            
    return drift_matrix
