"""
Mondrian Adaptive Conformal Prediction (MAC-Drift)
Module: Efficiency & Informativeness Metrics
"""

from typing import List, Dict
import numpy as np


def compute_efficiency_summary(
    prediction_sets: List[List[int]],
    num_classes: int
) -> Dict[str, float]:
    """
    Calculates efficiency metrics: average size, empty set rate, and singleton rate.
    """
    n_samples = len(prediction_sets)
    if n_samples == 0:
        return {
            "avg_size": 0.0,
            "median_size": 0.0,
            "empty_rate": 0.0,
            "singleton_rate": 0.0
        }

    sizes = np.array([len(s) for s in prediction_sets], dtype=np.float64)

    return {
        "avg_size": float(np.mean(sizes)),
        "median_size": float(np.median(sizes)),
        "empty_rate": float(np.mean(sizes == 0)),
        "singleton_rate": float(np.mean(sizes == 1)),
        "all_classes_rate": float(np.mean(sizes == num_classes))
    }


def compute_observed_unconfidence(
    all_class_scores: np.ndarray,
    true_labels: np.ndarray
) -> float:
    """
    OU-criterion: Mean non-conformity score of incorrect classes.
    Lower means higher confidence in the true classification.
    """
    n_samples, n_classes = all_class_scores.shape
    true_labels = np.asarray(true_labels).ravel()
    
    incorrect_scores = []
    for i in range(n_samples):
        y = true_labels[i]
        for c in range(n_classes):
            if c != y:
                incorrect_scores.append(all_class_scores[i, c])

    return float(np.mean(incorrect_scores)) if incorrect_scores else 0.0
