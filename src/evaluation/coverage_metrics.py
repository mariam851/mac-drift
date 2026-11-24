"""
Mondrian Adaptive Conformal Prediction (MAC-Drift)
Module: Conformal Coverage and Conditional Validity Metrics
"""

from typing import List, Dict, Any
import numpy as np


def compute_marginal_coverage(prediction_sets: List[Any], y_true: np.ndarray) -> float:
    """
    Computes overall marginal empirical coverage: (1/N) sum 1(y_i in S(x_i))
    """
    y_arr = np.asarray(y_true).ravel()
    n_samples = len(y_arr)
    if n_samples == 0:
        return 0.0

    hits = 0
    for i in range(n_samples):
        target = int(y_arr[i])
        # تحويل عناصر المجموعة إلى أرقام صحيحة للتأكد التام من نجاح المقارنة
        pset = {int(val) for val in prediction_sets[i]}
        if target in pset:
            hits += 1

    return float(hits / n_samples)


def compute_class_conditional_coverage(
    prediction_sets: List[Any],
    y_true: np.ndarray,
    n_classes: int
) -> Dict[int, float]:
    """
    Computes empirical coverage conditioned on each class family c in {0, ..., K-1}.
    """
    y_arr = np.asarray(y_true).ravel()
    class_coverage = {}

    for c in range(n_classes):
        indices = np.where(y_arr == c)[0]
        if len(indices) == 0:
            class_coverage[c] = 1.0
            continue

        hits = 0
        for idx in indices:
            pset = {int(val) for val in prediction_sets[idx]}
            if c in pset:
                hits += 1

        class_coverage[c] = float(hits / len(indices))

    return class_coverage


def compute_coverage_violations(
    class_coverage: Dict[int, float],
    target_coverage: float = 0.90
) -> Dict[str, float]:
    """
    Calculates coverage violations:
    - max_violation: max_c (target - cov_c)^+
    - min_class_coverage: min_c cov_c
    """
    if not class_coverage:
        return {"max_violation": 0.0, "min_class_coverage": 1.0}

    coverages = list(class_coverage.values())
    violations = [max(0.0, target_coverage - cov) for cov in coverages]

    return {
        "max_violation": float(np.max(violations)),
        "min_class_coverage": float(np.min(coverages))
    }
