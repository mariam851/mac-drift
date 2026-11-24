"""
Mondrian Adaptive Conformal Prediction (MAC-Drift)
Module: Nonconformity Scores and Prediction Set Construction
"""

from typing import List, Union, Optional
import numpy as np


def compute_aps_scores(
    probs: np.ndarray,
    y: np.ndarray,
    randomized: bool = False,
    randomize: Optional[bool] = None,
    seed: int = 42
) -> np.ndarray:
    """
    Computes Adaptive Prediction Set (APS) nonconformity scores for true labels:
    E_i = sum_{y' <= y} pi(x)_{y'}
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    n_samples, n_classes = probs.shape

    sorted_idx = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
    cum_probs = np.cumsum(sorted_probs, axis=1)

    scores = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        true_label = int(y[i])
        rank_matches = np.where(sorted_idx[i] == true_label)[0]
        if len(rank_matches) == 0:
            scores[i] = 1.0
        else:
            rank = rank_matches[0]
            scores[i] = float(cum_probs[i, rank])

    return scores


def compute_raps_scores_all_classes(
    probs: np.ndarray,
    lambda_reg: float = 0.01,
    k_reg: int = 1,
    randomized: bool = False,
    randomize: Optional[bool] = None,
    seed: int = 42
) -> np.ndarray:
    """
    Computes Regularized Adaptive Prediction Set (RAPS) scores for ALL candidate classes.
    For each class c, the score is its cumulative mass when ranked by probability + penalty.
    """
    probs = np.asarray(probs, dtype=np.float64)
    n_samples, n_classes = probs.shape

    sorted_idx = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
    cum_probs = np.cumsum(sorted_probs, axis=1)

    all_scores = np.zeros((n_samples, n_classes), dtype=np.float64)
    for i in range(n_samples):
        for rank in range(n_classes):
            cls = sorted_idx[i, rank]
            penalty = lambda_reg * max(0, rank + 1 - k_reg)
            all_scores[i, cls] = float(cum_probs[i, rank] + penalty)

    return all_scores


def construct_prediction_sets(
    scores_or_probs: np.ndarray,
    quantile: Union[float, np.ndarray, dict, list],
    randomized: bool = False,
    randomize: Optional[bool] = None,
    seed: int = 42
) -> List[List[int]]:
    """
    Constructs prediction sets: S(x) = {c : score(x, c) <= quantile_c}.
    Guarantees non-empty sets by keeping the argmin score (top-1 class).
    """
    scores_or_probs = np.asarray(scores_or_probs, dtype=np.float64)
    n_samples, n_classes = scores_or_probs.shape

    if isinstance(quantile, dict):
        default_q = 0.90
        q_thresh = np.array([float(quantile.get(c, default_q)) for c in range(n_classes)], dtype=np.float64)
    elif isinstance(quantile, (list, np.ndarray)):
        q_arr = np.asarray(quantile, dtype=np.float64).ravel()
        if len(q_arr) == n_classes:
            q_thresh = q_arr
        elif len(q_arr) == 1:
            q_thresh = np.full(n_classes, q_arr[0], dtype=np.float64)
        else:
            q_thresh = np.full(n_classes, q_arr[0], dtype=np.float64)
    else:
        q_thresh = np.full(n_classes, float(quantile), dtype=np.float64)

   
    prediction_sets = []

    for i in range(n_samples):
        row_scores = scores_or_probs[i]
        
        valid_classes = [c for c in range(n_classes) if row_scores[c] <= q_thresh[c]]
        
        if not valid_classes:
            valid_classes = [int(np.argmin(row_scores))]
            
        prediction_sets.append(sorted(valid_classes))

    return prediction_sets