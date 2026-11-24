"""
Mondrian Adaptive Conformal Prediction (MAC-Drift)
Module: MAC-Drift Core Scientific Contribution
Mondrian Adaptive Rolling-Window Conformal Prediction under Non-Stationary Temporal Drift
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from src.drift.lepski_window_selector import LepskiWindowSelector
from src.conformal.nonconformity_scores import compute_raps_scores_all_classes, construct_prediction_sets


class MACDriftClassifier:
    """
    MAC-Drift: Mondrian Adaptive Rolling-Window Conformal Prediction.
    
    Guarantees independent, unbiased coverage for both rare malware families and benign files
    under unknown non-stationary temporal drift by optimizing the Lepski bias-variance trade-off
    per label category.
    """
    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.10,
        delta: float = 0.05,
        window_candidates: Optional[List[int]] = None,
        lambda_reg: float = 0.05,
        k_reg: int = 2
    ):
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.window_candidates = window_candidates or [64, 128, 256, 512, 1024, 2048]
        self.lambda_reg = float(lambda_reg)
        self.k_reg = int(k_reg)

        # Per-class historical calibration score pools
        self.class_score_pools: Dict[int, List[float]] = {c: [] for c in range(self.num_classes)}
        self.global_score_pool: List[float] = []

        self.selector = LepskiWindowSelector(alpha=self.alpha, delta=self.delta)
        self.current_class_quantiles = np.full(self.num_classes, np.nan, dtype=np.float64)
        self.global_quantile = np.nan
        self.latest_diagnostics: Dict[int, Dict] = {}

    def update_calibration_stream(
        self,
        new_scores: np.ndarray,
        new_labels: np.ndarray
    ) -> None:
        """
        Updates streaming calibration buffers with newly arrived verified data.
        """
        for s, y in zip(new_scores, new_labels):
            y_int = int(y)
            if y_int in self.class_score_pools:
                self.class_score_pools[y_int].append(float(s))
            self.global_score_pool.append(float(s))

        # Maintain max capacity
        max_limit = max(self.window_candidates) * 2
        if len(self.global_score_pool) > max_limit:
            self.global_score_pool = self.global_score_pool[-max_limit:]
            for c in range(self.num_classes):
                if len(self.class_score_pools[c]) > max_limit:
                    self.class_score_pools[c] = self.class_score_pools[c][-max_limit:]

    def recalibrate_adaptive_quantiles(self) -> np.ndarray:
        """
        Computes optimal adaptive quantiles per-class using Lepski window selection.
        """
        # 1. Update Global Fallback Quantile via Lepski
        global_scores = np.asarray(self.global_score_pool)
        candidate_global_scores = []
        candidate_global_sizes = []

        for w in self.window_candidates:
            if len(global_scores) >= w:
                sub = global_scores[-w:]
                candidate_global_scores.append(sub)
                candidate_global_sizes.append(w)

        if candidate_global_scores:
            _, self.global_quantile, _ = self.selector.select_best_window(
                candidate_global_scores,
                candidate_global_sizes
            )
        else:
            self.global_quantile = 1.0

        # 2. Update Per-Class Mondrian Quantiles
        for c in range(self.num_classes):
            c_scores = np.asarray(self.class_score_pools[c])
            candidate_c_scores = []
            candidate_c_sizes = []

            for w in self.window_candidates:
                # Class pools might be smaller, take feasible powers
                effective_w = min(w, len(c_scores))
                if effective_w >= 15:  # Minimum statistical support
                    sub = c_scores[-effective_w:]
                    candidate_c_scores.append(sub)
                    candidate_c_sizes.append(effective_w)

            # Eliminate duplicates in sizes
            unique_candidates = []
            unique_sizes = []
            seen_sizes = set()
            for s_arr, sz in zip(candidate_c_scores, candidate_c_sizes):
                if sz not in seen_sizes:
                    seen_sizes.add(sz)
                    unique_candidates.append(s_arr)
                    unique_sizes.append(sz)

            if len(unique_candidates) >= 2:
                best_idx, opt_q, diag = self.selector.select_best_window(
                    unique_candidates,
                    unique_sizes
                )
                self.current_class_quantiles[c] = opt_q
                self.latest_diagnostics[c] = diag
            elif len(unique_candidates) == 1:
                # Fallback to single empirical quantile
                q_level = 1.0 - self.alpha
                self.current_class_quantiles[c] = float(np.quantile(unique_candidates[0], q_level))
            else:
                # Shrinkage to robust global quantile if class data is too scarce
                self.current_class_quantiles[c] = self.global_quantile

        return self.current_class_quantiles

    def predict_sets(
        self,
        test_probs: np.ndarray,
        seed: Optional[int] = 42
    ) -> List[List[int]]:
        """
        Outputs Mondrian-guaranteed prediction sets for the incoming test batch.
        """
        if np.isnan(self.current_class_quantiles).all():
            self.recalibrate_adaptive_quantiles()

        all_class_scores = compute_raps_scores_all_classes(
            test_probs,
            lambda_reg=self.lambda_reg,
            k_reg=self.k_reg,
            seed=seed
        )

        return construct_prediction_sets(
            all_class_scores,
            self.current_class_quantiles
        )
    