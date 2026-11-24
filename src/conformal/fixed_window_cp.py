"""
Mondrian Adaptive Conformal Prediction (MAC-Drift)
Module: Fixed-Window Conformal Baseline
"""

from typing import List, Optional
import numpy as np
from src.conformal.nonconformity_scores import compute_raps_scores_all_classes, construct_prediction_sets


class FixedWindowConformalClassifier:
    """
    Standard Rolling-Window Split Conformal Prediction with a pre-fixed look-back window size W.
    """
    def __init__(
        self,
        window_size: int = 1000,
        alpha: float = 0.10,
        lambda_reg: float = 0.05,
        k_reg: int = 2
    ):
        self.window_size = int(window_size)
        self.alpha = float(alpha)
        self.lambda_reg = float(lambda_reg)
        self.k_reg = int(k_reg)
        self.calibration_scores: List[float] = []

    def update_calibration(self, new_scores: np.ndarray) -> None:
        """Appends new scores and trims to fixed capacity W."""
        self.calibration_scores.extend(new_scores.tolist())
        if len(self.calibration_scores) > self.window_size:
            self.calibration_scores = self.calibration_scores[-self.window_size:]

    def predict_sets(self, test_probs: np.ndarray, seed: Optional[int] = 42) -> List[List[int]]:
        if not self.calibration_scores:
            raise RuntimeError("Calibration buffer is empty.")
            
        n_cal = len(self.calibration_scores)
        # Standard finite-sample correction: ceil((n + 1) * (1 - alpha)) / n
        q_level = np.clip(np.ceil((n_cal + 1) * (1.0 - self.alpha)) / n_cal, 0.0, 1.0)
        q_hat = float(np.quantile(self.calibration_scores, q_level, method="higher"))

        all_class_scores = compute_raps_scores_all_classes(
            test_probs,
            lambda_reg=self.lambda_reg,
            k_reg=self.k_reg,
            seed=seed
        )
        return construct_prediction_sets(all_class_scores, q_hat)
    