"""
Mondrian Adaptive Conformal Prediction (MAC-Drift)
Module: Adaptive Conformal Inference (ACI & AgACI) Baselines
"""

from typing import List, Optional
import numpy as np
from src.conformal.nonconformity_scores import compute_raps_scores_all_classes, construct_prediction_sets


class ACIConformalClassifier:
    """
    Adaptive Conformal Inference (ACI) for dynamic temporal adaptation:
    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    """
    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.01,
        window_size: int = 1000,
        lambda_reg: float = 0.05,
        k_reg: int = 2
    ):
        self.target_alpha = float(alpha)
        self.current_alpha = float(alpha)
        self.gamma = float(gamma)
        self.window_size = int(window_size)
        self.lambda_reg = float(lambda_reg)
        self.k_reg = int(k_reg)
        self.calibration_scores: List[float] = []

    def update_calibration(self, new_scores: np.ndarray) -> None:
        self.calibration_scores.extend(new_scores.tolist())
        if len(self.calibration_scores) > self.window_size:
            self.calibration_scores = self.calibration_scores[-self.window_size:]

    def adapt_alpha(self, batch_covered: np.ndarray) -> None:
        """
        Updates current effective alpha based on empirical batch error rate:
        err_rate = 1.0 - mean(covered)
        """
        err_rate = 1.0 - float(np.mean(batch_covered))
        self.current_alpha = self.current_alpha + self.gamma * (self.target_alpha - err_rate)
        # Projection to prevent infinite intervals or empty collapse
        self.current_alpha = float(np.clip(self.current_alpha, 0.001, 0.999))

    def predict_sets(self, test_probs: np.ndarray, seed: Optional[int] = 42) -> List[List[int]]:
        if not self.calibration_scores:
            raise RuntimeError("Calibration buffer is empty.")
            
        n_cal = len(self.calibration_scores)
        q_level = np.clip(np.ceil((n_cal + 1) * (1.0 - self.current_alpha)) / n_cal, 0.0, 1.0)
        q_hat = float(np.quantile(self.calibration_scores, q_level, method="higher"))

        all_class_scores = compute_raps_scores_all_classes(
            test_probs,
            lambda_reg=self.lambda_reg,
            k_reg=self.k_reg,
            seed=seed
        )
        return construct_prediction_sets(all_class_scores, q_hat)
    