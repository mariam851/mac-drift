"""
Mondrian Adaptive Conformal Prediction (MAC-Drift)
Module: ERAPS (Ensemble Regularized Adaptive Prediction Sets) Baseline
"""

from typing import List, Optional
import numpy as np
from src.conformal.nonconformity_scores import compute_raps_scores_all_classes, construct_prediction_sets


class ERAPSClassifier:
    """
    Ensemble Regularized Adaptive Prediction Set for time series classification.
    Maintains a FIFO sliding buffer of out-of-bag ensemble non-conformity scores.
    """
    def __init__(
        self,
        buffer_size: int = 2000,
        alpha: float = 0.10,
        lambda_reg: float = 0.05,
        k_reg: int = 2
    ):
        self.buffer_size = int(buffer_size)
        self.alpha = float(alpha)
        self.lambda_reg = float(lambda_reg)
        self.k_reg = int(k_reg)
        self.score_buffer = np.empty(0, dtype=np.float64)

    def slide_buffer(self, new_oob_scores: np.ndarray) -> None:
        """Slides the score buffer forward FIFO."""
        self.score_buffer = np.concatenate([self.score_buffer, new_oob_scores])
        if len(self.score_buffer) > self.buffer_size:
            self.score_buffer = self.score_buffer[-self.buffer_size:]

    def predict_sets(self, test_probs: np.ndarray, seed: Optional[int] = 42) -> List[List[int]]:
        if len(self.score_buffer) == 0:
            raise RuntimeError("Score buffer is empty.")
            
        n_cal = len(self.score_buffer)
        q_level = np.clip(np.ceil((n_cal + 1) * (1.0 - self.alpha)) / n_cal, 0.0, 1.0)
        q_hat = float(np.quantile(self.score_buffer, q_level, method="higher"))

        all_class_scores = compute_raps_scores_all_classes(
            test_probs,
            lambda_reg=self.lambda_reg,
            k_reg=self.k_reg,
            seed=seed
        )
        return construct_prediction_sets(all_class_scores, q_hat)
    