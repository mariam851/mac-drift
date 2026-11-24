"""
Mondrian Adaptive Conformal Prediction (MAC-Drift)
Module: Adaptive Window Selection via Lepski's Principle
"""

from typing import List, Tuple, Dict
import numpy as np


class LepskiWindowSelector:
    """
    Implements the Goldenshluger-Lepski method for data-driven adaptive
    look-back window selection under non-stationary distribution drift.
    """
    def __init__(self, alpha: float = 0.10, delta: float = 0.05, c_penalty: float = 0.65):
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.c_penalty = float(c_penalty)

    def compute_quantile(self, scores: np.ndarray) -> float:
        """Computes standard finite-sample conformal quantile."""
        n = len(scores)
        if n == 0:
            return 1.0
        q_level = np.clip(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 0.0, 1.0)
        return float(np.quantile(scores, q_level, method="higher"))

    def select_best_window(
        self,
        candidate_scores: List[np.ndarray],
        candidate_sizes: List[int]
    ) -> Tuple[int, float, Dict]:
        """
        Selects optimal window index k* balancing empirical drift bias and estimation variance.
        candidate_scores are ordered from smallest (most recent) to largest (longest history).
        """
        m = len(candidate_scores)
        if m == 1:
            q = self.compute_quantile(candidate_scores[0])
            return 0, q, {"selected_index": 0, "bias_proxy": [0.0], "variance_penalty": [0.0]}

        quantiles = np.array([self.compute_quantile(arr) for arr in candidate_scores], dtype=np.float64)
        sizes = np.array(candidate_sizes, dtype=np.float64)

        # Variance threshold: standard DKW/Bernstein type confidence width
        # psi(n) = c * sqrt(log(1/delta) / (2 * n))
        variance_penalty = self.c_penalty * np.sqrt(np.log(1.0 / self.delta) / (2.0 * sizes))

        # Goldenshluger-Lepski pairwise comparison:
        # phi_hat(k) = max_{j <= k} [ |q_k - q_j| - psi(n_j) ]_+
        bias_proxy = np.zeros(m, dtype=np.float64)
        for k in range(m):
            deviations = []
            for j in range(k + 1):
                diff = np.abs(quantiles[k] - quantiles[j]) - variance_penalty[j]
                deviations.append(max(0.0, diff))
            bias_proxy[k] = max(deviations) if deviations else 0.0

        # Total estimated risk: Bias + Variance penalty
        total_risk = bias_proxy + variance_penalty
        best_idx = int(np.argmin(total_risk))

        diagnostics = {
            "selected_index": best_idx,
            "quantiles": quantiles.tolist(),
            "bias_proxy": bias_proxy.tolist(),
            "variance_penalty": variance_penalty.tolist(),
            "total_risk": total_risk.tolist()
        }

        return best_idx, quantiles[best_idx], diagnostics
    