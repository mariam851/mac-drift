"""
Mondrian Adaptive Conformal Prediction (MAC-Drift)
Module: Temporal Sliding Splitter
"""

from typing import Generator, Tuple, List, Dict
import numpy as np


class TemporalStreamSplitter:
    """
    Generates rolling temporal evaluation splits for streaming evaluation.
    """
    def __init__(
        self,
        timestamps: np.ndarray,
        warmup_size: int,
        step_batch_size: int
    ):
        self.timestamps = np.asarray(timestamps)
        self.n_samples = len(self.timestamps)
        self.warmup_size = int(warmup_size)
        self.step_batch_size = int(step_batch_size)

        if self.warmup_size >= self.n_samples:
            raise ValueError("Warmup size must be smaller than total samples.")

    def generate_splits(self) -> Generator[Tuple[np.ndarray, np.ndarray, Dict], None, None]:
        """
        Yields:
            historical_indices: Indices of all data up to current evaluation point.
            test_batch_indices: Indices of the incoming test stream batch.
            metadata: Period dictionary including batch index and time boundaries.
        """
        current_idx = self.warmup_size
        batch_idx = 0

        while current_idx < self.n_samples:
            end_idx = min(current_idx + self.step_batch_size, self.n_samples)
            
            history_indices = np.arange(0, current_idx)
            test_indices = np.arange(current_idx, end_idx)

            meta = {
                "step": batch_idx,
                "history_start": self.timestamps[0],
                "current_time": self.timestamps[current_idx],
                "batch_size": len(test_indices)
            }

            yield history_indices, test_indices, meta

            current_idx = end_idx
            batch_idx += 1
            