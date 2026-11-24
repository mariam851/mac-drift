"""
Mondrian Adaptive Conformal Prediction (MAC-Drift)
Module: Leave-One-Out (LOO) Ensemble Manager
"""

from typing import List, Tuple
import numpy as np
from src.models.base_classifier import BoostedPEClassifier


class LOOEnsembleManager:
    """
    Trains B bootstrap models and calculates out-of-bag (LOO) probabilistic predictions.
    Avoids split-conformal data wastage in streaming settings.
    """
    def __init__(
        self,
        n_estimators: int = 20,
        subsample_ratio: float = 0.8,
        seed: int = 42
    ):
        self.n_estimators = int(n_estimators)
        self.subsample_ratio = float(subsample_ratio)
        self.seed = int(seed)
        self.models: List[BoostedPEClassifier] = []
        self.bootstrap_masks: List[np.ndarray] = []
        self.classes_: np.ndarray = np.array([])

    def _align_probabilities(
        self, 
        probs: np.ndarray, 
        model_classes: np.ndarray, 
        all_classes: np.ndarray
    ) -> np.ndarray:
        """
        Maps probability outputs to the global class dimension in case
        rare malware families were omitted during random bootstrap subsampling.
        """
        if len(model_classes) == len(all_classes):
            return probs

        n_samples = probs.shape[0]
        aligned_probs = np.zeros((n_samples, len(all_classes)), dtype=np.float64)
        for idx, cls in enumerate(model_classes):
            global_idx = np.where(all_classes == cls)[0][0]
            aligned_probs[:, global_idx] = probs[:, idx]
        return aligned_probs

    def fit_and_get_loo_probs(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, List[BoostedPEClassifier]]:
        """
        Trains B bootstrap models and computes LOO predictions for the training set.
        """
        # Ensure clean, contiguous float32 numpy arrays without pandas metadata
        X_arr = np.ascontiguousarray(np.asarray(X_train, dtype=np.float32))
        y_arr = np.asarray(y_train)

        n_samples = len(X_arr)
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        rng = np.random.default_rng(self.seed)

        self.models = []
        self.bootstrap_masks = []
        
        prob_sums = np.zeros((n_samples, n_classes), dtype=np.float64)
        counts = np.zeros((n_samples, 1), dtype=np.int32)

        for b in range(self.n_estimators):
            # Bootstrap subsample with replacement
            subsample_size = max(1, int(n_samples * self.subsample_ratio))
            indices = rng.choice(n_samples, size=subsample_size, replace=True)
            
            mask = np.zeros(n_samples, dtype=bool)
            mask[indices] = True
            
            clf = BoostedPEClassifier()
            clf.fit(X_arr[indices], y_arr[indices])
            self.models.append(clf)
            self.bootstrap_masks.append(mask)

            # Evaluate on out-of-bag samples (~37% of data)
            oob_indices = np.where(~mask)[0]
            if len(oob_indices) > 0:
                raw_oob_probs = clf.predict_proba(X_arr[oob_indices])
                # Handle cases where LightGBM/classifier missed an entire rare class in the subsample
                model_classes = getattr(clf.model, "classes_", self.classes_)
                aligned_probs = self._align_probabilities(raw_oob_probs, model_classes, self.classes_)
                
                prob_sums[oob_indices] += aligned_probs
                counts[oob_indices] += 1

        # Fallback for samples never omitted from any bootstrap bag
        unseen_mask = (counts.ravel() == 0)
        if np.any(unseen_mask):
            fallback_runs = []
            for m in self.models:
                m_classes = getattr(m.model, "classes_", self.classes_)
                p = self._align_probabilities(m.predict_proba(X_arr[unseen_mask]), m_classes, self.classes_)
                fallback_runs.append(p)
                
            prob_sums[unseen_mask] = np.mean(fallback_runs, axis=0)
            counts[unseen_mask] = 1

        loo_probs = prob_sums / counts
        # Normalize rows to ensure valid probability simplex sum(p) == 1
        row_sums = loo_probs.sum(axis=1, keepdims=True)
        loo_probs = np.divide(loo_probs, row_sums, out=loo_probs, where=row_sums != 0)

        return loo_probs, self.models

    def predict_test_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Aggregates predictions from all bootstrap estimators for incoming test points.
        """
        if not self.models:
            raise RuntimeError("Ensemble models are not trained yet.")
            
        X_test_arr = np.ascontiguousarray(np.asarray(X_test, dtype=np.float32))
        
        all_probs = []
        for clf in self.models:
            raw_probs = clf.predict_proba(X_test_arr)
            model_classes = getattr(clf.model, "classes_", self.classes_)
            all_probs.append(self._align_probabilities(raw_probs, model_classes, self.classes_))
            
        mean_probs = np.mean(all_probs, axis=0)
        row_sums = mean_probs.sum(axis=1, keepdims=True)
        return np.divide(mean_probs, row_sums, out=mean_probs, where=row_sums != 0)
    