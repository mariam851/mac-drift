from typing import Optional, Dict, Any
import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
class BoostedPEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "objective": "multiclass",
            "boosting_type": "gbdt",
            "n_estimators": 150,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "force_col_wise": True
        }
        self.params = params if params is not None else default_params
        self.model = lgb.LGBMClassifier(**self.params)
        self.classes_: Optional[np.ndarray] = None
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BoostedPEClassifier":
        X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        y_arr = np.asarray(y, dtype=np.int64).ravel()
        self.classes_ = np.unique(y_arr)
        self.model.fit(X_arr, y_arr)
        return self
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        probs = self.model.predict_proba(X_arr)
        eps = 1e-7
        probs = np.clip(probs, eps, 1.0 - eps)
        return probs / np.sum(probs, axis=1, keepdims=True)
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        return self.model.predict(X_arr)
