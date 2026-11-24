"""
Mondrian Adaptive Conformal Prediction (MAC-Drift)
Module: Temporal PE Dataset Loader
"""

from pathlib import Path
from typing import Tuple, Dict, Optional, Union
import numpy as np
import polars as pl


class TemporalPELoader:
    """
    High-performance temporal loader for PE tabular datasets (EMBER / BODMAS).
    Ensures strict chronological sorting and consistent label encoding.
    """
    def __init__(
        self,
        filepath: Union[str, Path],
        time_col: str = "timestamp",
        label_col: str = "family_label",
        feature_cols: Optional[list] = None
    ):
        self.filepath = Path(filepath)
        self.time_col = time_col
        self.label_col = label_col
        self.feature_cols = feature_cols
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}

    def load_stream(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads data, sorts chronologically, and encodes labels.
        
        Returns:
            X: 2D numpy array of static PE features (N, D).
            y: 1D numpy array of integer labels (N,).
            timestamps: 1D numpy array of Unix/DateTime timestamps (N,).
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset file not found at: {self.filepath}")

        # Polars lazy loading for scalability
        if self.filepath.suffix == ".parquet":
            df = pl.read_parquet(self.filepath)
        elif self.filepath.suffix == ".csv":
            df = pl.read_csv(self.filepath)
        else:
            raise ValueError(f"Unsupported format: {self.filepath.suffix}. Use Parquet or CSV.")

        # Ensure chronological ordering
        df = df.sort(self.time_col, descending=False)

        # Label encoding
        unique_labels = sorted(df[self.label_col].unique().to_list())
        self.label_to_idx = {str(lbl): idx for idx, lbl in enumerate(unique_labels)}
        self.idx_to_label = {idx: str(lbl) for lbl, idx in self.label_to_idx.items()}

        df = df.with_columns(
            pl.col(self.label_col).map_elements(
                lambda x: self.label_to_idx[str(x)], 
                return_dtype=pl.Int32
            ).alias("target_encoded")
        )

        y = df["target_encoded"].to_numpy().astype(np.int64)
        timestamps = df[self.time_col].to_numpy()

        # Extract features
        if self.feature_cols is None:
            excluded = {self.time_col, self.label_col, "target_encoded"}
            active_cols = [c for c in df.columns if c not in excluded]
        else:
            active_cols = self.feature_cols

        X = df.select(active_cols).to_numpy().astype(np.float32)

        return X, y, timestamps
    