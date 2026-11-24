"""
Statistical distribution drift detection and Lepski window adaptation algorithms.
"""

from src.drift.kolmogorov_smirnov import compute_ks_distance
from src.drift.lepski_window_selector import LepskiWindowSelector

__all__ = [
    "compute_ks_distance",
    "LepskiWindowSelector",
]
