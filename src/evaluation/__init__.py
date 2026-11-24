"""
Statistical coverage, safety violation, and efficiency metrics.
"""

from src.evaluation.coverage_metrics import (
    compute_marginal_coverage,
    compute_class_conditional_coverage,
    compute_coverage_violations,
)
from src.evaluation.efficiency_metrics import (
    compute_efficiency_summary,
    compute_observed_unconfidence,
)

__all__ = [
    "compute_marginal_coverage",
    "compute_class_conditional_coverage",
    "compute_coverage_violations",
    "compute_efficiency_summary",
    "compute_observed_unconfidence",
]