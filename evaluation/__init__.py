# Evaluation module for MAGEO

from evaluation.candidate_selector import (
    DEFAULT_EPSILON,
    DEFAULT_FIDELITY_THRESHOLD,
    DEFAULT_K_PATIENCE,
    DEFAULT_MAX_ROUNDS,
    PRIMARY_METRICS,
    SAFETY_METRICS,
    EarlyStopResult,
    SelectionResult,
    check_early_stopping,
    is_safe_enough,
    net_improvement,
    select_best_candidate,
)
from evaluation.metrics import (
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA,
    UnifiedMetrics,
    compute_dsv_cf_score,
    compute_delta_metrics,
    compute_wc_pwc_for_answer,
    compute_wc_pwc_for_record,
    compute_wlv_dpa_for_answer,
    extract_sentences,
    tokenize_len,
)
from evaluation.simulated_evaluator import evaluate_in_simulated_GE

__all__ = [
    # Metrics
    "UnifiedMetrics",
    "compute_delta_metrics",
    "compute_wc_pwc_for_answer",
    "compute_wc_pwc_for_record",
    "extract_sentences",
    "tokenize_len",
    # Simulated evaluation
    "evaluate_in_simulated_GE",
    # Candidate selection
    "SelectionResult",
    "EarlyStopResult",
    "select_best_candidate",
    "check_early_stopping",
    "is_safe_enough",
    "net_improvement",
    # Constants
    "PRIMARY_METRICS",
    "SAFETY_METRICS",
    "DEFAULT_MAX_ROUNDS",
    "DEFAULT_K_PATIENCE",
    "DEFAULT_EPSILON",
    "DEFAULT_FIDELITY_THRESHOLD",
    "DEFAULT_LAMBDA",
    "DEFAULT_GAMMA",
    "compute_dsv_cf_score",
    "compute_wlv_dpa_for_answer",
]
