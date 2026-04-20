"""
Candidate selection for the paper-aligned MAGEO loop.

The paper optimizes the DSV-CF objective under a fidelity-aware gate.
This module therefore:
1. Rejects candidates that break attribution / faithfulness constraints.
2. Selects the candidate with the highest predicted DSV-CF gain.
3. Uses score plateauing for early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.metrics import UnifiedMetrics, compute_dsv_cf_score


DEFAULT_MAX_ROUNDS = 10
DEFAULT_K_PATIENCE = 2
DEFAULT_EPSILON = 0.01
DEFAULT_FIDELITY_THRESHOLD = 6.0

PRIMARY_METRICS = ["WLV", "DPA", "CP", "SI", "AA", "FA", "KC", "AD", "DSV-CF"]
SAFETY_METRICS = ["AA", "FA"]


@dataclass
class SelectionResult:
    candidate_id: str
    content: str
    applied_ops: list[dict[str, Any]]
    metrics: dict[str, float]
    objective_score: float
    improvement: float
    is_safe: bool


@dataclass
class EarlyStopResult:
    should_stop: bool
    reason: str


def _to_metric_dict(metrics: dict[str, float] | UnifiedMetrics) -> dict[str, float]:
    if isinstance(metrics, UnifiedMetrics):
        return metrics.get_primary_vector()
    return metrics


def _metric_value(metrics: dict[str, float], name: str) -> float:
    for key in (
        name,
        name.lower(),
        name.upper(),
        f"ssv.{name}",
        f"ssv.{name.upper()}",
        f"isi.{name}",
        f"isi.{name.upper()}",
        f"overall.{name}",
    ):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def objective_score(metrics: dict[str, float] | UnifiedMetrics) -> float:
    metric_dict = _to_metric_dict(metrics)
    explicit_score = _metric_value(metric_dict, "DSV-CF")
    if explicit_score:
        return explicit_score
    return compute_dsv_cf_score(metric_dict)


def is_safe_enough(
    new_metrics: dict[str, float] | UnifiedMetrics,
    old_metrics: dict[str, float] | UnifiedMetrics,
    epsilon: float = DEFAULT_EPSILON,
    fidelity_threshold: float = DEFAULT_FIDELITY_THRESHOLD,
) -> bool:
    new_vec = _to_metric_dict(new_metrics)
    old_vec = _to_metric_dict(old_metrics)

    new_fa = _metric_value(new_vec, "FA")
    old_fa = _metric_value(old_vec, "FA")
    new_aa = _metric_value(new_vec, "AA")
    old_aa = _metric_value(old_vec, "AA")

    if new_fa < fidelity_threshold:
        return False
    if new_fa + epsilon < old_fa:
        return False
    if new_aa + epsilon < old_aa:
        return False
    return True


def net_improvement(
    new_metrics: dict[str, float] | UnifiedMetrics,
    old_metrics: dict[str, float] | UnifiedMetrics,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    delta = objective_score(new_metrics) - objective_score(old_metrics)
    if abs(delta) <= epsilon:
        return 0.0
    return delta


def select_best_candidate(
    candidates: list[dict[str, Any]],
    current_metrics: dict[str, float] | UnifiedMetrics,
    epsilon: float = DEFAULT_EPSILON,
    fidelity_threshold: float = DEFAULT_FIDELITY_THRESHOLD,
) -> SelectionResult | None:
    if not candidates:
        return None

    current_score = objective_score(current_metrics)
    best_candidate: dict[str, Any] | None = None
    best_score = float("-inf")
    best_improvement = float("-inf")

    for candidate in candidates:
        predicted = candidate.get("predicted_scores") or candidate.get("predicted_metrics", {})
        if not is_safe_enough(
            predicted,
            current_metrics,
            epsilon=epsilon,
            fidelity_threshold=fidelity_threshold,
        ):
            continue

        score = objective_score(predicted)
        improvement = score - current_score
        if improvement <= epsilon:
            continue

        if score > best_score:
            best_candidate = candidate
            best_score = score
            best_improvement = improvement

    if best_candidate is None:
        return None

    return SelectionResult(
        candidate_id=best_candidate.get("candidate_id", "unknown"),
        content=best_candidate.get("revised_content", ""),
        applied_ops=best_candidate.get("applied_edit_ops", []),
        metrics=best_candidate.get("predicted_scores") or best_candidate.get("predicted_metrics", {}),
        objective_score=best_score,
        improvement=best_improvement,
        is_safe=True,
    )


def check_early_stopping(
    current_round: int,
    no_improve_rounds: int,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    k_patience: int = DEFAULT_K_PATIENCE,
    safety_degraded: bool = False,
) -> EarlyStopResult:
    if current_round + 1 >= max_rounds:
        return EarlyStopResult(True, "max_rounds")
    if safety_degraded:
        return EarlyStopResult(True, "fidelity_gate")
    if no_improve_rounds >= k_patience:
        return EarlyStopResult(True, "score_plateau")
    return EarlyStopResult(False, "")
