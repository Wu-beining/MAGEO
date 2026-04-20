"""
MAGEO Memory Schema

Data models for MAGEO memory system:
- EditOp: Atomic edit operation
- RevisionPlanStep: Planning step from Planner Agent
- StepMemoryRecord: Single-round editing experience
- CreatorMemoryRecord: Cross-task aggregated best patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class EditOp:
    """
    Atomic edit operation applied to a document.

    Attributes:
        edit_type: Type of edit (Structure/Evidence/Safety/Style)
        target_span: Target location (paragraph/sentence index)
        op_pattern: Tagged description of the operation pattern
    """

    edit_type: str  # "Structure" / "Evidence" / "Safety" / "Style"
    target_span: str  # e.g., "section_2_paragraph_3", "intro_para_1"
    op_pattern: str  # e.g., "add_stats_official_source", "split_long_paragraph"


@dataclass(frozen=True)
class RevisionPlanStep:
    """
    A single planning step from Planner Agent.

    Attributes:
        step_id: Unique identifier for this step
        target_span: Target location for editing
        edit_type: Type of edit to perform
        target_metrics: Metrics to improve/constrain
        risk_constraints: Safety constraints to respect
        rationale: Why this edit is needed
        suggested_operations: Specific operation suggestions
    """

    step_id: str
    target_span: str
    edit_type: str
    target_metrics: list[str] = field(default_factory=list)
    risk_constraints: list[str] = field(default_factory=list)
    rationale: str = ""
    suggested_operations: list[str] = field(default_factory=list)


@dataclass
class StepMemoryRecord:
    """
    Step-level Memory: Records single-round editing experience.

    Stores the atomic-level experience from one optimization round,
    including what was planned, what was actually executed, and the
    resulting metric changes.

    Attributes:
        record_id: Unique identifier for this record
        article_id: Article being optimized
        engine_id: Target generative engine
        query: User query being optimized for
        round_id: Round number in the optimization process
        from_version: Version number before edit
        to_version: Version number after edit
        planner_plans: Planning steps from Planner Agent
        applied_ops: Actual edit operations applied
        old_metrics: Metrics before edit
        new_metrics: Metrics after edit
        delta_metrics: Metric changes (new - old)
        doc_embedding: Document embedding for similarity search (placeholder)
        query_embedding: Query embedding for similarity search (placeholder)
        created_at: Timestamp when record was created
    """

    record_id: str
    article_id: str
    engine_id: str
    query: str
    round_id: int
    from_version: int
    to_version: int
    planner_plans: list[RevisionPlanStep]
    applied_ops: list[EditOp]
    old_metrics: dict[str, Any]
    new_metrics: dict[str, Any]
    delta_metrics: dict[str, float]
    doc_embedding: list[float] | None = None
    query_embedding: list[float] | None = None
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "article_id": self.article_id,
            "engine_id": self.engine_id,
            "query": self.query,
            "round_id": self.round_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "planner_plans": [self._plan_step_to_dict(p) for p in self.planner_plans],
            "applied_ops": [self._edit_op_to_dict(op) for op in self.applied_ops],
            "old_metrics": self.old_metrics,
            "new_metrics": self.new_metrics,
            "delta_metrics": self.delta_metrics,
            "doc_embedding": self.doc_embedding,
            "query_embedding": self.query_embedding,
            "created_at": self.created_at,
        }

    @staticmethod
    def _plan_step_to_dict(step: RevisionPlanStep) -> dict[str, Any]:
        return {
            "step_id": step.step_id,
            "target_span": step.target_span,
            "edit_type": step.edit_type,
            "target_metrics": step.target_metrics,
            "risk_constraints": step.risk_constraints,
            "rationale": step.rationale,
            "suggested_operations": step.suggested_operations,
        }

    @staticmethod
    def _edit_op_to_dict(op: EditOp) -> dict[str, Any]:
        return {
            "edit_type": op.edit_type,
            "target_span": op.target_span,
            "op_pattern": op.op_pattern,
        }


@dataclass
class BestEditPattern:
    """
    Best edit pattern extracted from successful optimizations.

    Represents a high-impact edit pattern that consistently improved
    metrics across multiple rounds or tasks.

    Attributes:
        edit_type: Type of edit that was successful
        op_pattern: Pattern tag
        target_metrics: Metrics that this pattern improves
        success_count: How many times this pattern succeeded
        avg_improvement: Average metric improvement
    """

    edit_type: str
    op_pattern: str
    target_metrics: list[str]
    success_count: int = 1
    avg_improvement: float = 0.0


@dataclass
class VersionHistoryEntry:
    """Single version entry in creator-level memory."""

    version_id: int
    metrics: dict[str, Any]


@dataclass
class CreatorMemoryRecord:
    """
    Creator-level Memory: Cross-task aggregated experience.

    Stores the complete optimization trajectory for an article and
    extracts best edit patterns for reuse in future tasks.

    Attributes:
        record_id: Unique identifier
        article_id: Article being optimized
        engine_id: Target engine
        query: User query
        final_version_id: Final version number after optimization
        final_metrics: Final metrics after optimization
        version_history: All versions and their metrics
        best_edit_patterns: Extracted high-impact patterns
        summary: Natural language summary of the optimization strategy
        created_at: Timestamp
    """

    record_id: str
    article_id: str
    engine_id: str
    query: str
    final_version_id: int
    final_metrics: dict[str, Any]
    version_history: list[VersionHistoryEntry]
    best_edit_patterns: list[BestEditPattern]
    summary: str
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "article_id": self.article_id,
            "engine_id": self.engine_id,
            "query": self.query,
            "final_version_id": self.final_version_id,
            "final_metrics": self.final_metrics,
            "version_history": [
                {"version_id": v.version_id, "metrics": v.metrics}
                for v in self.version_history
            ],
            "best_edit_patterns": [
                {
                    "edit_type": p.edit_type,
                    "op_pattern": p.op_pattern,
                    "target_metrics": p.target_metrics,
                    "success_count": p.success_count,
                    "avg_improvement": p.avg_improvement,
                }
                for p in self.best_edit_patterns
            ],
            "summary": self.summary,
            "created_at": self.created_at,
        }


def _flatten_metrics(metrics: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """
    Flatten nested metrics dictionary for delta calculation.

    Example: {"lexical": {"WC": 0.1}} → {"lexical.WC": 0.1}
    """
    items: list[tuple[str, Any]] = []
    for key, value in metrics.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(_flatten_metrics(value, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def create_step_record(
    article_id: str,
    engine_id: str,
    query: str,
    round_id: int,
    from_version: int,
    to_version: int,
    planner_plans: list[RevisionPlanStep],
    applied_ops: list[EditOp],
    old_metrics: dict[str, Any],
    new_metrics: dict[str, Any],
    doc_embedding: list[float] | None = None,
    query_embedding: list[float] | None = None,
) -> StepMemoryRecord:
    """
    Factory function to create a StepMemoryRecord with auto-generated ID.

    Calculates delta_metrics as new_metrics - old_metrics for numeric values.
    Handles nested metric structures by flattening them with dot notation.
    """
    delta_metrics: dict[str, float] = {}

    # Flatten both metrics dictionaries
    old_flat = _flatten_metrics(old_metrics)
    new_flat = _flatten_metrics(new_metrics)

    # Calculate deltas for all numeric values
    for key, new_val in new_flat.items():
        old_val = old_flat.get(key)
        if isinstance(new_val, (int, float)) and isinstance(old_val, (int, float)):
            delta_metrics[key] = float(new_val) - float(old_val)

    return StepMemoryRecord(
        record_id=str(uuid4()),
        article_id=article_id,
        engine_id=engine_id,
        query=query,
        round_id=round_id,
        from_version=from_version,
        to_version=to_version,
        planner_plans=planner_plans,
        applied_ops=applied_ops,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        delta_metrics=delta_metrics,
        doc_embedding=doc_embedding,
        query_embedding=query_embedding,
    )


def create_creator_record(
    article_id: str,
    engine_id: str,
    query: str,
    final_version_id: int,
    final_metrics: dict[str, Any],
    version_history: list[tuple[int, dict[str, Any]]],
    step_records: list[StepMemoryRecord],
    summary: str,
) -> CreatorMemoryRecord:
    """
    Factory function to create a CreatorMemoryRecord from step records.

    Extracts best_edit_patterns from the step records by analyzing
    which edit operations led to the largest metric improvements.
    """
    # Extract best patterns from step records
    pattern_stats: dict[tuple[str, str], list[float]] = {}

    for record in step_records:
        for op in record.applied_ops:
            key = (op.edit_type, op.op_pattern)
            if key not in pattern_stats:
                pattern_stats[key] = []

            # Calculate total improvement from delta_metrics
            improvement = sum(
                delta
                for delta in record.delta_metrics.values()
                if isinstance(delta, (int, float)) and delta > 0
            )
            pattern_stats[key].append(improvement)

    # Convert to BestEditPattern list
    best_patterns: list[BestEditPattern] = []
    for (edit_type, op_pattern), improvements in pattern_stats.items():
        positive_keys = [
            key
            for key, value in record.delta_metrics.items()
            if isinstance(value, (int, float)) and value > 0
        ]
        if positive_keys:
            key_scores: dict[str, float] = {}
            for key in positive_keys:
                key_scores[key] = key_scores.get(key, 0.0) + float(record.delta_metrics[key])
            target_metrics = [
                key for key, _ in sorted(key_scores.items(), key=lambda item: item[1], reverse=True)[:3]
            ]
        else:
            target_metrics = ["overall.DSV-CF"]
        avg_improvement = sum(improvements) / len(improvements)

        best_patterns.append(
            BestEditPattern(
                edit_type=edit_type,
                op_pattern=op_pattern,
                target_metrics=target_metrics,
                success_count=len(improvements),
                avg_improvement=avg_improvement,
            )
        )

    # Sort by average improvement
    best_patterns.sort(key=lambda p: p.avg_improvement, reverse=True)

    return CreatorMemoryRecord(
        record_id=str(uuid4()),
        article_id=article_id,
        engine_id=engine_id,
        query=query,
        final_version_id=final_version_id,
        final_metrics=final_metrics,
        version_history=[
            VersionHistoryEntry(version_id=vid, metrics=metrics)
            for vid, metrics in version_history
        ],
        best_edit_patterns=best_patterns[:10],  # Top 10 patterns
        summary=summary,
    )
