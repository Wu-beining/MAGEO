"""
MAGEO Memory Bank

Hierarchical memory system for MAGEO:
- Step-level Memory: Records single-round editing experiences
- Creator-level Memory: Aggregates best patterns across tasks

The memory bank enables experience-driven optimization by retrieving
similar successful editing patterns for use in future optimization tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memory.schema import (
    BestEditPattern,
    CreatorMemoryRecord,
    EditOp,
    RevisionPlanStep,
    StepMemoryRecord,
    create_creator_record,
    create_step_record,
)


@dataclass
class RetrievedMemoryExample:
    """
    A retrieved memory example formatted for Planner Agent consumption.

    Represents a successful editing pattern from past optimizations that
    can serve as a few-shot example for current planning.
    """

    doc_id: str
    query: str
    engine_id: str
    plan_steps: list[dict[str, Any]]
    applied_edit_ops: list[dict[str, Any]]
    delta_metrics: dict[str, float]
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM prompt."""
        return {
            "doc_id": self.doc_id,
            "query": self.query,
            "engine_id": self.engine_id,
            "plan_steps": self.plan_steps,
            "applied_edit_ops": self.applied_edit_ops,
            "delta_metrics": self.delta_metrics,
            "summary": self.summary,
        }


class MemoryBank:
    """
    Hierarchical Memory Bank for MAGEO optimization.

    Manages both step-level (tactical) and creator-level (strategic) memories.
    Provides retrieval interface for the Planner Agent to fetch similar
    successful editing patterns.

    Storage is currently in-memory; search interface is provided for
    future vector database integration.

    Attributes:
        storage_path: Directory path for persistence (optional)
        step_records: All step-level memory records
        creator_records: All creator-level memory records
    """

    def __init__(self, storage_path: str | Path | None = None):
        """
        Initialize the Memory Bank.

        Args:
            storage_path: Optional path for persistent storage.
                          If provided, records will be saved to disk.
        """
        self._storage_path = Path(storage_path) if storage_path else None
        self._step_records: dict[str, StepMemoryRecord] = {}
        self._creator_records: dict[str, CreatorMemoryRecord] = {}

        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # ===========================
    # Step-level Memory Operations
    # ===========================

    def add_step_record(self, record: StepMemoryRecord) -> None:
        """
        Add a step-level memory record.

        Args:
            record: The step record to add
        """
        self._step_records[record.record_id] = record
        self._save_step_to_disk(record)

    def add_step_from_edit(
        self,
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
        Create and add a step record from editing operation.

        Convenience method that creates the record and adds it in one call.

        Args:
            article_id: Article being optimized
            engine_id: Target engine
            query: User query
            round_id: Round number
            from_version: Version before edit
            to_version: Version after edit
            planner_plans: Planning steps
            applied_ops: Actual edit operations
            old_metrics: Metrics before edit
            new_metrics: Metrics after edit
            doc_embedding: Document embedding (optional)
            query_embedding: Query embedding (optional)

        Returns:
            The created StepMemoryRecord
        """
        record = create_step_record(
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
            doc_embedding=doc_embedding,
            query_embedding=query_embedding,
        )
        self.add_step_record(record)
        return record

    def get_step_record(self, record_id: str) -> StepMemoryRecord | None:
        """Get a step record by ID."""
        return self._step_records.get(record_id)

    def get_step_records_by_article(self, article_id: str) -> list[StepMemoryRecord]:
        """Get all step records for a specific article."""
        return [
            r for r in self._step_records.values() if r.article_id == article_id
        ]

    def get_step_records_by_engine(self, engine_id: str) -> list[StepMemoryRecord]:
        """Get all step records for a specific engine."""
        return [r for r in self._step_records.values() if r.engine_id == engine_id]

    # ===========================
    # Creator-level Memory Operations
    # ===========================

    def add_creator_record(self, record: CreatorMemoryRecord) -> None:
        """
        Add a creator-level memory record.

        Args:
            record: The creator record to add
        """
        self._creator_records[record.record_id] = record
        self._save_creator_to_disk(record)

    def add_creator_from_trajectory(
        self,
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
        Create and add a creator record from optimization trajectory.

        Convenience method that extracts best patterns and adds the record.

        Args:
            article_id: Article being optimized
            engine_id: Target engine
            query: User query
            final_version_id: Final version number
            final_metrics: Final metrics after optimization
            version_history: List of (version_id, metrics) tuples
            step_records: All step records from this optimization
            summary: Natural language summary of the strategy

        Returns:
            The created CreatorMemoryRecord
        """
        record = create_creator_record(
            article_id=article_id,
            engine_id=engine_id,
            query=query,
            final_version_id=final_version_id,
            final_metrics=final_metrics,
            version_history=version_history,
            step_records=step_records,
            summary=summary,
        )
        self.add_creator_record(record)
        return record

    def get_creator_record(self, record_id: str) -> CreatorMemoryRecord | None:
        """Get a creator record by ID."""
        return self._creator_records.get(record_id)

    def get_creator_records_by_article(self, article_id: str) -> list[CreatorMemoryRecord]:
        """Get all creator records for a specific article."""
        return [
            r for r in self._creator_records.values() if r.article_id == article_id
        ]

    def get_best_patterns(
        self, engine_id: str | None = None, limit: int = 10
    ) -> list[BestEditPattern]:
        """
        Get the best edit patterns across all creator records.

        Args:
            engine_id: Optional engine filter
            limit: Maximum number of patterns to return

        Returns:
            List of best patterns sorted by average improvement
        """
        all_patterns: list[BestEditPattern] = []
        for record in self._creator_records.values():
            if engine_id is None or record.engine_id == engine_id:
                all_patterns.extend(record.best_edit_patterns)

        # Sort by average improvement
        all_patterns.sort(key=lambda p: p.avg_improvement, reverse=True)
        return all_patterns[:limit]

    # ===========================
    # Retrieval Interface (Placeholder)
    # ===========================

    def retrieve_for_planner(
        self,
        article_id: str,
        current_metrics: dict[str, Any],
        engine_id: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedMemoryExample]:
        """
        Retrieve similar editing examples for Planner Agent.

        This is a placeholder implementation that returns recent successful
        examples. A production version would use vector similarity search
        on doc_embedding and query_embedding.

        Args:
            article_id: Current article being optimized
            current_metrics: Current metrics for context matching
            engine_id: Optional engine filter
            top_k: Maximum number of examples to return

        Returns:
            List of retrieved memory examples formatted for LLM consumption
        """
        candidates: list[StepMemoryRecord] = []

        # Filter by engine if specified
        if engine_id:
            candidates = [r for r in self._step_records.values() if r.engine_id == engine_id]
        else:
            candidates = list(self._step_records.values())

        # Filter to successful edits (positive delta in paper-aligned metrics).
        primary_metric_patterns = {
            "WLV",
            "DPA",
            "CP",
            "SI",
            "AA",
            "FA",
            "KC",
            "AD",
            "DSV-CF",
        }
        successful: list[tuple[StepMemoryRecord, float]] = []

        for record in candidates:
            # Calculate total positive improvement
            # Match if any part of the flattened key matches a primary metric
            improvement = sum(
                delta
                for key, delta in record.delta_metrics.items()
                if isinstance(delta, (int, float)) and delta > 0 and
                any(pattern in key.split(".") for pattern in primary_metric_patterns)
            )
            if improvement > 0:
                successful.append((record, improvement))

        # Sort by total improvement and take top_k
        successful.sort(key=lambda x: x[1], reverse=True)
        top_records = [r for r, _ in successful[:top_k]]

        # Convert to RetrievedMemoryExample format
        examples = []
        for record in top_records:
            examples.append(
                RetrievedMemoryExample(
                    doc_id=record.article_id,
                    query=record.query,
                    engine_id=record.engine_id,
                    plan_steps=[self._plan_step_to_dict(p) for p in record.planner_plans],
                    applied_edit_ops=[self._edit_op_to_dict(op) for op in record.applied_ops],
                    delta_metrics=record.delta_metrics,
                    summary=f"Round {record.round_id}: {len(record.applied_ops)} edits, "
                    f"dsv_cf_delta: {record.delta_metrics.get('overall.DSV-CF', 0.0):.3f}",
                )
            )

        return examples

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

    # ===========================
    # Persistence
    # ===========================

    def _save_step_to_disk(self, record: StepMemoryRecord) -> None:
        """Save step record to disk if storage_path is configured."""
        if self._storage_path:
            step_dir = self._storage_path / "steps"
            step_dir.mkdir(parents=True, exist_ok=True)
            import json

            file_path = step_dir / f"{record.record_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

    def _save_creator_to_disk(self, record: CreatorMemoryRecord) -> None:
        """Save creator record to disk if storage_path is configured."""
        if self._storage_path:
            creator_dir = self._storage_path / "creators"
            creator_dir.mkdir(parents=True, exist_ok=True)
            import json

            file_path = creator_dir / f"{record.record_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

    def _load_from_disk(self) -> None:
        """Load existing records from disk."""
        if not self._storage_path:
            return

        import json

        # Load step records
        step_dir = self._storage_path / "steps"
        if step_dir.exists():
            for file_path in step_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Deserialize to StepMemoryRecord
                    record = self._step_record_from_dict(data)
                    self._step_records[record.record_id] = record
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        # Load creator records
        creator_dir = self._storage_path / "creators"
        if creator_dir.exists():
            for file_path in creator_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    record = self._creator_record_from_dict(data)
                    self._creator_records[record.record_id] = record
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

    @staticmethod
    def _step_record_from_dict(data: dict[str, Any]) -> StepMemoryRecord:
        """Deserialize dict to StepMemoryRecord."""
        # Reconstruct RevisionPlanStep objects
        plan_steps = [
            RevisionPlanStep(
                step_id=p.get("step_id", ""),
                target_span=p.get("target_span", ""),
                edit_type=p.get("edit_type", ""),
                target_metrics=p.get("target_metrics", []),
                risk_constraints=p.get("risk_constraints", []),
                rationale=p.get("rationale", ""),
                suggested_operations=p.get("suggested_operations", []),
            )
            for p in data.get("planner_plans", [])
        ]

        # Reconstruct EditOp objects
        applied_ops = [
            EditOp(
                edit_type=o.get("edit_type", ""),
                target_span=o.get("target_span", ""),
                op_pattern=o.get("op_pattern", ""),
            )
            for o in data.get("applied_ops", [])
        ]

        return StepMemoryRecord(
            record_id=data.get("record_id", ""),
            article_id=data.get("article_id", ""),
            engine_id=data.get("engine_id", ""),
            query=data.get("query", ""),
            round_id=data.get("round_id", 0),
            from_version=data.get("from_version", 0),
            to_version=data.get("to_version", 0),
            planner_plans=plan_steps,
            applied_ops=applied_ops,
            old_metrics=data.get("old_metrics", {}),
            new_metrics=data.get("new_metrics", {}),
            delta_metrics=data.get("delta_metrics", {}),
            doc_embedding=data.get("doc_embedding"),
            query_embedding=data.get("query_embedding"),
            created_at=data.get("created_at", ""),
        )

    @staticmethod
    def _creator_record_from_dict(data: dict[str, Any]) -> CreatorMemoryRecord:
        """Deserialize dict to CreatorMemoryRecord."""
        from memory.schema import VersionHistoryEntry, BestEditPattern

        version_history = [
            VersionHistoryEntry(
                version_id=v.get("version_id", 0),
                metrics=v.get("metrics", {}),
            )
            for v in data.get("version_history", [])
        ]

        best_patterns = [
            BestEditPattern(
                edit_type=p.get("edit_type", ""),
                op_pattern=p.get("op_pattern", ""),
                target_metrics=p.get("target_metrics", []),
                success_count=p.get("success_count", 1),
                avg_improvement=p.get("avg_improvement", 0.0),
            )
            for p in data.get("best_edit_patterns", [])
        ]

        return CreatorMemoryRecord(
            record_id=data.get("record_id", ""),
            article_id=data.get("article_id", ""),
            engine_id=data.get("engine_id", ""),
            query=data.get("query", ""),
            final_version_id=data.get("final_version_id", 0),
            final_metrics=data.get("final_metrics", {}),
            version_history=version_history,
            best_edit_patterns=best_patterns,
            summary=data.get("summary", ""),
            created_at=data.get("created_at", ""),
        )

    # ===========================
    # Utility Methods
    # ===========================

    def clear(self) -> None:
        """Clear all in-memory records."""
        self._step_records.clear()
        self._creator_records.clear()

    @property
    def step_count(self) -> int:
        """Number of step records."""
        return len(self._step_records)

    @property
    def creator_count(self) -> int:
        """Number of creator records."""
        return len(self._creator_records)

    def stats(self) -> dict[str, Any]:
        """Get statistics about the memory bank."""
        # Count by engine
        step_engines: dict[str, int] = {}
        for record in self._step_records.values():
            step_engines[record.engine_id] = step_engines.get(record.engine_id, 0) + 1

        creator_engines: dict[str, int] = {}
        for record in self._creator_records.values():
            creator_engines[record.engine_id] = creator_engines.get(record.engine_id, 0) + 1

        return {
            "total_step_records": self.step_count,
            "total_creator_records": self.creator_count,
            "step_records_by_engine": step_engines,
            "creator_records_by_engine": creator_engines,
        }
