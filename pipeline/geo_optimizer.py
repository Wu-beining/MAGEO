"""
MAGEO Optimizer - main control loop for generative engine optimization.

This module implements the complete optimization pipeline:
1. Initial evaluation using simulated generative engine
2. Preference profiling
3. Memory retrieval for similar cases
4. Planning with PlannerAgent
5. Editing with EditorAgent (generates K candidates)
6. Evaluation with paper-aligned DSV-CF metrics
7. Candidate selection under the fidelity gate
8. Memory writeback (step-level and creator-level)
9. Early stopping based on DSV-CF plateauing
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent import (
    EditorAgent,
    EvaluationAgent,
    PlannerAgent,
    PreferenceAgent,
    QAAgent,
    QueryRewriteAgent,
)
from config.base import ModelConfig
from evaluation import (
    DEFAULT_EPSILON,
    DEFAULT_K_PATIENCE,
    DEFAULT_MAX_ROUNDS,
    SelectionResult,
    UnifiedMetrics,
    check_early_stopping,
    evaluate_in_simulated_GE,
    net_improvement,
    select_best_candidate,
)
from memory import MemoryBank, RevisionPlanStep, StepMemoryRecord
from memory.schema import EditOp


@dataclass
class OptimizationConfig:
    """Configuration for the GEO optimization process."""

    # Iteration control
    max_rounds: int = DEFAULT_MAX_ROUNDS
    k_patience: int = DEFAULT_K_PATIENCE
    epsilon: float = DEFAULT_EPSILON

    # Candidate generation
    k_candidates: int = 2  # Number of candidates per round

    # Model selection
    preference_model: str = "gpt-5-mini"
    planner_model: str = "gpt-5-mini"
    editor_model: str = "gpt-5-mini"
    evaluator_model: str = "gpt-5-mini"
    qa_model: str = "gpt-5-mini"

    # Feature flags
    use_query_rewrite: bool = False  # Whether to use query rewriting

    # Logging
    log_dir: str = "log/optimization"
    save_history: bool = True


@dataclass
class VersionHistory:
    """History entry for a single version."""

    version_id: int
    content: str
    metrics: dict[str, float]
    applied_ops: list[dict[str, Any]]
    delta_metrics: dict[str, float]


@dataclass
class OptimizationResult:
    """Final result of the optimization process."""

    # Input info
    article_id: str
    engine_id: str
    query: str

    # Final output
    best_content: str
    best_metrics: dict[str, float]
    final_version_id: int

    # History
    version_history: list[VersionHistory]
    total_rounds: int

    # Improvement summary
    initial_metrics: dict[str, float]
    total_improvement: dict[str, float]

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "article_id": self.article_id,
            "engine_id": self.engine_id,
            "query": self.query,
            "best_content": self.best_content,
            "best_metrics": self.best_metrics,
            "final_version_id": self.final_version_id,
            "version_history": [
                {
                    "version_id": v.version_id,
                    "content": v.content,
                    "metrics": v.metrics,
                    "applied_ops": v.applied_ops,
                    "delta_metrics": v.delta_metrics,
                }
                for v in self.version_history
            ],
            "total_rounds": self.total_rounds,
            "initial_metrics": self.initial_metrics,
            "total_improvement": self.total_improvement,
            "created_at": self.created_at,
        }


class GEOOptimizer:
    """
    Main optimizer for MAGEO.

    Coordinates the complete optimization pipeline:
    - Memory retrieval
    - Multi-agent planning, editing, evaluation
    - Candidate selection with safety constraints
    - Early stopping and memory writeback
    """

    def __init__(
        self,
        config: OptimizationConfig | None = None,
        model_config: ModelConfig | None = None,
        memory_bank: MemoryBank | None = None,
    ):
        """
        Initialize the GEO optimizer.

        Args:
            config: Optimization configuration
            model_config: Model configuration (loads from default if None)
            memory_bank: Optional existing memory bank
        """
        self._config = config or OptimizationConfig()
        self._model_config = model_config or ModelConfig.load()
        self._memory_bank = memory_bank or MemoryBank(storage_path=self._config.log_dir)
        self._current_step_records: list[StepMemoryRecord] = []

        # Initialize models
        from model.litellm_model import LiteLLMModel

        self._planner_model = LiteLLMModel(
            **self._model_config.get_model(self._config.planner_model)
        )
        self._preference_model = LiteLLMModel(
            **self._model_config.get_model(self._config.preference_model)
        )
        self._editor_model = LiteLLMModel(
            **self._model_config.get_model(self._config.editor_model)
        )
        self._evaluator_model = LiteLLMModel(
            **self._model_config.get_model(self._config.evaluator_model)
        )
        self._qa_model = LiteLLMModel(
            **self._model_config.get_model(self._config.qa_model)
        )
        self._rewriter_model = None
        if self._config.use_query_rewrite:
            self._rewriter_model = LiteLLMModel(
                **self._model_config.get_model(self._config.planner_model)
            )

        # Initialize agents
        self._preference = PreferenceAgent(model=self._preference_model)
        self._planner = PlannerAgent(model=self._planner_model)
        self._editor = EditorAgent(model=self._editor_model)
        self._evaluator = EvaluationAgent(model=self._evaluator_model)
        self._qa = QAAgent(model=self._qa_model)
        self._rewriter: QueryRewriteAgent | None = None
        if self._rewriter_model is not None:
            self._rewriter = QueryRewriteAgent(model=self._rewriter_model)

        # Setup logging
        Path(self._config.log_dir).mkdir(parents=True, exist_ok=True)

    async def optimize(
        self,
        article_id: str,
        engine_id: str,
        query: str,
        content: str,
        engine_rules: str,
        source_url: str = "",
        all_search_results: list[dict] | None = None,
        selected_idx: int | None = None,
    ) -> OptimizationResult:
        """
        Run the complete optimization pipeline.

        Args:
            article_id: Article identifier
            engine_id: Target engine (e.g., "gpt-4o", "perplexity")
            query: User query to optimize for
            content: Initial article content
            engine_rules: Engine preference rules (JSON string)
            source_url: Optional source URL for metadata
            all_search_results: Optional list of all search results for RAG context
            selected_idx: Optional index of the selected document in all_search_results

        Returns:
            OptimizationResult with best version and history
        """
        # Store search results context for evaluation
        self._all_search_results = all_search_results
        self._selected_idx = selected_idx
        self._current_step_records = []
        preference_profile = await self._build_preference_profile(engine_id, engine_rules)

        # Initialize state
        current_content = content
        version_id = 0
        no_improve_rounds = 0
        rounds_run = 0
        version_history: list[VersionHistory] = []
        best_overall: dict[str, Any] = {
            "content": current_content,
            "metrics": {},
            "version_id": version_id,
        }

        # Step 2: Initial evaluation
        if self._config.save_history:
            print(f"[GEO] Step 1: Initial evaluation for {article_id}...")

        initial_metrics = await self._evaluate_document(
            query, current_content, engine_id, article_id, version_id
        )
        current_metrics_vec = initial_metrics.get_primary_vector()
        best_overall["metrics"] = current_metrics_vec

        # Step 3: Retrieve from memory
        memory_examples = await self._retrieve_memory_examples(
            article_id, current_metrics_vec, engine_id
        )

        # Main optimization loop
        for round_idx in range(self._config.max_rounds):
            rounds_run = round_idx + 1
            if self._config.save_history:
                print(f"[GEO] Round {round_idx + 1}/{self._config.max_rounds}")

            # Step 4: Planning
            plan = await self._generate_plan(
                query, current_content, preference_profile, memory_examples
            )

            # Step 5: Editing
            candidates = await self._generate_candidates(
                current_content, plan, preference_profile
            )

            # Step 6: Evaluation
            eval_results = await self._evaluate_candidates(
                query, current_content, candidates, preference_profile
            )

            # Step 7: Selection
            selection = await self._select_best_version(
                eval_results, candidates, current_metrics_vec
            )

            if selection is None:
                # No safe or meaningfully improving candidates.
                no_improve_rounds += 1
                if self._config.save_history:
                    print("[GEO] No safe improving candidates, skipping...")

                stop_result = check_early_stopping(
                    current_round=round_idx,
                    no_improve_rounds=no_improve_rounds,
                    max_rounds=self._config.max_rounds,
                    k_patience=self._config.k_patience,
                )
                if stop_result.should_stop:
                    if self._config.save_history:
                        print(f"[GEO] Early stop: {stop_result.reason}")
                    break
                continue

            # Step 7b: Real evaluation of selected version
            previous_metrics_vec = dict(current_metrics_vec)
            new_metrics = await self._evaluate_document(
                query, selection.content, engine_id, article_id, version_id + 1
            )
            new_metrics_vec = new_metrics.get_primary_vector()

            # Calculate deltas
            delta_metrics = {
                k: new_metrics_vec.get(k, 0.0) - previous_metrics_vec.get(k, 0.0)
                for k in set(previous_metrics_vec.keys()) | set(new_metrics_vec.keys())
            }

            round_net_imp = net_improvement(new_metrics_vec, previous_metrics_vec)
            if round_net_imp <= 0:
                no_improve_rounds += 1
                if self._config.save_history:
                    print(
                        f"[GEO] Candidate accepted by evaluator but failed real DSV-CF re-check "
                        f"(delta={round_net_imp:.4f}), keeping previous version."
                    )

                stop_result = check_early_stopping(
                    current_round=round_idx,
                    no_improve_rounds=no_improve_rounds,
                    max_rounds=self._config.max_rounds,
                    k_patience=self._config.k_patience,
                )
                if stop_result.should_stop:
                    if self._config.save_history:
                        print(f"[GEO] Early stop: {stop_result.reason}")
                    break
                continue

            # Update version tracking
            version_id += 1
            current_content = selection.content
            current_metrics_vec = new_metrics_vec

            # Record history
            version_history.append(
                VersionHistory(
                    version_id=version_id,
                    content=current_content,
                    metrics=new_metrics_vec,
                    applied_ops=selection.applied_ops,
                    delta_metrics=delta_metrics,
                )
            )

            # Step 8: Memory writeback
            await self._write_step_memory(
                article_id,
                engine_id,
                query,
                round_idx,
                version_id - 1,
                version_id,
                plan,
                selection.applied_ops,
                previous_metrics_vec,
                new_metrics_vec,
            )

            # Step 8b: Check early stopping
            net_imp = net_improvement(new_metrics_vec, best_overall["metrics"])
            if net_imp > 0:
                # New overall best
                best_overall = {
                    "content": current_content,
                    "metrics": new_metrics_vec,
                    "version_id": version_id,
                }
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            # Check stopping conditions
            stop_result = check_early_stopping(
                current_round=round_idx,
                no_improve_rounds=no_improve_rounds,
                max_rounds=self._config.max_rounds,
                k_patience=self._config.k_patience,
            )

            if self._config.save_history:
                print(
                    f"[GEO] Round {round_idx + 1}: dsv_cf_delta={net_imp:.4f}, "
                    f"no_improve={no_improve_rounds}/{self._config.k_patience}"
                )

            if stop_result.should_stop:
                if self._config.save_history:
                    print(f"[GEO] Early stop: {stop_result.reason}")
                break

        # Step 9: Creator-level memory writeback
        await self._write_creator_memory(
            article_id,
            engine_id,
            query,
            version_id,
            best_overall["metrics"],
            version_history,
        )

        # Assemble final result
        final_metrics_vec = best_overall["metrics"]
        total_improvement = {
            k: final_metrics_vec.get(k, 0.0)
            - initial_metrics.get_primary_vector().get(k, 0.0)
            for k in set(initial_metrics.get_primary_vector().keys())
            | set(final_metrics_vec.keys())
        }

        result = OptimizationResult(
            article_id=article_id,
            engine_id=engine_id,
            query=query,
            best_content=best_overall["content"],
            best_metrics=best_overall["metrics"],
            final_version_id=best_overall["version_id"],
            version_history=version_history,
            total_rounds=rounds_run,
            initial_metrics=initial_metrics.get_primary_vector(),
            total_improvement=total_improvement,
        )

        # Save result if enabled
        if self._config.save_history:
            self._save_result(result)

        return result

    # ===========================
    # Helper methods
    # ===========================

    async def _evaluate_document(
        self,
        query: str,
        content: str,
        engine_id: str,
        article_id: str,
        version_id: int,
    ) -> UnifiedMetrics:
        """Evaluate a document using the simulated GE pipeline."""
        return await evaluate_in_simulated_GE(
            query=query,
            document=content,
            engine_id=engine_id,
            article_id=article_id,
            version_id=version_id,
            query_rewrite_agent=self._rewriter,
            qa_agent=self._qa,
            evaluation_agent=self._evaluator,
            all_search_results=self._all_search_results,
            selected_idx=self._selected_idx,
        )

    async def _build_preference_profile(self, engine_id: str, engine_rules: str) -> str:
        """Construct the paper-aligned engine preference profile once per run."""
        return await self._preference.run(engine_id=engine_id, engine_rules=engine_rules)

    async def _retrieve_memory_examples(
        self,
        article_id: str,
        metrics: dict[str, float],
        engine_id: str,
    ) -> str:
        """Retrieve memory examples and format for LLM consumption."""
        examples = self._memory_bank.retrieve_for_planner(
            article_id=article_id,
            current_metrics=metrics,
            engine_id=engine_id,
            top_k=5,
        )

        # Format as JSON for prompt
        return json.dumps([e.to_dict() for e in examples], ensure_ascii=False)

    async def _generate_plan(
        self,
        query: str,
        content: str,
        engine_rules: str,
        memory_examples: str,
    ) -> str:
        """Generate revision plan using PlannerAgent."""
        return await self._planner.run(
            query=query,
            document=content,
            engine_rules=engine_rules,
            memory_examples=memory_examples,
        )

    async def _generate_candidates(
        self,
        content: str,
        plan: str,
        engine_rules: str,
    ) -> str:
        """Generate K candidates using EditorAgent."""
        return await self._editor.run(
            document=content,
            revision_plan=plan,
            engine_rules=engine_rules,
            k=self._config.k_candidates,
        )

    async def _evaluate_candidates(
        self,
        query: str,
        baseline: str,
        candidates: str,
        engine_rules: str,
    ) -> str:
        """Evaluate candidates using EvaluationAgent."""
        return await self._evaluator.run(
            user_query=query,
            baseline_content=baseline,
            candidates=candidates,
            engine_rules=engine_rules,
        )

    async def _select_best_version(
        self,
        eval_results: str,
        editor_candidates: str,
        current_metrics: dict[str, float],
    ) -> SelectionResult | None:
        """
        Select the best candidate from evaluation results.

        Args:
            eval_results: JSON from EvaluationAgent with predicted_scores
            editor_candidates: JSON from EditorAgent with revised_content
            current_metrics: Current baseline metrics
        """
        # Parse evaluation results
        eval_data = json.loads(eval_results)
        eval_candidates_list = eval_data.get("evaluations", [])

        # Parse editor candidates to get content
        editor_data = json.loads(editor_candidates)
        editor_candidates_list = editor_data.get("candidates", [])

        # Create lookup map for candidate_id -> content/ops
        candidate_content_map = {
            cand.get("candidate_id"): cand
            for cand in editor_candidates_list
        }

        # Merge evaluation scores with editor content
        merged_candidates = []
        for eval_cand in eval_candidates_list:
            cid = eval_cand.get("candidate_id")
            if cid in candidate_content_map:
                # Merge evaluation scores with editor content
                merged = {
                    **eval_cand,
                    **candidate_content_map[cid],
                    # Keep eval's predicted_scores, take editor's content/ops
                    "predicted_scores": eval_cand.get("predicted_scores", {}),
                    "revised_content": candidate_content_map[cid].get("revised_content", ""),
                    "applied_edit_ops": candidate_content_map[cid].get("applied_edit_ops", []),
                }
                merged_candidates.append(merged)

        # Debug: print candidate safety check
        if self._config.save_history and merged_candidates:
            print(f"[DEBUG] Evaluating {len(merged_candidates)} candidates...")
            for cand in merged_candidates:
                pred_scores = cand.get("predicted_scores", {})
                fa_val = pred_scores.get("fa", pred_scores.get("FA", "N/A"))
                aa_val = pred_scores.get("aa", pred_scores.get("AA", "N/A"))
                print(f"[DEBUG] Candidate {cand.get('candidate_id')}: fa={fa_val}, aa={aa_val}")

        result = select_best_candidate(merged_candidates, current_metrics, self._config.epsilon)

        if result is None and self._config.save_history:
            print("[DEBUG] No safe candidates selected!")

        return result

    async def _write_step_memory(
        self,
        article_id: str,
        engine_id: str,
        query: str,
        round_id: int,
        from_version: int,
        to_version: int,
        plan: str,
        applied_ops: list[dict[str, Any]],
        old_metrics: dict[str, float],
        new_metrics: dict[str, float],
    ) -> None:
        """Write step-level memory record."""
        plan_steps = self._parse_plan_steps(plan)
        edit_ops = self._parse_edit_ops(applied_ops)

        record = self._memory_bank.add_step_from_edit(
            article_id=article_id,
            engine_id=engine_id,
            query=query,
            round_id=round_id,
            from_version=from_version,
            to_version=to_version,
            planner_plans=plan_steps,
            applied_ops=edit_ops,
            old_metrics=old_metrics,
            new_metrics=new_metrics,
        )
        self._current_step_records.append(record)

    async def _write_creator_memory(
        self,
        article_id: str,
        engine_id: str,
        query: str,
        final_version_id: int,
        final_metrics: dict[str, float],
        version_history: list[VersionHistory],
    ) -> None:
        """Write creator-level memory record."""
        # Convert version history
        history_tuples = [(v.version_id, v.metrics) for v in version_history]

        # Generate summary
        best_patterns = self._memory_bank.get_best_patterns(engine_id=engine_id, limit=3)
        pattern_summary = ", ".join(
            f"{pattern.edit_type}:{pattern.op_pattern}" for pattern in best_patterns
        )
        summary = (
            f"MAGEO optimized article {article_id} for {engine_id} across {final_version_id} "
            f"accepted versions. Reusable high-impact patterns: {pattern_summary or 'none'}."
        )

        # Write to memory bank
        self._memory_bank.add_creator_from_trajectory(
            article_id=article_id,
            engine_id=engine_id,
            query=query,
            final_version_id=final_version_id,
            final_metrics=final_metrics,
            version_history=history_tuples,
            step_records=self._current_step_records,
            summary=summary,
        )

    def _parse_plan_steps(self, plan: str) -> list[RevisionPlanStep]:
        try:
            plan_data = json.loads(plan)
        except (json.JSONDecodeError, TypeError):
            return []

        result: list[RevisionPlanStep] = []
        for raw_step in plan_data.get("plan_steps", []):
            if not isinstance(raw_step, dict):
                continue
            result.append(
                RevisionPlanStep(
                    step_id=str(raw_step.get("step_id", "")).strip() or f"step_{len(result) + 1}",
                    target_span=str(raw_step.get("target_span", "")).strip() or "unknown",
                    edit_type=str(raw_step.get("edit_type", "")).strip() or "Structure",
                    target_metrics=self._to_str_list(raw_step.get("target_metrics")),
                    risk_constraints=self._to_str_list(raw_step.get("risk_constraints")),
                    rationale=str(raw_step.get("rationale", "")).strip(),
                    suggested_operations=self._to_str_list(raw_step.get("suggested_operations")),
                )
            )
        return result

    def _parse_edit_ops(self, applied_ops: list[dict[str, Any]]) -> list[EditOp]:
        result: list[EditOp] = []
        for op in applied_ops:
            if not isinstance(op, dict):
                continue
            result.append(
                EditOp(
                    edit_type=str(op.get("edit_type", "")).strip() or "Structure",
                    target_span=str(op.get("target_span", "")).strip() or "unknown",
                    op_pattern=str(op.get("op_pattern", "")).strip(),
                )
            )
        return result

    @staticmethod
    def _to_str_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _save_result(self, result: OptimizationResult) -> None:
        """Save optimization result to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.article_id}_{timestamp}.json"
        filepath = Path(self._config.log_dir) / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"[GEO] Result saved to {filepath}")
