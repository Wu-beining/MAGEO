"""
Simulated Generative Engine Evaluator for MAGEO.

This module provides the evaluate_in_simulated_GE function which simulates
how a generative engine would process a document and generate metrics.

The simulation pipeline:
1. Optionally normalize the query
2. Use QAAgent to generate a response with citations under a frozen retrieval list
3. Compute paper-aligned exposure metrics (WLV / DPA) from the response
4. Use EvaluationAgent to score the remaining DSV-CF dimensions

This replaces the need to call actual search engines during optimization.
"""

from __future__ import annotations

from typing import Any

from agent import EvaluationAgent, QAAgent, QueryRewriteAgent
from evaluation.metrics import UnifiedMetrics, compute_wlv_dpa_for_answer


async def evaluate_in_simulated_GE(
    query: str,
    document: str,
    engine_id: str,
    article_id: str = "",
    version_id: int = 0,
    query_rewrite_agent: QueryRewriteAgent | None = None,
    qa_agent: QAAgent | None = None,
    evaluation_agent: EvaluationAgent | None = None,
    all_search_results: list[dict] | None = None,
    selected_idx: int | None = None,
) -> UnifiedMetrics:
    """
    Evaluate a document in a simulated generative engine environment.

    This function simulates the paper's twin-branch setting under a frozen retrieval list.

    Args:
        query: User query to optimize for
        document: Document content to evaluate (the one being optimized)
        engine_id: Target engine ID (for logging/metadata)
        article_id: Article identifier (default: "")
        version_id: Version number (default: 0)
        query_rewrite_agent: Optional agent for query rewriting
        qa_agent: Required agent for RAG generation
        evaluation_agent: Required agent for L2/L3 evaluation
        all_search_results: Optional list of all search results for RAG context
        selected_idx: Optional index of the selected document in all_search_results

    Returns:
        UnifiedMetrics object with the paper-aligned DSV-CF metrics.

    Raises:
        ValueError: If qa_agent or evaluation_agent is not provided
    """
    if qa_agent is None:
        raise ValueError("qa_agent is required for simulated evaluation")
    if evaluation_agent is None:
        raise ValueError("evaluation_agent is required for simulated evaluation")

    # Step 1: (Optional) Rewrite query
    effective_query = query
    if query_rewrite_agent is not None:
        query_result = await query_rewrite_agent.run(query)
        # Parse JSON to extract main_query
        import json

        try:
            query_data = json.loads(query_result)
            effective_query = query_data.get("main_query", query)
        except (json.JSONDecodeError, TypeError):
            effective_query = query

    # Step 2: Generate RAG answer
    # If all_search_results is provided, use them as context (with optimized document)
    # Otherwise, use just the single document being evaluated
    if all_search_results is not None and selected_idx is not None:
        # Format all search results as RAG context
        formatted_doc = _format_all_search_results(
            all_search_results, selected_idx, document
        )
    else:
        # Legacy behavior: format just the single document
        formatted_doc = _format_document_as_search_result(document)
    answer = await qa_agent.run(effective_query, formatted_doc)

    # Step 3: Compute exposure metrics from the generated answer.
    exposure = compute_wlv_dpa_for_answer(answer)
    target_citation_id = (selected_idx + 1) if selected_idx is not None else 1
    wlv = float(exposure.get("wlv", {}).get(target_citation_id, 0.0))
    dpa = float(exposure.get("dpa", {}).get(target_citation_id, 0.0))

    # Step 4: Get L2/L3 metrics from evaluation agent
    # Since we need to evaluate a single document (not candidates),
    # we construct a minimal candidates structure
    candidates_json = _construct_single_candidate(document)

    eval_result = await evaluation_agent.run(
        user_query=effective_query,
        baseline_content="",  # Not needed for single-doc eval
        candidates=candidates_json,
        engine_rules=f"Engine: {engine_id}",
    )

    # Parse evaluation result
    eval_data = _parse_evaluation_result(eval_result)

    # Step 5: Assemble unified metrics
    metrics = UnifiedMetrics(
        article_id=article_id or "unknown",
        version_id=version_id,
        engine_id=engine_id,
        query=query,
        wlv=wlv,
        dpa=dpa,
        cp=eval_data.get("cp", 5.0),
        si=eval_data.get("si", 5.0),
        aa=eval_data.get("aa", 5.0),
        fa=eval_data.get("fa", 5.0),
        kc=eval_data.get("kc", 5.0),
        ad=eval_data.get("ad", 5.0),
    )

    return metrics


def _format_document_as_search_result(document: str) -> str:
    """
    Format a document as a search result with citation markers.

    Args:
        document: Raw document content

    Returns:
        Formatted string with [1] citation markers
    """
    # Simple formatting: wrap document with citation [1]
    # In a real system, this would have proper structure
    lines = document.strip().split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if line:
            formatted_lines.append(f"{line} [1]")

    return "\n".join(formatted_lines)


def _format_all_search_results(
    results: list[dict], selected_idx: int, optimized_document: str
) -> str:
    """
    Format all search results as RAG context for QA agent.

    The selected document is replaced with the optimized version.

    Args:
        results: List of search results with title, content, link
        selected_idx: Index of the selected document in results
        optimized_document: The optimized content to use for the selected document

    Returns:
        Formatted string with citation markers [1], [2], etc.
    """
    formatted_parts = []
    for i, r in enumerate(results):
        if i == selected_idx:
            # Use the optimized document content for the selected one
            marker = f" [{i+1}]*"
            content = optimized_document
        else:
            marker = f" [{i+1}]"
            content = r.get("content", "")
        formatted_parts.append(
            f"## 来源 {i+1}{marker}\n标题：{r.get('title', '')}\n链接：{r.get('link', '')}\n内容：{content}"
        )
    return "\n\n".join(formatted_parts)


def _construct_single_candidate(document: str) -> str:
    """
    Construct a candidates JSON string for single-document evaluation.

    Args:
        document: Document content

    Returns:
        JSON string with candidates array
    """
    import json

    candidates = {
        "candidates": [
            {
                "candidate_id": "V0",
                "description": "Current document version",
                "applied_edit_ops": [],
                "revised_content": document,
            }
        ]
    }
    return json.dumps(candidates, ensure_ascii=False)


def _parse_evaluation_result(eval_result: str) -> dict[str, float]:
    """
    Parse evaluation agent output into a flat metrics dict.

    Args:
        eval_result: JSON string from evaluation agent

    Returns:
        Dict with metric names as keys and float scores as values
    """
    import json

    try:
        data = json.loads(eval_result)
        evaluations = data.get("evaluations", [])
        if evaluations:
            # Take the first (and only) evaluation
            first_eval = evaluations[0]
            scores = first_eval.get("predicted_scores", {})
            return {k: float(v) for k, v in scores.items()}
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    # Fallback: return default scores
    return {
        "wlv": 5.0,
        "dpa": 5.0,
        "cp": 5.0,
        "si": 5.0,
        "aa": 5.0,
        "fa": 5.0,
        "kc": 5.0,
        "ad": 5.0,
    }
