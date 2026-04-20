"""
Interactive MAGEO Optimization Script

This script provides an interactive command-line interface for:
1. Taking a user query input
2. Rewriting the query for better search
3. Performing web search to get relevant results
4. Selecting a document to optimize
5. Running the GEO optimization pipeline

Usage:
    # Interactive mode (default)
    python -m pipeline.interactive_optimize

    # Auto mode with query and auto-select first result
    python -m pipeline.interactive_optimize --query "your query here" --auto
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import QueryRewriteAgent, QAAgent
from config.base import ModelConfig
from evaluation import evaluate_in_simulated_GE
from memory import MemoryBank
from model.litellm_model import LiteLLMModel
from pipeline import GEOOptimizer, OptimizationConfig
from tool.web_search import web_search


def print_separator(char="=", length=60):
    """Print a separator line."""
    print(char * length)


def print_section(title: str):
    """Print a section header."""
    print_separator()
    print(f"  {title}")
    print_separator()


def format_metrics(metrics_dict: dict[str, float]) -> str:
    """Format metrics dict for display."""
    lines = []
    for key, val in sorted(metrics_dict.items()):
        lines.append(f"  {key}: {val:.2f}")
    return "\n".join(lines)


async def step1_rewrite_query(user_query: str, model_config: ModelConfig) -> str:
    """
    Step 1: Rewrite the user query for better search results.
    """
    print_section("Step 1: Query Rewriting")

    rewriter = QueryRewriteAgent(
        model=LiteLLMModel(**model_config.get_model("gpt-5-mini")),
    )

    result = await rewriter.run(user_query)
    data = json.loads(result)

    main_query = data.get("main_query", user_query)
    alternatives = data.get("alternative_queries", [])

    print(f"Original query: {user_query}")
    print(f"Rewritten query: {main_query}")

    if alternatives:
        print(f"\nAlternative queries:")
        for i, alt in enumerate(alternatives, 1):
            print(f"  {i}. {alt}")

    # Use the rewritten query
    return main_query


async def step2_web_search(search_query: str) -> list[dict]:
    """
    Step 2: Perform web search and return results.
    """
    print_section("Step 2: Web Search")
    print(f"Searching for: {search_query}")
    print("Please wait...\n")

    results = web_search(search_query)

    print(f"Found {len(results)} results:")
    for i, r in enumerate(results):
        print(f"\n[{i}] {r['title']}")
        print(f"    Link: {r['link']}")
        content_preview = (
            r["content"][:100] + "..." if len(r["content"]) > 100 else r["content"]
        )
        print(f"    Content: {content_preview}")

    return results


def step3_select_document(results: list[dict]) -> tuple[dict, int]:
    """
    Step 3: Let user select a document to optimize.

    Returns:
        Tuple of (selected_document, selected_index)
    """
    print_section("Step 3: Select Document to Optimize")

    while True:
        try:
            choice = input(f"\nEnter result number (0-{len(results)-1}): ").strip()
            idx = int(choice)
            if 0 <= idx < len(results):
                selected = results[idx]
                print(f"\nSelected: {selected['title']}")
                return selected, idx
            print(
                f"Invalid choice. Please enter a number between 0 and {len(results)-1}"
            )
        except (ValueError, KeyboardInterrupt):
            print("\nInvalid input. Please enter a valid number.")


def format_all_search_results(
    results: list[dict], selected_idx: int | None = None
) -> str:
    """
    Format all search results as RAG context for QA agent.

    Args:
        results: List of search results with title, content, link
        selected_idx: Optional index of selected result (will be marked)

    Returns:
        Formatted string with citation markers [1], [2], etc.
    """
    formatted_parts = []
    for i, r in enumerate(results):
        marker = (
            f" [{i+1}]"
            if selected_idx is None
            else (f" [{i+1}]*" if i == selected_idx else f" [{i+1}]")
        )
        formatted_parts.append(
            f"## 来源 {i+1}{marker}\n标题：{r['title']}\n链接：{r['link']}\n内容：{r['content']}"
        )
    return "\n\n".join(formatted_parts)


async def step4_generate_answer_and_evaluate(
    query: str,
    all_results: list[dict],
    selected_doc: dict,
    selected_idx: int,
    model_config: ModelConfig,
) -> tuple[str, str, dict]:
    """
    Step 4: Generate initial answer using ALL search results and calculate metrics.

    The selected document is marked with * to indicate it will be optimized.

    Returns:
        Tuple of (selected_content, answer_content, metrics_dict)
    """
    print_section("Step 4: Generate Answer & Calculate Metrics")

    # Combine title and content as the document for the selected one
    selected_content = f"# {selected_doc['title']}\n\n{selected_doc['content']}"

    print(f"Selected Document: {selected_doc['title']}")
    print(f"Total search results: {len(all_results)}")
    print(f"Using ALL results for answer generation (marked * will be optimized)")
    print("\nGenerating answer and calculating metrics...\n")

    # Format all search results as RAG context
    all_context = format_all_search_results(all_results, selected_idx)

    # Initialize agents for evaluation
    from agent import EvaluationAgent

    qa_agent = QAAgent(
        model=LiteLLMModel(**model_config.get_model("gpt-5.1")),
    )

    eval_agent = EvaluationAgent(
        model=LiteLLMModel(**model_config.get_model("gpt-5-mini")),
    )

    rewriter_agent = QueryRewriteAgent(
        model=LiteLLMModel(**model_config.get_model("gpt-5-mini")),
    )

    # Generate answer using ALL search results
    answer = await qa_agent.run(query, all_context)

    print("\n--- Initial Answer (using all search results) ---")
    print(answer[:800] + "..." if len(answer) > 800 else answer)

    # Evaluate metrics for the selected document only (this is what we're optimizing)
    metrics = await evaluate_in_simulated_GE(
        query=query,
        document=selected_content,
        engine_id="gpt-5.1",
        article_id=selected_doc["link"],
        version_id=0,
        query_rewrite_agent=rewriter_agent,
        qa_agent=qa_agent,
        evaluation_agent=eval_agent,
        all_search_results=all_results,
        selected_idx=selected_idx,
    )

    print("\n--- Initial Metrics (for selected document) ---")
    primary_vector = metrics.get_primary_vector()
    print(format_metrics(primary_vector))

    return selected_content, answer, primary_vector


async def step5_optimize(
    query: str,
    content: str,
    document: dict,
    all_results: list[dict],
    selected_idx: int,
    model_config: ModelConfig,
) -> None:
    """
    Step 5: Run the GEO optimization pipeline on selected document,
    then regenerate answer using ALL results (optimized selected + others).
    """
    print_section("Step 5: GEO Optimization")

    # Generate article_id from URL
    article_id = f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = OptimizationConfig(
        max_rounds=10,
        k_patience=2,
        k_candidates=2,
        preference_model="gpt-5-mini",
        planner_model="gpt-5-mini",
        editor_model="gpt-5-mini",
        evaluator_model="gpt-5-mini",
        qa_model="gpt-5.1",
        save_history=True,
        log_dir="log/optimization_web",
    )

    # Initialize optimizer
    optimizer = GEOOptimizer(
        config=config,
        model_config=model_config,
        memory_bank=MemoryBank(storage_path="log/memory_web"),
    )

    # Simple engine rules
    engine_rules = """{
  "Authority_TLD_Ratio": 0,
  "Deep_URL_Ratio": 0,
  "Listicle_Title_Ratio": 0,
  "Question_Title_Ratio": 0,
  "Platform_Bias_Academic/Reference": 0,
  "Platform_Bias_News/Media": 0,
  "Platform_Bias_Other": 0,
  "Avg_Citation_Length": 0,
  "Numeric_Density": 0,
  "Avg_Sentence_Length": 0,
  "Citation_Overlap_Score": 0,
  "External_Lexicon_Ratio": 0,
  "Markdown_Usage_Score": 0
}"""

    print(f"Article ID: {article_id}")
    print(f"Source: {document['link']}")
    print(f"Query: {query}")
    print(f"\nStarting optimization...\n")

    # Run optimization
    result = await optimizer.optimize(
        article_id=article_id,
        engine_id="gpt-5.1",
        query=query,
        content=content,
        engine_rules=engine_rules,
        all_search_results=all_results,
        selected_idx=selected_idx,
    )

    # Display results
    print_section("Optimization Complete!")
    print(f"\nTotal rounds: {result.total_rounds}")
    print(f"Final version: {result.final_version_id}")

    print("\n--- Optimized Content ---")
    print(result.best_content)

    print("\n--- Metric Improvements ---")
    for key, val in result.total_improvement.items():
        if abs(val) > 0.01:
            status = "+" if val > 0 else ""
            symbol = "✓" if val > 0 else "✗"
            print(f"{symbol} {key}: {status}{val:.3f}")

    print("\n--- Version History ---")
    for v in result.version_history:
        delta_sum = sum(v.delta_metrics.values())
        print(
            f"Version {v.version_id}: delta_metrics={delta_sum:.3f}, ops={len(v.applied_ops)}"
        )

    print(f"\nResult saved to: {config.log_dir}")

    # Also save the web source info
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    source_info_path = log_dir / f"{article_id}_source.json"
    with open(source_info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "article_id": article_id,
                "original_link": document["link"],
                "original_title": document["title"],
                "query": query,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Source info saved to: {source_info_path}")

    # Regenerate answer using ALL results with optimized selected document
    print_section("Final Answer Generation")
    print(
        "\nGenerating answer using ALL search results (with optimized selected document)...\n"
    )

    # Create modified results with optimized content
    modified_results = all_results.copy()
    # Update the selected document with optimized content
    modified_results[selected_idx] = {
        **modified_results[selected_idx],
        "content": result.best_content,
        "title": f"[OPTIMIZED] {document['title']}",
    }

    # Format all results with the optimized one
    final_context = format_all_search_results(modified_results, selected_idx)

    # Generate final answer
    qa_agent = QAAgent(
        model=LiteLLMModel(**model_config.get_model("gpt-5.1")),
    )
    final_answer = await qa_agent.run(query, final_context)

    print("\n--- Final Answer (using optimized content + all other results) ---")
    print(final_answer)

    # Save final answer
    final_answer_path = log_dir / f"{article_id}_final_answer.txt"
    with open(final_answer_path, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\n")
        f.write(f"Optimized Document: {document['title']}\n")
        f.write(f"Original Link: {document['link']}\n\n")
        f.write("=" * 60 + "\n")
        f.write("FINAL ANSWER\n")
        f.write("=" * 60 + "\n\n")
        f.write(final_answer)
    print(f"\nFinal answer saved to: {final_answer_path}")


async def main(args: argparse.Namespace):
    """Main interactive flow."""

    print_section("MAGEO Interactive Optimization")

    # Get user query
    if args.query:
        user_query = args.query
        print(f"Query: {user_query}")
    else:
        user_query = input("Enter your query: ").strip()

    if not user_query:
        print("Query cannot be empty!")
        return

    # Load model config
    model_config = ModelConfig.load()

    # Step 1: Rewrite query
    search_query = await step1_rewrite_query(user_query, model_config)

    # Step 2: Web search
    results = await step2_web_search(search_query)

    if not results:
        print("No search results found. Please try a different query.")
        return

    # Step 3: Select document
    if args.auto:
        selected_doc = results[0]
        selected_idx = 0
        print(f"\nAuto-selected: {selected_doc['title']}")
    else:
        selected_doc, selected_idx = step3_select_document(results)

    # Step 4: Generate answer and evaluate (using ALL search results)
    selected_content, initial_answer, metrics = (
        await step4_generate_answer_and_evaluate(
            search_query, results, selected_doc, selected_idx, model_config
        )
    )

    # Ask if user wants to proceed with optimization
    if args.auto or args.yes:
        print_section("Auto-running optimization...")
        # Step 5: Run optimization
        await step5_optimize(
            search_query,
            selected_content,
            selected_doc,
            results,
            selected_idx,
            model_config,
        )
    else:
        print_section("Continue to Optimization?")
        choice = input("Run GEO optimization on this document? (y/n): ").strip().lower()

        if choice == "y" or choice == "yes":
            # Step 5: Run optimization
            await step5_optimize(
                search_query,
                selected_content,
                selected_doc,
                results,
                selected_idx,
                model_config,
            )
        else:
            print("Optimization skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MAGEO Interactive Optimization Script"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Search query (if not provided, will prompt interactively)",
    )
    parser.add_argument(
        "--auto",
        "-a",
        action="store_true",
        help="Auto mode: automatically select first result and run optimization",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation and run optimization",
    )

    args = parser.parse_args()

    # Check for required API keys after parsing so `--help` still works.
    if not os.getenv("WEB_SEARCH_API_KEY"):
        print("Error: WEB_SEARCH_API_KEY environment variable not set")
        exit(1)

    asyncio.run(main(args))
