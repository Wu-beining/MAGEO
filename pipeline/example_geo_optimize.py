"""
Example: MAGEO Optimization Pipeline

This script demonstrates how to use the GEOOptimizer to optimize
content for generative search engines.

Usage:
    python -m pipeline.example_geo_optimize
"""

import asyncio
import os

from config.base import ModelConfig
from memory import MemoryBank
from pipeline import GEOOptimizer, OptimizationConfig


def get_sample_engine_rules() -> str:
    """
    Sample engine preference rules for demonstration.

    In production, this would be loaded from a pre-trained analysis
    of the target engine's behavior.
    """
    return """{
  "Authority_TLD_Ratio": 0.6,
  "Deep_URL_Ratio": 0.2,
  "Listicle_Title_Ratio": 0.3,
  "Question_Title_Ratio": 0.2,
  "Platform_Bias_Academic/Reference": 0.4,
  "Platform_Bias_News/Media": 0.3,
  "Platform_Bias_Other": 0.3,
  "Avg_Citation_Length": 150.0,
  "Numeric_Density": 0.05,
  "Avg_Sentence_Length": 15.0,
  "Citation_Overlap_Score": 0.5,
  "External_Lexicon_Ratio": 0.4,
  "Markdown_Usage_Score": 25.0
}"""


async def main():
    """Run a sample optimization."""

    # Configuration
    config = OptimizationConfig(
        max_rounds=3,  # Keep small for demo
        k_patience=2,
        k_candidates=2,
        preference_model="gpt-5-mini",
        planner_model="gpt-5-mini",
        editor_model="gpt-5-mini",
        evaluator_model="gpt-5-mini",
        qa_model="gpt-5-mini",
        save_history=True,
        log_dir="log/optimization_demo",
    )

    # Initialize optimizer
    optimizer = GEOOptimizer(
        config=config,
        model_config=ModelConfig.load(),
        memory_bank=MemoryBank(storage_path="log/memory_demo"),
    )

    # Sample input
    article_id = "demo_article_001"
    engine_id = "gpt-4o"
    query = "如何提高网页在搜索引擎中的排名？"

    # Original content (simple version)
    original_content = """
提高网页排名需要注意以下几点：
1. 优化关键词密度
2. 增加外链数量
3. 定期更新内容
""".strip()

    # Get engine rules
    engine_rules = get_sample_engine_rules()

    print("=" * 60)
    print("MAGEO Optimization Demo")
    print("=" * 60)
    print(f"Article ID: {article_id}")
    print(f"Engine: {engine_id}")
    print(f"Query: {query}")
    print(f"\nOriginal content:\n{original_content}")
    print("-" * 60)

    # Run optimization
    result = await optimizer.optimize(
        article_id=article_id,
        engine_id=engine_id,
        query=query,
        content=original_content,
        engine_rules=engine_rules,
    )

    # Display results
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print(f"\nTotal rounds: {result.total_rounds}")
    print(f"Final version: {result.final_version_id}")

    print("\n--- Optimized Content ---")
    print(result.best_content)

    print("\n--- Metric Improvements ---")
    for key, val in result.total_improvement.items():
        if abs(val) > 0.01:
            status = "✓" if val > 0 else "✗"
            print(f"{status} {key}: {val:+.3f}")

    print("\n--- Version History ---")
    for v in result.version_history:
        print(
            f"Version {v.version_id}: delta_metrics={sum(v.delta_metrics.values()):.3f}"
        )

    print(f"\nResult saved to: {config.log_dir}")


if __name__ == "__main__":
    # Check for required API keys

    asyncio.run(main())
