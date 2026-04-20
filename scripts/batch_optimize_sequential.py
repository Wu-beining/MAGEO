"""
Sequential batch optimization script for running multiple queries.

This script runs queries one by one, updating the JSON file with progress.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path so we can import from pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.interactive_optimize import main as optimize_main


def get_latest_log_file(
    log_dir: str = "log/optimization_web", after_ts: float | None = None
) -> str | None:
    """
    Get the most recently modified log file from the log directory.

    Args:
        log_dir: Path to the log directory
        after_ts: Only return files modified after this timestamp

    Returns:
        Filename of the latest log file, or None if directory is empty
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return None

    # Get all JSON files in the directory
    json_files = list(log_path.glob("*.json"))
    if not json_files:
        return None

    # Filter by timestamp if provided
    if after_ts is not None:
        json_files = [f for f in json_files if f.stat().st_mtime > after_ts]

    if not json_files:
        return None

    # Sort by modification time, get the latest
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    return latest.name


def run_single_query(query: str, index: int) -> tuple[bool, str | None]:
    """
    Run interactive_optimize.py for a single query.

    Args:
        query: The query string to optimize
        index: The index of the query (for display purposes)

    Returns:
        Tuple of (success: bool, log_file: str | None)
    """
    # Get current time before running
    before_ts = time.time()

    # Create mock args for the optimize function
    args = argparse.Namespace(
        query=query,
        auto=True,
        yes=True,
    )

    try:
        # Run the async optimization function
        asyncio.run(optimize_main(args))

        # Small delay to ensure file is written
        time.sleep(1)

        # Find the new log file
        log_file = get_latest_log_file(after_ts=before_ts)

        return True, log_file

    except Exception as e:
        print(f"[Query {index}] Exception: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def load_queries(json_path: str = "test_queries.json") -> list[dict[str, Any]]:
    """Load queries from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_queries(queries: list[dict[str, Any]], json_path: str = "test_queries.json") -> None:
    """Save queries to JSON file."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)


def run_batch_optimize(
    json_path: str = "test_queries.json",
    delay_between_queries: int = 0,
) -> None:
    """
    Run optimization sequentially, one query at a time.

    Args:
        json_path: Path to the JSON file containing queries
        delay_between_queries: Seconds to wait between each query (for API rate limit)
    """
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    import os

    os.chdir(project_root)

    # Load queries
    queries = load_queries(json_path)

    # Filter out already optimized queries
    pending = [q for q in queries if not q.get("is_optimized", False)]

    if not pending:
        print("No pending queries to optimize!")
        return

    print(f"{'='*60}")
    print(f"Total queries: {len(queries)}")
    print(f"Already optimized: {len(queries) - len(pending)}")
    print(f"Pending: {len(pending)}")
    if delay_between_queries > 0:
        print(f"Delay between queries: {delay_between_queries}s")
    print(f"{'='*60}\n")

    # Create log directory if it doesn't exist
    Path("log/optimization_web").mkdir(parents=True, exist_ok=True)

    # Process queries one by one
    total_processed = 0
    for i, q in enumerate(pending):
        idx = q["index"]
        query_text = q["query"]

        print(f"\n[{i+1}/{len(pending)}] Running Query {idx}")
        print(f"Query: {query_text[:80]}{'...' if len(query_text) > 80 else ''}")
        print("-" * 60)

        # Run the query
        success, log_file = run_single_query(query_text, idx)

        # Update the query object
        for query in queries:
            if query["index"] == idx:
                query["is_optimized"] = success
                query["log"] = log_file if log_file else ""
                break

        total_processed += 1
        status = "OK" if success else "FAIL"
        log_info = f"| Log: {log_file}" if log_file else ""

        print("-" * 60)
        print(f"[{status}] Query {idx} complete {log_info}")
        print(f"Progress: {total_processed}/{len(pending)}")

        # Save after each query completes (in case of interruption)
        save_queries(queries)

        # Wait between queries (except the last one)
        if i < len(pending) - 1 and delay_between_queries > 0:
            print(f"Waiting {delay_between_queries}s before next query...\n")
            time.sleep(delay_between_queries)

    # Final save
    save_queries(queries)

    print(f"\n{'='*60}")
    print("Batch optimization complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential batch optimization script for running queries one by one"
    )
    parser.add_argument(
        "--json",
        "-j",
        type=str,
        default="test_queries.json",
        help="Path to the JSON file containing queries",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=int,
        default=0,
        help="Seconds to wait between each query (default: 0)",
    )

    args = parser.parse_args()

    run_batch_optimize(
        json_path=args.json,
        delay_between_queries=args.delay,
    )
