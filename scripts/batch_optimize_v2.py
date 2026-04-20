"""
Batch optimization script for running multiple queries.

This script uses pure asyncio for concurrency to avoid event loop issues
with litellm when using ThreadPoolExecutor.
"""

import argparse
import asyncio
import json
import os
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


async def run_single_query_async(
    query: str, index: int
) -> tuple[bool, str | None]:
    """
    Run interactive_optimize.py for a single query (async version).

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
        # Run the async optimization function (reuse existing event loop)
        await optimize_main(args)

        # Small delay to ensure file is written
        await asyncio.sleep(1)

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


def save_queries(
    queries: list[dict[str, Any]], json_path: str = "test_queries.json"
) -> None:
    """Save queries to JSON file."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)


async def run_batch_async(
    batch: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    pending_count: int,
    delay_within_batch: int,
    total_processed: list[int],
) -> None:
    """
    Run a batch of queries in parallel using asyncio.

    Args:
        batch: List of query dicts in this batch
        queries: Full list of queries (for updating)
        pending_count: Total number of pending queries
        delay_within_batch: Seconds to wait between starting each query
        total_processed: Mutable list to track progress (passed by reference)
    """
    # Create tasks with staggered start times
    tasks = []
    for j, q in enumerate(batch):
        task = asyncio.create_task(run_single_query_async(q["query"], q["index"]))
        tasks.append((task, q))
        print(f"  [{j+1}/{len(batch)}] Query {q['index']} started")

        # Wait before starting next task (except the last one)
        if j < len(batch) - 1 and delay_within_batch > 0:
            await asyncio.sleep(delay_within_batch)

    # Wait for all tasks to complete and collect results
    for task, q in tasks:
        idx = q["index"]
        try:
            success, log_file = await task
            # Update the query object
            for query in queries:
                if query["index"] == idx:
                    query["is_optimized"] = success
                    query["log"] = log_file if log_file else ""
                    break

            total_processed[0] += 1
            status = "OK" if success else "FAIL"
            print(
                f"[{status}] Query {idx} complete | Progress: {total_processed[0]}/{pending_count}"
            )

        except Exception as e:
            print(f"[Query {idx}] Exception: {e}")
            for query in queries:
                if query["index"] == idx:
                    query["is_optimized"] = False
                    query["log"] = ""
                    break

        # Save after each query completes (in case of interruption)
        save_queries(queries)


async def run_batch_optimize_async(
    json_path: str = "test_queries.json",
    batch_size: int = 5,
    delay_between_batches: int = 10,
    delay_within_batch: int = 1,
) -> None:
    """
    Run optimization in batches using pure asyncio.

    Args:
        json_path: Path to the JSON file containing queries
        batch_size: Number of queries to run per batch
        delay_between_batches: Seconds to wait between batches
        delay_within_batch: Seconds to wait between starting each query in a batch
    """
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Load queries
    queries = load_queries(json_path)

    # Filter out already optimized queries
    pending = [q for q in queries if not q.get("is_optimized", False)]

    if not pending:
        print("No pending queries to optimize!")
        return

    print(f"Total queries: {len(queries)}")
    print(f"Already optimized: {len(queries) - len(pending)}")
    print(f"Pending: {len(pending)}")
    print(f"Batch size: {batch_size}")
    print(f"Delay between batches: {delay_between_batches}s")
    print(f"Delay within batch: {delay_within_batch}s")

    # Create log directory if it doesn't exist
    Path("log/optimization_web").mkdir(parents=True, exist_ok=True)

    # Process in batches
    total_processed = [0]  # Use list for mutability in nested function
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(pending) + batch_size - 1) // batch_size

        print(f"\n{'#'*60}")
        print(f"# Batch {batch_num}/{total_batches} - Starting {len(batch)} queries")
        print(f"{'#'*60}\n")

        # Run batch in parallel
        await run_batch_async(batch, queries, len(pending), delay_within_batch, total_processed)

        print(f"\n{'='*60}")
        print(f"Batch {batch_num} complete! ({len(batch)} queries finished)")
        print(f"{'='*60}")

        # Wait between batches (except the last batch)
        if i + batch_size < len(pending) and delay_between_batches > 0:
            print(f"\nWaiting {delay_between_batches}s before next batch...\n")
            await asyncio.sleep(delay_between_batches)

    # Final save
    save_queries(queries)

    print(f"\n{'='*60}")
    print("Batch optimization complete!")
    print(f"{'='*60}")


def run_batch_optimize(
    json_path: str = "test_queries.json",
    batch_size: int = 5,
    delay_between_batches: int = 10,
    delay_within_batch: int = 1,
) -> None:
    """
    Run optimization in batches (sync wrapper for async implementation).

    Args:
        json_path: Path to the JSON file containing queries
        batch_size: Number of queries to run per batch
        delay_between_batches: Seconds to wait between batches
        delay_within_batch: Seconds to wait between starting each query in a batch
    """
    asyncio.run(
        run_batch_optimize_async(
            json_path=json_path,
            batch_size=batch_size,
            delay_between_batches=delay_between_batches,
            delay_within_batch=delay_within_batch,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch optimization script for running multiple queries"
    )
    parser.add_argument(
        "--json",
        "-j",
        type=str,
        default="test_queries.json",
        help="Path to the JSON file containing queries",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=5,
        help="Number of queries to run per batch",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=int,
        default=10,
        help="Seconds to wait between batches",
    )
    parser.add_argument(
        "--delay-within",
        "-w",
        type=int,
        default=1,
        dest="delay_within_batch",
        help="Seconds to wait between starting each query in a batch",
    )

    args = parser.parse_args()

    run_batch_optimize(
        json_path=args.json,
        batch_size=args.batch_size,
        delay_between_batches=args.delay,
        delay_within_batch=args.delay_within_batch,
    )
