"""
Batch optimization script for running multiple queries.

Reads queries from test_queries.json, runs interactive_optimize.py
in batches of 5 (parallel within batch), and updates the JSON file with log paths.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_latest_log_file(log_dir: str = "log/optimization_web") -> str | None:
    """
    Get the most recently modified log file from the log directory.

    Args:
        log_dir: Path to the log directory

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

    # Sort by modification time, get the latest
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    return latest.name


def run_single_query(query: str, index: int) -> tuple[bool, str | None]:
    """
    Run interactive_optimize.py for a single query.

    Waits for the process to complete before returning.

    Args:
        query: The query string to optimize
        index: The index of the query (for display purposes)

    Returns:
        Tuple of (success: bool, log_file: str | None)
    """
    # Get log files before running
    log_files_before = (
        set(Path("log/optimization_web").glob("*.json"))
        if Path("log/optimization_web").exists()
        else set()
    )

    # Run the optimization script
    cmd = [
        sys.executable,
        "-m",
        "pipeline.interactive_optimize",
        "--query",
        query,
        "--auto",
        "-y",
    ]

    try:
        # Run the process and wait for completion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,  # 20 minutes timeout
            cwd=PROJECT_ROOT,
        )

        if result.returncode == 0:
            # Find the new log file
            log_files_after = (
                set(Path("log/optimization_web").glob("*.json"))
                if Path("log/optimization_web").exists()
                else set()
            )
            new_files = log_files_after - log_files_before

            if new_files:
                latest_log = max(new_files, key=lambda p: p.stat().st_mtime)
                return True, latest_log.name
            else:
                return True, None
        else:
            print(f"[Query {index}] Error: {result.stderr[:200]}")
            return False, None

    except subprocess.TimeoutExpired:
        print(f"[Query {index}] Timeout")
        return False, None
    except Exception as e:
        print(f"[Query {index}] Exception: {e}")
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


def run_batch_optimize(
    json_path: str = "test_queries.json",
    batch_size: int = 5,
    delay_between_batches: int = 10,
) -> None:
    """
    Run optimization in batches.

    Queries within a batch run in parallel.
    Different batches run sequentially with delay between them.

    Args:
        json_path: Path to the JSON file containing queries
        batch_size: Number of queries to run per batch (parallel)
        delay_between_batches: Seconds to wait between batches (for API rate limit)
    """
    # Load queries
    os.chdir(PROJECT_ROOT)
    queries = load_queries(json_path)

    # Filter out already optimized queries
    pending = [q for q in queries if not q.get("is_optimized", False)]

    if not pending:
        print("No pending queries to optimize!")
        return

    print(f"Total queries: {len(queries)}")
    print(f"Already optimized: {len(queries) - len(pending)}")
    print(f"Pending: {len(pending)}")
    print(f"Batch size: {batch_size} (parallel within batch)")
    print(f"Delay between batches: {delay_between_batches}s")

    # Create log directory if it doesn't exist
    Path("log/optimization_web").mkdir(parents=True, exist_ok=True)

    # Process in batches
    total_processed = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(pending) + batch_size - 1) // batch_size

        print(f"\n{'#'*60}")
        print(f"# Batch {batch_num}/{total_batches} - Starting {len(batch)} queries in parallel")
        print(f"{'#'*60}\n")

        # Run queries in parallel within the batch
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            # Submit all tasks
            future_to_query = {
                executor.submit(run_single_query, q["query"], q["index"]): q
                for q in batch
            }

            # Collect results as they complete
            for future in as_completed(future_to_query):
                query_obj = future_to_query[future]
                idx = query_obj["index"]

                try:
                    success, log_file = future.result()
                    # Update the query object
                    for q in queries:
                        if q["index"] == idx:
                            q["is_optimized"] = success
                            q["log"] = log_file if log_file else ""
                            break

                    total_processed += 1
                    status = "✓" if success else "✗"
                    print(
                        f"[{status}] Query {idx} complete | Progress: {total_processed}/{len(pending)}"
                    )

                except Exception as e:
                    print(f"[Query {idx}] Exception: {e}")
                    for q in queries:
                        if q["index"] == idx:
                            q["is_optimized"] = False
                            q["log"] = ""
                            break

                # Save after each query completes (in case of interruption)
                save_queries(queries)

        print(f"\n{'='*60}")
        print(f"Batch {batch_num} complete! ({len(batch)} queries finished)")
        print(f"{'='*60}")

        # Wait between batches (except the last batch)
        if i + batch_size < len(pending) and delay_between_batches > 0:
            print(f"\nWaiting {delay_between_batches}s before next batch...\n")
            time.sleep(delay_between_batches)

    # Final save
    save_queries(queries)

    print(f"\n{'='*60}")
    print("Batch optimization complete!")
    print(f"{'='*60}")


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
        help="Number of queries to run in parallel within each batch",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=int,
        default=10,
        help="Seconds to wait between batches",
    )

    args = parser.parse_args()

    run_batch_optimize(
        json_path=args.json,
        batch_size=args.batch_size,
        delay_between_batches=args.delay,
    )
