"""
Convert Excel test set to JSON format.

Input: 测试集.xlsx (5 columns, 5th column is 'query')
Output: test_queries.json with fields: index, query, is_optimized, log
"""

import json
from pathlib import Path

import pandas as pd


def convert_excel_to_json(
    excel_path: str = "测试集.xlsx",
    output_path: str = "test_queries.json",
) -> None:
    """
    Convert Excel file to JSON format.

    Args:
        excel_path: Path to input Excel file
        output_path: Path to output JSON file
    """
    # Read Excel file
    df = pd.read_excel(excel_path)

    # Extract query column (5th column, index 4)
    queries = df.iloc[:, 4].tolist()  # Using iloc for position-based indexing

    # Build result list
    result = []
    for idx, query in enumerate(queries):
        # Skip if query is NaN or empty
        if pd.isna(query) or str(query).strip() == "":
            continue

        result.append(
            {
                "index": idx,
                "query": str(query).strip(),
                "is_optimized": False,
                "log": "",
            }
        )

    # Write to JSON file
    output_file = Path(output_path)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(result)} queries to {output_file}")


if __name__ == "__main__":
    convert_excel_to_json()
