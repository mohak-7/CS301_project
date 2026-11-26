"""Check the processed datasets against the paper's Table 2 and summarise their contents.

Implements the Phase 3 checks (implementation_plan.md, §3.2 checks, §3.4, Phase 3 deliverables).
Writes results/tables/data_stats.csv and exits with status 1 if any statistic differs.
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.common import PROJECT_ROOT, load_config, read_json, read_jsonl, setup_logging

logger = logging.getLogger(__name__)


def dataset_statistics(dataset_dir: Path) -> dict[str, Any]:
    """Compute the Table 2 statistics and summary counts for one processed dataset.

    Args:
        dataset_dir: Directory holding schemas.json, test.jsonl and optionally train.jsonl.

    Returns:
        Dict of statistics: databases, tables, columns, test and train sizes, foreign-key
        columns, mean gold tables per test question, and the test difficulty distribution.
    """
    schemas = read_json(dataset_dir / "schemas.json")
    test = read_jsonl(dataset_dir / "test.jsonl")
    train_path = dataset_dir / "train.jsonl"
    tables = [table for database in schemas.values() for table in database]
    columns = [column for table in tables for column in table["columns"]]
    difficulty = Counter(str(instance["difficulty"]) for instance in test)
    return {
        "databases": len(schemas),
        "tables": len(tables),
        "columns": len(columns),
        "test": len(test),
        "train": len(read_jsonl(train_path)) if train_path.exists() else None,
        "fk_columns": sum("foreign_key" in column for column in columns),
        "mean_gold_tables": round(sum(len(i["gold_tables"]) for i in test) / len(test), 2),
        "difficulty": dict(sorted(difficulty.items())),
    }


def compare_with_paper(name: str, stats: dict[str, Any], expected: dict[str, int]) -> list[str]:
    """List the statistics that differ from the paper's Table 2.

    Args:
        name: Dataset name.
        stats: Output of dataset_statistics.
        expected: Expected values for this dataset (configs/data.yaml, `expected`).

    Returns:
        Human-readable mismatch descriptions (empty if everything matches).
    """
    return [
        f"{name}.{key}: ours {stats[key]} vs paper {value}"
        for key, value in expected.items()
        if stats[key] != value
    ]


def main() -> None:
    """Check every processed dataset and write the statistics table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", help="data config")
    parser.add_argument("--output", default="results/tables/data_stats.csv", help="CSV output")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    processed_dir = PROJECT_ROOT / config["processed_dir"]
    rows, mismatches = [], []
    for name, expected in config["expected"].items():
        stats = dataset_statistics(processed_dir / name)
        mismatches += compare_with_paper(name, stats, expected)
        rows.append({"dataset": name, **stats})
        logger.info("%s: %s", name, stats)

    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    logger.info("wrote %s", output.relative_to(PROJECT_ROOT))
    if mismatches:
        logger.error("differences from the paper's Table 2:\n  %s", "\n  ".join(mismatches))
        sys.exit(1)
    logger.info("all statistics match the paper's Table 2")


if __name__ == "__main__":
    main()
