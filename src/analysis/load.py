"""Load processed instances and per-question results for the Phase 7 analysis.

Implements the shared data access of implementation_plan.md, Phase 7. Each seed retriever
forms a family: the baseline itself (equal to GraphRAG G0, §5.7) and GraphRAG G1-G3 (R-main).
"""

from pathlib import Path
from typing import Any

from src.common import PROJECT_ROOT, read_json, read_jsonl

DATASETS = ["spider", "bird", "fiben", "spider_syn", "spider_realistic"]
SEEDS = ["bm25", "sxfmr"]
VARIANTS = ["G0", "G1", "G2", "G3"]
Records = dict[str, dict[str, Any]]  # instance id -> prediction record


def method_name(seed: str, variant: str) -> str:
    """Return the display name of a method, e.g. "bm25" for G0 and "bm25-G2" otherwise."""
    return seed if variant == "G0" else f"{seed}-{variant}"


def result_path(seed: str, variant: str, dataset: str, results_dir: Path) -> Path:
    """Return the per-question results file of one method on one dataset.

    Args:
        seed: "bm25" or "sxfmr".
        variant: G0 (the baseline run) or G1-G3 (GraphRAG, R-main).
        dataset: Dataset name.
        results_dir: The results/ directory.

    Returns:
        Path to the JSON Lines file written in Phase 4 or Phase 6.
    """
    if variant == "G0":
        return results_dir / seed / f"{dataset}.jsonl"
    return results_dir / "main" / f"graphrag_{seed}_{variant}_R-main" / f"{dataset}_test.jsonl"


def load_records(seed: str, variant: str, dataset: str, results_dir: Path) -> Records:
    """Load one method's per-question records, keyed by instance id.

    Args:
        seed: "bm25" or "sxfmr".
        variant: G0-G3.
        dataset: Dataset name.
        results_dir: The results/ directory.

    Returns:
        Mapping from instance id to its record (ranked databases, ranked tables, metrics).
    """
    return {r["id"]: r for r in read_jsonl(result_path(seed, variant, dataset, results_dir))}


def load_dataset(dataset: str, processed_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the test instances and the schema collection of a dataset.

    Args:
        dataset: Dataset name.
        processed_dir: The data/processed directory.

    Returns:
        (test instances, schemas).
    """
    directory = processed_dir / dataset
    return read_jsonl(directory / "test.jsonl"), read_json(directory / "schemas.json")


def load_all(results_dir: Path | None = None) -> dict[tuple[str, str, str], Records]:
    """Load every (seed, variant, dataset) result set.

    Args:
        results_dir: The results/ directory (default: PROJECT_ROOT/results).

    Returns:
        Mapping from (seed, variant, dataset) to records.
    """
    results_dir = results_dir or PROJECT_ROOT / "results"
    return {
        (seed, variant, dataset): load_records(seed, variant, dataset, results_dir)
        for seed in SEEDS
        for variant in VARIANTS
        for dataset in DATASETS
    }
