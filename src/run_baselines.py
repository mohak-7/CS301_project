"""Run the BM25 and SXFMR baselines on all benchmarks and validate them against the paper.

Implements Phase 4 (implementation_plan.md, §4.3, §4.4, D2): computes full score matrices
(cached for Phase 5), applies the paper's top-100 ranking rule, evaluates, and compares the
per-question metrics with the baseline rows of the paper's Tables 3 and 4.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.common import (
    PROJECT_ROOT,
    load_config,
    read_json,
    read_jsonl,
    set_seed,
    setup_logging,
    write_jsonl,
)
from src.evaluate import question_metrics, ranx_summary, summarise
from src.rank_databases import rank_databases, top_tables
from src.retrievers.base import ScoreMatrix
from src.retrievers.bm25 import bm25_scores
from src.retrievers.dense import dense_scores

logger = logging.getLogger(__name__)
COMPARED_METRICS = ["DR@1", "DR@5", "TR@5", "TR@15"]


def compute_scores(
    method: str,
    dataset: str,
    config: dict[str, Any],
    force: bool = False,
    split: str = "test",
) -> ScoreMatrix:
    """Return the full score matrix of a method on a dataset, computing it if not cached.

    Args:
        method: "bm25" or "sxfmr".
        dataset: Dataset name.
        config: Baseline configuration (configs/baselines.yaml).
        force: Recompute even if a cached matrix exists.
        split: "test", or "train" for the tuning questions (Spider and Bird only, D7).

    Returns:
        ScoreMatrix for every question of the split and every table of the collection.
    """
    file_name = f"{dataset}.npz" if split == "test" else f"{dataset}_{split}.npz"
    cache_path = PROJECT_ROOT / config["scores_dir"] / method / file_name
    if cache_path.exists() and not force:
        return ScoreMatrix.load(cache_path)
    processed = PROJECT_ROOT / config["processed_dir"] / dataset
    schemas = read_json(processed / "schemas.json")
    instances = read_jsonl(processed / f"{split}.jsonl")
    collection = config["collections"][dataset]
    retriv_dir = PROJECT_ROOT / config["retriv_dir"]
    settings = config["methods"][method]
    started = time.perf_counter()
    if method == "bm25":
        matrix = bm25_scores(schemas, instances, f"{collection}_bm25", retriv_dir, **settings)
    else:
        matrix = dense_scores(schemas, instances, f"{collection}_sxfmr", retriv_dir, **settings)
    logger.info("%s/%s/%s: scored in %.1f s", method, dataset, split, time.perf_counter() - started)
    matrix.save(cache_path)
    return matrix


def predict_and_evaluate(
    matrix: ScoreMatrix, instances: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply the paper's ranking rule to a score matrix and evaluate every question.

    Args:
        matrix: Full score matrix (rows in the same order as `instances`).
        instances: Processed test instances.
        config: Baseline configuration (cutoff and metric cutoffs).

    Returns:
        Prediction records with ranked databases, ranked tables (top `cutoff`) and metrics.
    """
    metric_settings = config["metrics"]
    records = []
    for row, instance in enumerate(instances):
        ranked_tables = top_tables(matrix.scores[row], matrix.table_ids, config["cutoff"])
        ranked_databases = rank_databases(ranked_tables)
        metrics = question_metrics(
            instance,
            ranked_databases,
            ranked_tables,
            metric_settings["database_ks"],
            metric_settings["table_ks"],
            metric_settings["full_schema_ks"],
        )
        records.append(
            {
                "id": instance["id"],
                "ranked_databases": ranked_databases,
                "ranked_tables": ranked_tables,
                "metrics": metrics,
            }
        )
    return records


def run_one(method: str, dataset: str, config: dict[str, Any], force: bool) -> dict[str, Any]:
    """Score, rank, evaluate and save one method on one dataset.

    Args:
        method: "bm25" or "sxfmr".
        dataset: Dataset name.
        config: Baseline configuration.
        force: Recompute cached score matrices.

    Returns:
        Summary row: method, dataset, our metrics and the ranx cross-check.
    """
    instances = read_jsonl(PROJECT_ROOT / config["processed_dir"] / dataset / "test.jsonl")
    matrix = compute_scores(method, dataset, config, force)
    assert matrix.question_ids == [i["id"] for i in instances], "score rows out of order"
    records = predict_and_evaluate(matrix, instances, config)
    write_jsonl(PROJECT_ROOT / config["results_dir"] / method / f"{dataset}.jsonl", records)
    metric_settings = config["metrics"]
    summary = summarise([r["metrics"] for r in records], metric_settings["table_ks"])
    summary |= ranx_summary(
        instances, records, metric_settings["database_ks"], metric_settings["table_ks"]
    )
    return {"method": method, "dataset": dataset, **summary}


def validation_table(
    summaries: list[dict[str, Any]], reference: dict[str, Any], tolerance: float
) -> pd.DataFrame:
    """Compare our per-question metrics with the paper's baseline rows (§4.4).

    Args:
        summaries: Outputs of `run_one`.
        reference: Paper numbers (configs/reference.yaml).
        tolerance: Largest acceptable absolute difference, in recall points.

    Returns:
        One row per (method, dataset, metric) with ours, paper, difference and a within flag.
    """
    rows = []
    for summary in summaries:
        paper = reference[summary["dataset"]][summary["method"]]
        for metric in COMPARED_METRICS:
            difference = summary[metric] - paper[metric]
            rows.append(
                {
                    "dataset": summary["dataset"],
                    "method": summary["method"],
                    "metric": metric,
                    "ours": round(summary[metric], 2),
                    "paper": paper[metric],
                    "difference": round(difference, 2),
                    "within_tolerance": abs(difference) <= tolerance,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run all baselines, write per-question results, summaries and the validation table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--reference", default="configs/reference.yaml")
    parser.add_argument("--methods", nargs="+", default=["bm25", "sxfmr"])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--force", action="store_true", help="recompute cached score matrices")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seed(config["seed"])
    datasets = args.datasets or list(config["collections"])
    summaries = [
        run_one(method, dataset, config, args.force)
        for method in args.methods
        for dataset in datasets
    ]
    validation = validation_table(summaries, load_config(args.reference), config["tolerance"])
    logger.info("validation against the paper:\n%s", validation.to_string(index=False))
    # The summary tables cover every method and dataset; a partial run only logs, so it cannot
    # overwrite the full tables with a subset.
    if args.datasets is not None or set(args.methods) != set(config["methods"]):
        logger.info("partial run: summary tables in results/tables/ left unchanged")
        return
    tables_dir = PROJECT_ROOT / config["results_dir"] / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).round(2).to_csv(tables_dir / "baselines_summary.csv", index=False)
    validation.to_csv(tables_dir / "baseline_validation.csv", index=False)
    write_markdown(validation, tables_dir / "baseline_validation.md")


def write_markdown(validation: pd.DataFrame, path: Path) -> None:
    """Write the validation table as Markdown, one row per (dataset, method).

    Args:
        validation: Output of `validation_table`.
        path: Destination .md file.
    """
    lines = ["| Dataset | Method | " + " | ".join(COMPARED_METRICS) + " |"]
    lines.append("|---|---|" + "---|" * len(COMPARED_METRICS))
    for (dataset, method), group in validation.groupby(["dataset", "method"], sort=False):
        cells = [
            f"{row.ours:.2f} ({row.paper:.2f}, {row.difference:+.2f})" for row in group.itertuples()
        ]
        lines.append(f"| {dataset} | {method} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cells: ours (paper, difference). Table recall is per-question (macro).")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
