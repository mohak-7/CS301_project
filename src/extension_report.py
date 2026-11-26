"""Results tables and paired tests for the extensions.

Implements the evaluation of implementation_plan.md, §9 (hybrid seeds and cross-encoder
reranking) and §10 (LLM table descriptions). Compares the new seed methods (with and without graph
scoring) against the earlier results and DBCopilot's reported numbers, and runs paired t-tests
that separate the effect of each step.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.analysis.load import DATASETS
from src.analysis.significance import paired_test
from src.common import PROJECT_ROOT, load_config, read_jsonl, setup_logging

logger = logging.getLogger(__name__)
METRICS = ["DR@1", "DR@5", "pooled TR@5", "pooled TR@15"]
TEST_METRICS = ["DR@1", "DR@5", "TR@5", "TR@15"]
# Output group of each seed method's test runs (src/run_graphrag.py --output-group).
GROUPS = {
    "bm25": "main", "sxfmr": "main", "hybrid": "extension", "hybrid_rr": "extension",
    "desc": "descriptions", "hybrid_desc": "descriptions", "hybrid_desc_rr": "descriptions",
}  # fmt: skip
# Comparisons per extension: (candidate, baseline).
COMPARISONS = {
    "extension": [
        ("hybrid-G0", "bm25"), ("hybrid-G0", "sxfmr"), ("hybrid_rr-G0", "hybrid-G0"),
        ("hybrid_rr-G1", "hybrid_rr-G0"), ("hybrid_rr-G2", "hybrid_rr-G0"),
        ("hybrid_rr-G3", "hybrid_rr-G0"), ("hybrid_rr-G2", "bm25-G2"),
        ("hybrid_rr-G2", "sxfmr-G2"), ("hybrid_rr-G1", "bm25-G1"), ("hybrid_rr-G1", "sxfmr-G2"),
    ],
    "descriptions": [
        ("desc-G0", "sxfmr"), ("hybrid_desc-G0", "hybrid-G0"),
        ("hybrid_desc_rr-G0", "hybrid_desc-G0"), ("hybrid_desc_rr-G0", "hybrid_rr-G0"),
        ("hybrid_desc_rr-G1", "hybrid_desc_rr-G0"), ("hybrid_desc_rr-G2", "hybrid_desc_rr-G0"),
        ("hybrid_desc_rr-G3", "hybrid_desc_rr-G0"), ("hybrid_desc_rr-G1", "hybrid_rr-G1"),
        ("hybrid_desc_rr-G2", "hybrid_rr-G2"),
    ],
}  # fmt: skip
# Test-summary files read for each extension's summary table (besides the Phase 4/6 files).
SUMMARY_FILES = {
    "extension": ["extension_test_summary.csv"],
    "descriptions": ["extension_test_summary.csv", "descriptions_test_summary.csv"],
}


def records_path(results: Path, method: str, dataset: str) -> Path:
    """Return the per-question file of a method ("bm25", "bm25-G2", "hybrid_rr-G0", ...)."""
    seed, _, variant = method.partition("-")
    if not variant:
        return results / seed / f"{dataset}.jsonl"
    group = GROUPS.get(seed, "extension")
    return results / group / f"graphrag_{seed}_{variant}_R-main" / f"{dataset}_test.jsonl"


def load(results: Path, method: str, dataset: str) -> dict[str, dict]:
    """Load one method's per-question records keyed by question id."""
    return {r["id"]: r for r in read_jsonl(records_path(results, method, dataset))}


def summary_table(tables: Path, reference: dict, stage: str) -> pd.DataFrame:
    """Dataset-level metrics of the old and new configurations, with DBCopilot's reported row.

    Args:
        tables: results/tables.
        reference: Paper numbers (DBCopilot's table recall is pooled).
        stage: "extension" (§9) or "descriptions" (§10).

    Returns:
        One row per (dataset, method).
    """
    old = pd.concat(
        [
            pd.read_csv(tables / "baselines_summary.csv"),
            pd.read_csv(tables / "graphrag_test_summary.csv"),
        ]
    )
    new = [pd.read_csv(tables / name) for name in SUMMARY_FILES[stage]]
    frame = pd.concat([old, *new])
    frame = frame[~frame["method"].str.contains("R-(?:seed|hier)", regex=True)].copy()
    frame["method"] = (
        frame["method"]
        .str.replace("graphrag_", "")
        .str.replace("_R-main", "")
        .str.replace(r"_(G\d)$", r"-\1", regex=True)
    )
    rows = [frame[["dataset", "method", *METRICS]]]
    for dataset in DATASETS:
        paper = reference[dataset]["dbcopilot"]
        rows.append(
            pd.DataFrame(
                [{"dataset": dataset, "method": "DBCopilot (reported)", "DR@1": paper["DR@1"],
                  "DR@5": paper["DR@5"], "pooled TR@5": paper["TR@5"],
                  "pooled TR@15": paper["TR@15"]}]
            )
        )  # fmt: skip
    return pd.concat(rows, ignore_index=True).round(2)


def effect_tests(results: Path, stage: str) -> pd.DataFrame:
    """Paired t-tests for the effect of each extension step on every dataset.

    §9 comparisons: hybrid seeds vs. each single retriever; reranking vs. hybrid; graph scoring
    (G1-G3) on reranked seeds vs. reranked seeds alone; and new configurations vs. the best Phase 6
    configurations. §10 comparisons: description embeddings vs. schema embeddings; each
    description-based seed vs. its §9 counterpart; and graph scoring on the new seeds.

    Args:
        results: The results/ directory.
        stage: "extension" (§9) or "descriptions" (§10).

    Returns:
        One row per (dataset, comparison, metric) with the difference and p-value.
    """
    rows = []
    for dataset in DATASETS:
        for candidate, baseline in COMPARISONS[stage]:
            after, before = load(results, candidate, dataset), load(results, baseline, dataset)
            for metric in TEST_METRICS:
                difference, p_value = paired_test(before, after, metric)
                mark = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                rows.append(
                    {"dataset": dataset, "comparison": f"{candidate} vs {baseline}",
                     "metric": metric, "difference": round(difference, 2),
                     "p_value": p_value, "significance": mark}
                )  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    """Write results/tables/{stage}_summary.csv and results/tables/{stage}_tests.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="configs/reference.yaml")
    parser.add_argument("--stage", choices=sorted(COMPARISONS), default="extension")
    args = parser.parse_args()

    setup_logging()
    results = PROJECT_ROOT / "results"
    tables = results / "tables"
    summary = summary_table(tables, load_config(args.reference), args.stage)
    summary.to_csv(tables / f"{args.stage}_summary.csv", index=False)
    tests = effect_tests(results, args.stage)
    tests.to_csv(tables / f"{args.stage}_tests.csv", index=False)
    logger.info("wrote %s summary and %d tests to %s", args.stage, len(tests), tables)


if __name__ == "__main__":
    main()
