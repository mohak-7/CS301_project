"""GQ3: how consistent are each method's strengths across benchmark characteristics?

Implements implementation_plan.md, §7.3: results by query difficulty, by database size (the
view of the paper's Figure 7a), the drop under semantic mismatch (Spider vs. Spider-syn and
Spider-realistic), and table recall@k for k = 1..50 (the paper's Figure 7b).
"""

from typing import Any

import numpy as np
import pandas as pd

from src.analysis.load import DATASETS, SEEDS, VARIANTS, Records, method_name
from src.evaluate import gold_table_ids

ROBUSTNESS = {"spider_syn": "Spider-syn", "spider_realistic": "Spider-realistic"}


def by_difficulty(
    results: dict[tuple[str, str, str], Records], instances: dict[str, list[dict[str, Any]]]
) -> pd.DataFrame:
    """Mean DR@1, TR@5 and TR@15 per difficulty level, for datasets that have labels.

    Args:
        results: Output of `load_all`.
        instances: Test instances per dataset.

    Returns:
        One row per (dataset, method, difficulty) with question count and metrics.
    """
    rows = []
    for (seed, variant, dataset), records in results.items():
        labels = {i["id"]: i["difficulty"] for i in instances[dataset]}
        if not any(labels.values()):
            continue
        frame = pd.DataFrame(
            [{**r["metrics"], "difficulty": labels[q]} for q, r in records.items()]
        )
        for level, part in frame.dropna(subset=["difficulty"]).groupby("difficulty"):
            rows.append(
                {
                    "dataset": dataset,
                    "method": method_name(seed, variant),
                    "difficulty": level,
                    "questions": len(part),
                    **{m: round(100 * part[m].mean(), 2) for m in ("DR@1", "TR@5", "TR@15")},
                }
            )
    return pd.DataFrame(rows)


def by_database_size(
    results: dict[tuple[str, str, str], Records],
    instances: dict[str, list[dict[str, Any]]],
    schemas: dict[str, dict[str, Any]],
    dataset: str = "spider",
    min_questions: int = 10,
) -> pd.DataFrame:
    """Table mAP by the number of tables in the gold database (paper's Figure 7a view).

    Args:
        results: Output of `load_all`.
        instances: Test instances per dataset.
        schemas: Schema collection per dataset.
        dataset: Dataset to analyse.
        min_questions: Sizes with fewer questions are left out (too noisy to plot).

    Returns:
        One row per (method, database size) with question count and mAP.
    """
    sizes = {i["id"]: len(schemas[dataset][i["gold_db"]]) for i in instances[dataset]}
    counts = pd.Series(sizes).value_counts()
    kept = set(counts[counts >= min_questions].index)
    rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            records = results[(seed, variant, dataset)]
            frame = pd.DataFrame(
                [{"size": sizes[q], "AP": r["metrics"]["AP"]} for q, r in records.items()]
            )
            for size, part in frame[frame["size"].isin(kept)].groupby("size"):
                rows.append(
                    {
                        "method": method_name(seed, variant),
                        "tables_in_database": int(size),
                        "questions": len(part),
                        "mAP": round(100 * part["AP"].mean(), 2),
                    }
                )
    return pd.DataFrame(rows)


def mismatch_drop(summary: pd.DataFrame, reference: dict[str, Any]) -> pd.DataFrame:
    """Drop in DR@1 and pooled TR@5 from Spider to each robustness set, ours and DBCopilot's.

    Args:
        summary: Our per-dataset summaries (method, dataset, metric columns).
        reference: Paper numbers (configs/reference.yaml); DBCopilot's table recall is pooled.

    Returns:
        One row per (method, robustness set) with the Spider value, the robustness value and
        the drop, for DR@1 and pooled TR@5.
    """
    rows = []
    for method in summary["method"].unique():
        spider = summary[(summary.method == method) & (summary.dataset == "spider")].iloc[0]
        for dataset, label in ROBUSTNESS.items():
            other = summary[(summary.method == method) & (summary.dataset == dataset)].iloc[0]
            rows.append(_drop_row(method, label, spider, other, "pooled TR@5"))
    dbcopilot_spider = {**reference["spider"]["dbcopilot"]}
    for dataset, label in ROBUSTNESS.items():
        other = reference[dataset]["dbcopilot"]
        rows.append(_drop_row("DBCopilot (reported)", label, dbcopilot_spider, other, "TR@5"))
    return pd.DataFrame(rows).round(2)


def _drop_row(method: str, label: str, spider: Any, other: Any, table_metric: str) -> dict:
    """Build one row of `mismatch_drop`."""
    return {
        "method": method,
        "test_set": label,
        "DR@1 Spider": spider["DR@1"],
        "DR@1 drop": spider["DR@1"] - other["DR@1"],
        "pooled TR@5 Spider": spider[table_metric],
        "pooled TR@5 drop": spider[table_metric] - other[table_metric],
    }


def recall_curve(
    results: dict[tuple[str, str, str], Records],
    instances: dict[str, list[dict[str, Any]]],
    dataset: str = "spider",
    max_k: int = 50,
) -> pd.DataFrame:
    """Per-question table recall@k for k = 1..max_k (paper's Figure 7b view).

    Args:
        results: Output of `load_all`.
        instances: Test instances per dataset.
        dataset: Dataset to analyse.
        max_k: Largest cutoff.

    Returns:
        One row per (method, k) with the mean table recall in percent.
    """
    gold = {i["id"]: set(gold_table_ids(i)) for i in instances[dataset]}
    rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            records = results[(seed, variant, dataset)]
            hits = np.zeros(max_k)
            for question_id, record in records.items():
                ranked = [t for t, _ in record["ranked_tables"][:max_k]]
                found = np.cumsum(
                    [t in gold[question_id] for t in ranked] + [0] * (max_k - len(ranked))
                )
                hits += found / len(gold[question_id])
            rows += [
                {
                    "method": method_name(seed, variant),
                    "k": k + 1,
                    "TR": 100 * hits[k] / len(records),
                }
                for k in range(max_k)
            ]
    return pd.DataFrame(rows).round(2)


def dataset_overview(summary: pd.DataFrame) -> pd.DataFrame:
    """Best GraphRAG gain over its seed retriever per dataset (dataset-scale view of GQ3).

    Args:
        summary: Our per-dataset summaries.

    Returns:
        One row per (dataset, seed) with the seed's DR@1/TR@5 and the best variant's gain.
    """
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            part = summary[summary.dataset == dataset]
            base = part[part.method == seed].iloc[0]
            variants = part[part.method.str.startswith(f"{seed}-")]
            for metric in ("DR@1", "pooled TR@5", "pooled TR@15"):
                best = variants.loc[variants[metric].idxmax()]
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "metric": metric,
                        "seed value": base[metric],
                        "best variant": best["method"],
                        "gain": round(best[metric] - base[metric], 2),
                    }
                )
    return pd.DataFrame(rows)
