"""Schema-routing metrics, with a cross-check against `ranx`.

Computes database recall, table recall (per-question and pooled), full-schema recall and mAP.
Implements the evaluation pipeline (implementation_plan.md, §2.3, §4.2, D11). Run as a script to
evaluate one predictions file written by a retrieval run.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate

from src.common import read_jsonl, setup_logging
from src.rank_databases import RankedDatabases, RankedTables
from src.retrievers.base import table_id

logger = logging.getLogger(__name__)


def gold_table_ids(instance: dict[str, Any]) -> list[str]:
    """Return the gold tables of an instance as collection-wide table ids.

    Args:
        instance: Processed instance with `gold_db` and `gold_tables`.

    Returns:
        List of "db_id.table_name" ids.
    """
    return [table_id(instance["gold_db"], name) for name in instance["gold_tables"]]


def average_precision(ranked_ids: list[str], gold: set[str]) -> float:
    """Average precision of a ranked list, as computed by ranx's `map` (no cutoff).

    Args:
        ranked_ids: Retrieved ids, best first.
        gold: Relevant ids.

    Returns:
        Mean over gold items of precision at each hit's rank; missed items count as 0.
    """
    hits, precision_sum = 0, 0.0
    for rank, identifier in enumerate(ranked_ids, start=1):
        if identifier in gold:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(gold) if gold else 0.0


def question_metrics(
    instance: dict[str, Any],
    ranked_databases: RankedDatabases,
    ranked_tables: RankedTables,
    database_ks: list[int],
    table_ks: list[int],
    full_schema_ks: list[int],
) -> dict[str, float]:
    """Compute all metrics of one question.

    Args:
        instance: Processed instance (gold database and tables).
        ranked_databases: Database ranking, best first.
        ranked_tables: Table ranking over the whole collection, best first.
        database_ks: Cutoffs for database recall.
        table_ks: Cutoffs for table recall.
        full_schema_ks: Cutoffs for full-schema recall.

    Returns:
        Dict with DR@k (database recall), TR@k (per-question table recall), hits@k (gold tables
        found in the top k, used for pooled recall), FS@k (1 if every gold table is in the top
        k), AP (average precision of the table list) and n_gold.

    Raises:
        ValueError: If the instance has no gold tables; table recall is undefined for it.
    """
    gold = set(gold_table_ids(instance))
    if not gold:
        raise ValueError(f"{instance['id']} has no gold tables; remove it before evaluation")
    database_ids = [db_id for db_id, _ in ranked_databases]
    table_ids = [identifier for identifier, _ in ranked_tables]
    metrics: dict[str, float] = {"n_gold": len(gold)}
    for k in database_ks:
        metrics[f"DR@{k}"] = float(instance["gold_db"] in database_ids[:k])
    for k in sorted(set(table_ks) | set(full_schema_ks)):
        hits = len(gold & set(table_ids[:k]))
        if k in table_ks:
            metrics[f"TR@{k}"] = hits / len(gold)
            metrics[f"hits@{k}"] = hits
        if k in full_schema_ks:
            metrics[f"FS@{k}"] = float(hits == len(gold))
    metrics["AP"] = average_precision(table_ids, gold)
    return metrics


def summarise(per_question: list[dict[str, float]], table_ks: list[int]) -> dict[str, float]:
    """Aggregate per-question metrics into dataset-level scores (in percent).

    Per-question metrics are averaged over questions (macro). Pooled table recall divides all
    gold tables found by all gold tables (micro), as DBCopilot's own metric does (D11).

    Args:
        per_question: Outputs of `question_metrics`.
        table_ks: Cutoffs for table recall.

    Returns:
        Dict of metric name to percentage, plus `questions`.
    """
    keys = [k for k in per_question[0] if k not in ("n_gold",) and not k.startswith("hits@")]
    summary = {key: 100 * float(np.mean([q[key] for q in per_question])) for key in keys}
    summary["mAP"] = summary.pop("AP")
    total_gold = sum(q["n_gold"] for q in per_question)
    for k in table_ks:
        summary[f"pooled TR@{k}"] = 100 * sum(q[f"hits@{k}"] for q in per_question) / total_gold
    summary["questions"] = len(per_question)
    return summary


def ranx_summary(
    instances: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    database_ks: list[int],
    table_ks: list[int],
) -> dict[str, float]:
    """Compute per-question recall with `ranx`, exactly as the authors' baseline script does.

    `ranx` breaks score ties in its own order, so small differences from `summarise` are
    possible when tied tables straddle a cutoff; the run log reports the difference.

    Args:
        instances: Processed instances.
        predictions: Prediction records with `ranked_databases` and `ranked_tables`.
        database_ks: Cutoffs for database recall.
        table_ks: Cutoffs for table recall.

    Returns:
        Dict of "ranx DR@k" and "ranx TR@k" percentages.
    """
    database_qrels = Qrels({i["id"]: {i["gold_db"]: 1} for i in instances})
    table_qrels = Qrels({i["id"]: {t: 1 for t in gold_table_ids(i)} for i in instances})
    database_run = Run({p["id"]: dict(p["ranked_databases"]) for p in predictions})
    table_run = Run({p["id"]: dict(p["ranked_tables"]) for p in predictions})
    database_scores = ranx_evaluate(
        database_qrels, database_run, [f"recall@{k}" for k in database_ks]
    )
    table_scores = ranx_evaluate(table_qrels, table_run, [f"recall@{k}" for k in table_ks])
    summary = {f"ranx DR@{k}": 100 * database_scores[f"recall@{k}"] for k in database_ks}
    summary |= {f"ranx TR@{k}": 100 * table_scores[f"recall@{k}"] for k in table_ks}
    return summary


def main() -> None:
    """Re-evaluate a saved predictions file and log its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="results/<method>/<dataset>.jsonl")
    parser.add_argument("--table-ks", type=int, nargs="+", default=[5, 10, 15, 20])
    args = parser.parse_args()

    setup_logging()
    records = read_jsonl(args.predictions)
    summary = summarise([record["metrics"] for record in records], args.table_ks)
    for name, value in summary.items():
        logger.info("%-14s %.2f", name, value)


if __name__ == "__main__":
    main()
