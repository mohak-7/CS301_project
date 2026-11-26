"""Tune the hybrid weight and the reranking settings on training questions.

Implements the seed tuning of the extension (implementation_plan.md, §9, D7): the same fixed
sample of Spider and Bird training questions as Phase 6, evaluated with seeds only (G0) and the
paper's ranking rule. Writes configs/seeds.yaml. No test question is read.
"""

import argparse
import itertools
import logging
from typing import Any

import pandas as pd
import yaml

from src.common import PROJECT_ROOT, load_config, read_json, set_seed, setup_logging
from src.retrievers.graphrag import GraphRAGParams
from src.retrievers.hybrid import hybrid_scores
from src.retrievers.rerank import apply_rerank, cross_encoder_scores
from src.tune import TuningSet, sample_training_sets, score_setting

logger = logging.getLogger(__name__)


def hybrid_sets(bm25_set: TuningSet, dense_set: TuningSet, weight: float) -> TuningSet:
    """Combine the BM25 and dense tuning samples into hybrid seeds with the given weight."""
    return [
        (instances, hybrid_scores(bm25, dense, weight), graph)
        for (instances, bm25, graph), (_, dense, _) in zip(bm25_set, dense_set, strict=True)
    ]


def best_row(rows: list[dict[str, Any]], objective: str, tie_breaker: str) -> dict[str, Any]:
    """Return the row with the highest objective, then tie-breaker (first in grid order)."""
    return max(rows, key=lambda row: (round(row[objective], 6), round(row[tie_breaker], 6)))


def tune_hybrid(
    bm25_set: TuningSet, dense_set: TuningSet, config: dict[str, Any], metrics: dict[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    """Choose the BM25 weight of the hybrid seeds (G0, paper's ranking rule).

    Args:
        bm25_set: BM25 tuning sample.
        dense_set: Dense tuning sample (same questions).
        config: Seed tuning configuration.
        metrics: Metric cutoffs.

    Returns:
        (chosen weight, one row per weight tried).
    """
    params = GraphRAGParams(variant="G0", cutoff=config["cutoff"])
    rows = [
        {
            "stage": "hybrid",
            "weight": w,
            **score_setting(params, hybrid_sets(bm25_set, dense_set, w), metrics),
        }
        for w in config["hybrid_weights"]
    ]
    return best_row(rows, config["objective"], config["tie_breaker"])["weight"], rows


def tune_rerank(
    hybrid_set: TuningSet, config: dict[str, Any], metrics: dict[str, Any], processed: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose how many tables to rerank and the cross-encoder's weight.

    The cross-encoder scores each question's largest pool once; smaller top-k values rerank a
    prefix of the same pool.

    Args:
        hybrid_set: Hybrid tuning sample with the chosen weight.
        config: Seed tuning configuration.
        metrics: Metric cutoffs.
        processed: data/processed directory.

    Returns:
        (chosen {top_k, weight}, one row per combination tried).
    """
    rerank = config["rerank"]
    pool = max(rerank["top_k"])
    cross = []
    for (instances, seeds, _), dataset in zip(hybrid_set, config["datasets"], strict=True):
        schemas = read_json(processed / dataset / "schemas.json")
        cross.append(
            cross_encoder_scores(
                seeds, instances, schemas, rerank["model"], pool,
                rerank["max_length"], rerank["batch_size"], rerank["device"],
            )
        )  # fmt: skip
    params = GraphRAGParams(variant="G0", cutoff=config["cutoff"])
    rows = []
    for top_k, weight in itertools.product(rerank["top_k"], rerank["weight"]):
        reranked = [
            (instances, apply_rerank(seeds, scores, top_k, weight), graph)
            for (instances, seeds, graph), scores in zip(hybrid_set, cross, strict=True)
        ]
        rows.append(
            {
                "stage": "rerank",
                "top_k": top_k,
                "weight": weight,
                **score_setting(params, reranked, metrics),
            }
        )
    best = best_row(rows, config["objective"], config["tie_breaker"])
    return {"top_k": best["top_k"], "weight": best["weight"]}, rows


def main() -> None:
    """Tune the hybrid weight, then the reranking settings, and write configs/seeds.yaml."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tuning_seeds.yaml")
    parser.add_argument("--baseline-config", default="configs/baselines.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    baseline_config = load_config(args.baseline_config)
    set_seed(config["seed"])
    metrics = baseline_config["metrics"]
    # The Phase 6 sampler draws the same questions for both seed methods (re-seeded generator).
    sampler_config = {key: config[key] for key in ("seed", "sample_size", "datasets")}
    bm25_set = sample_training_sets("bm25", sampler_config, baseline_config)
    dense_set = sample_training_sets("sxfmr", sampler_config, baseline_config)
    weight, hybrid_rows = tune_hybrid(bm25_set, dense_set, config, metrics)
    logger.info("hybrid: best BM25 weight %.1f", weight)
    processed = PROJECT_ROOT / baseline_config["processed_dir"]
    chosen, rerank_rows = tune_rerank(
        hybrid_sets(bm25_set, dense_set, weight), config, metrics, processed
    )
    logger.info("rerank: best %s", chosen)

    results = PROJECT_ROOT / config["results"]
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hybrid_rows + rerank_rows).round(4).to_csv(results / "seeds.csv", index=False)
    rerank = config["rerank"]
    settings = {
        "hybrid": {"weight": weight},
        "rerank": {
            "model": rerank["model"],
            "max_length": rerank["max_length"],
            "batch_size": rerank["batch_size"],
            "device": rerank["device"],
            "pool_size": chosen["top_k"],
            "top_k": chosen["top_k"],
            "weight": chosen["weight"],
        },
    }
    output = PROJECT_ROOT / config["output"]
    output.write_text(
        "# Seed settings tuned on training questions by src/tune_seeds.py (§9).\n"
        + yaml.safe_dump(settings, sort_keys=False)
    )
    logger.info("wrote %s", output.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
