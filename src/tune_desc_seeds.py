"""Tune the description-based seeds on training questions.

Implements the seed tuning of the description extension (implementation_plan.md, §10, D7): the
mixing weights of BM25, schema embeddings and description embeddings (a grid over the simplex),
then the cross-encoder reranking depth and weight with descriptions as the table text. Uses the
Phase 6 training sample and seeds only (G0). Adds the result to configs/seeds.yaml.
"""

import argparse
import itertools
import logging
from typing import Any

import pandas as pd
import yaml

from src.common import PROJECT_ROOT, load_config, read_json, set_seed, setup_logging
from src.retrievers.descriptions import description_texts, load_descriptions
from src.retrievers.graphrag import GraphRAGParams
from src.retrievers.hybrid import weighted_mix
from src.retrievers.rerank import apply_rerank, cross_encoder_scores
from src.tune import TuningSet, sample_training_sets, score_setting
from src.tune_seeds import best_row

logger = logging.getLogger(__name__)
CHANNELS = ("bm25", "schema", "desc")


def simplex(step: float) -> list[tuple[float, float, float]]:
    """All weight triples on a grid with the given step that sum to 1."""
    n = round(1 / step)
    return [
        (round(a * step, 2), round(b * step, 2), round((n - a - b) * step, 2))
        for a in range(n + 1)
        for b in range(n + 1 - a)
    ]


def mixed_set(sets: list[TuningSet], weights: tuple[float, ...]) -> TuningSet:
    """Mix the per-channel tuning samples (same questions) with the given weights."""
    return [
        (parts[0][0], weighted_mix([p[1] for p in parts], list(weights)), parts[0][2])
        for parts in zip(*sets, strict=True)
    ]


def tune_mix(
    sets: list[TuningSet], config: dict[str, Any], metrics: dict[str, Any]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Choose the channel weights (G0, paper's ranking rule).

    Args:
        sets: Tuning samples for BM25, schema embeddings and description embeddings.
        config: Tuning configuration.
        metrics: Metric cutoffs.

    Returns:
        (chosen weights by channel, one row per weight triple).
    """
    params = GraphRAGParams(variant="G0", cutoff=config["cutoff"])
    rows = []
    for weights in simplex(config["weight_step"]):
        scores = score_setting(params, mixed_set(sets, weights), metrics)
        rows.append({"stage": "mix", **dict(zip(CHANNELS, weights, strict=True)), **scores})
    best = best_row(rows, config["objective"], config["tie_breaker"])
    return {channel: best[channel] for channel in CHANNELS}, rows


def tune_rerank(
    mixed: TuningSet, texts: list[dict[str, str]], config: dict[str, Any], metrics: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose the reranking depth and cross-encoder weight, with descriptions as table text.

    Args:
        mixed: Tuning sample with the chosen mix.
        texts: Description text per table id, one mapping per dataset.
        config: Tuning configuration.
        metrics: Metric cutoffs.

    Returns:
        (chosen {top_k, weight}, one row per combination).
    """
    rerank = config["rerank"]
    cross = [
        cross_encoder_scores(
            seeds, instances, {}, rerank["model"], max(rerank["top_k"]),
            rerank["max_length"], rerank["batch_size"], rerank["device"], table_texts,
        )
        for (instances, seeds, _), table_texts in zip(mixed, texts, strict=True)
    ]  # fmt: skip
    params = GraphRAGParams(variant="G0", cutoff=config["cutoff"])
    rows = []
    for top_k, weight in itertools.product(rerank["top_k"], rerank["weight"]):
        reranked = [
            (instances, apply_rerank(seeds, scores, top_k, weight), graph)
            for (instances, seeds, graph), scores in zip(mixed, cross, strict=True)
        ]
        scores = score_setting(params, reranked, metrics)
        rows.append({"stage": "rerank", "top_k": top_k, "weight": weight, **scores})
    best = best_row(rows, config["objective"], config["tie_breaker"])
    return {"top_k": best["top_k"], "weight": best["weight"]}, rows


def main() -> None:
    """Tune the description seeds and add them to configs/seeds.yaml."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tuning_desc_seeds.yaml")
    parser.add_argument("--baseline-config", default="configs/baselines.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    baseline_config = load_config(args.baseline_config)
    set_seed(config["seed"])
    metrics = baseline_config["metrics"]
    sampler = {key: config[key] for key in ("seed", "sample_size", "datasets")}
    sets = [
        sample_training_sets(method, sampler, baseline_config)
        for method in ("bm25", "sxfmr", "desc")
    ]
    desc_only = score_setting(
        GraphRAGParams(variant="G0", cutoff=config["cutoff"]), sets[2], metrics
    )
    logger.info("descriptions alone: TR@5 %.2f, DR@1 %.2f", desc_only["TR@5"], desc_only["DR@1"])
    weights, mix_rows = tune_mix(sets, config, metrics)
    logger.info("mix: best weights %s", weights)

    description_dir = PROJECT_ROOT / load_config("configs/descriptions.yaml")["output_dir"]
    texts = [
        description_texts(
            read_json(PROJECT_ROOT / baseline_config["processed_dir"] / d / "schemas.json"),
            load_descriptions(description_dir / f"{baseline_config['collections'][d]}.jsonl"),
        )
        for d in config["datasets"]
    ]
    mixed = mixed_set(sets, tuple(weights[c] for c in CHANNELS))
    chosen, rerank_rows = tune_rerank(mixed, texts, config, metrics)
    logger.info("rerank: best %s", chosen)

    results = PROJECT_ROOT / config["results"]
    rows = [{"stage": "desc_only", **desc_only}] + mix_rows + rerank_rows
    pd.DataFrame(rows).round(4).to_csv(results / "desc_seeds.csv", index=False)
    output = PROJECT_ROOT / config["output"]
    settings = yaml.safe_load(output.read_text())
    rerank = config["rerank"]
    settings["hybrid_desc"] = {"weights": weights}
    settings["rerank_desc"] = {
        **{key: rerank[key] for key in ("model", "max_length", "batch_size", "device")},
        "pool_size": chosen["top_k"], "top_k": chosen["top_k"], "weight": chosen["weight"],
    }  # fmt: skip
    output.write_text(
        "# Seed settings tuned on training questions (src/tune_seeds.py, §9;"
        " src/tune_desc_seeds.py, §10).\n" + yaml.safe_dump(settings, sort_keys=False)
    )
    logger.info("updated %s", output.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
