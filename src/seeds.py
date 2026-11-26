"""Seed score matrices for every seed method: BM25, SXFMR, hybrid and hybrid + reranking.

Implements the seed side of the extensions (implementation_plan.md, §9 and §10). BM25 and SXFMR
come from the Phase 4 cache; "hybrid" mixes them (src/retrievers/hybrid.py); "hybrid_rr" reranks
the hybrid's top tables with a cross-encoder (src/retrievers/rerank.py). "desc" matches questions
to LLM-written table descriptions (src/retrievers/descriptions.py); "hybrid_desc" mixes BM25,
SXFMR and "desc"; "hybrid_desc_rr" reranks it with the cross-encoder reading the descriptions.
Tuned settings are read from configs/seeds.yaml (src/tune_seeds.py, src/tune_desc_seeds.py),
encoder settings for descriptions from configs/descriptions.yaml.
"""

import logging
import time
from pathlib import Path
from typing import Any

from src.common import PROJECT_ROOT, load_config, read_json, read_jsonl
from src.retrievers.base import ScoreMatrix
from src.retrievers.descriptions import description_scores, description_texts, load_descriptions
from src.retrievers.hybrid import hybrid_scores, weighted_mix
from src.retrievers.rerank import apply_rerank, cross_encoder_scores
from src.run_baselines import compute_scores

logger = logging.getLogger(__name__)
SEED_METHODS = ("bm25", "sxfmr", "hybrid", "hybrid_rr", "desc", "hybrid_desc", "hybrid_desc_rr")
DESCRIPTION_CONFIG = "configs/descriptions.yaml"


def cached_cross_scores(
    hybrid: ScoreMatrix,
    dataset: str,
    split: str,
    rerank: dict[str, Any],
    cache_tag: str,
    baseline_config: dict[str, Any],
    table_texts: dict[str, str] | None = None,
) -> ScoreMatrix:
    """Return cross-encoder scores for the hybrid's top tables, computing them once.

    The cache file name records the seed settings (`cache_tag`) and pool size, because both
    decide which tables are scored.

    Args:
        hybrid: Hybrid seed matrix of the dataset split.
        dataset: Dataset name.
        split: "test" or "train".
        rerank: Rerank settings (model, pool_size, max_length, batch_size, device).
        cache_tag: Identifies the seed settings in the cache file name.
        baseline_config: configs/baselines.yaml.
        table_texts: Text shown to the cross-encoder per table (default: schema text).

    Returns:
        Cross-encoder ScoreMatrix (NaN outside each question's pool).
    """
    name = f"{dataset}_{split}_{cache_tag}_pool{rerank['pool_size']}.npz"
    path = PROJECT_ROOT / baseline_config["scores_dir"] / "cross_encoder" / name
    if path.exists():
        return ScoreMatrix.load(path)
    processed = PROJECT_ROOT / baseline_config["processed_dir"] / dataset
    instances = read_jsonl(processed / f"{split}.jsonl")
    started = time.perf_counter()
    scores = cross_encoder_scores(
        hybrid,
        instances,
        read_json(processed / "schemas.json"),
        rerank["model"],
        rerank["pool_size"],
        rerank["max_length"],
        rerank["batch_size"],
        rerank["device"],
        table_texts,
    )
    logger.info("cross-encoder %s/%s: %.1f s", dataset, split, time.perf_counter() - started)
    scores.save(path)
    return scores


def description_matrix(
    dataset: str, split: str, baseline_config: dict[str, Any]
) -> tuple[ScoreMatrix, dict[str, str]]:
    """Return the description-similarity matrix of a dataset split, and the description texts.

    Args:
        dataset: Dataset name.
        split: "test" or "train".
        baseline_config: configs/baselines.yaml.

    Returns:
        (ScoreMatrix, description text per table id).
    """
    settings = load_config(DESCRIPTION_CONFIG)
    processed = PROJECT_ROOT / baseline_config["processed_dir"] / dataset
    collection = baseline_config["collections"][dataset]
    descriptions = load_descriptions(PROJECT_ROOT / settings["output_dir"] / f"{collection}.jsonl")
    texts = description_texts(read_json(processed / "schemas.json"), descriptions)
    file_name = f"{dataset}.npz" if split == "test" else f"{dataset}_{split}.npz"
    path = PROJECT_ROOT / baseline_config["scores_dir"] / "desc" / file_name
    if path.exists():
        return ScoreMatrix.load(path), texts
    started = time.perf_counter()
    matrix = description_scores(
        read_jsonl(processed / f"{split}.jsonl"), texts, settings["retrieval"]
    )
    logger.info("descriptions %s/%s: %.1f s", dataset, split, time.perf_counter() - started)
    matrix.save(path)
    return matrix, texts


def seed_scores(
    method: str,
    dataset: str,
    split: str,
    baseline_config: dict[str, Any],
    seeds_config: Path | str = "configs/seeds.yaml",
) -> ScoreMatrix:
    """Return the seed score matrix of a seed method on a dataset split.

    Args:
        method: One of SEED_METHODS.
        dataset: Dataset name.
        split: "test" or "train".
        baseline_config: configs/baselines.yaml.
        seeds_config: Tuned hybrid and rerank settings.

    Returns:
        ScoreMatrix over every question of the split and every table of the collection.
    """
    if method in ("bm25", "sxfmr"):
        return compute_scores(method, dataset, baseline_config, split=split)
    bm25 = compute_scores("bm25", dataset, baseline_config, split=split)
    dense = compute_scores("sxfmr", dataset, baseline_config, split=split)
    if method == "desc":
        return description_matrix(dataset, split, baseline_config)[0]
    settings = load_config(seeds_config)
    if method in ("hybrid", "hybrid_rr"):
        weight = settings["hybrid"]["weight"]
        hybrid = hybrid_scores(bm25, dense, weight)
        if method == "hybrid":
            return hybrid
        rerank = settings["rerank"]
        cross = cached_cross_scores(hybrid, dataset, split, rerank, f"w{weight}", baseline_config)
        return apply_rerank(hybrid, cross, rerank["top_k"], rerank["weight"])
    descriptions, texts = description_matrix(dataset, split, baseline_config)
    weights = settings["hybrid_desc"]["weights"]
    mixed = weighted_mix(
        [bm25, dense, descriptions], [weights["bm25"], weights["schema"], weights["desc"]]
    )
    if method == "hybrid_desc":
        return mixed
    rerank = settings["rerank_desc"]
    tag = "desc_" + "_".join(str(weights[k]) for k in ("bm25", "schema", "desc"))
    cross = cached_cross_scores(mixed, dataset, split, rerank, tag, baseline_config, texts)
    return apply_rerank(mixed, cross, rerank["top_k"], rerank["weight"])
