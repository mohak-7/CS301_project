"""Cross-encoder reranking of each question's top seed tables.

Implements the reranking extension (implementation_plan.md, §9). A cross-encoder reads the
question and a table's text together, so it can match wording that bag-of-words and embedding
similarity miss. It is pretrained on passage ranking and is not trained on our schemas.
"""

from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from src.retrievers.base import ScoreMatrix, table_documents
from src.retrievers.graphrag import min_max_scale
from src.schemas import SchemaCollection


def resolve_device(device: str) -> str:
    """Return "mps" (Apple GPU) or "cpu" for device "auto"; other values are passed through."""
    if device != "auto":
        return device
    return "mps" if torch.backends.mps.is_available() else "cpu"


def cross_encoder_scores(
    seeds: ScoreMatrix,
    instances: list[dict[str, Any]],
    schemas: SchemaCollection,
    model_name: str,
    pool_size: int,
    max_length: int,
    batch_size: int,
    device: str = "auto",
    table_texts: dict[str, str] | None = None,
) -> ScoreMatrix:
    """Score each question against its `pool_size` best seed tables with a cross-encoder.

    Args:
        seeds: Seed score matrix (rows in the same order as `instances`).
        instances: Processed instances (for the question text).
        schemas: Schema collection (for the table text, as in §2.4).
        model_name: Hugging Face id of the cross-encoder.
        pool_size: Number of top seed tables scored per question.
        max_length: Maximum tokens per (question, table) pair.
        batch_size: Prediction batch size.
        device: "auto", "mps" or "cpu".
        table_texts: Text shown for each table; default: the schema text of §2.4. The
            description extension (§10) passes table descriptions instead.

    Returns:
        ScoreMatrix of raw cross-encoder scores; NaN for tables outside each question's pool.
    """
    text = table_texts or {
        document["id"]: document["text"] for document in table_documents(schemas)
    }
    order = np.argsort(-np.nan_to_num(seeds.scores, nan=-np.inf), axis=1, kind="stable")
    pools = order[:, :pool_size]
    pairs = [
        (instance["question"], text[seeds.table_ids[column]])
        for instance, row in zip(instances, pools, strict=True)
        for column in row
    ]
    model = CrossEncoder(model_name, device=resolve_device(device), max_length=max_length)
    predicted = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    scores = np.full(seeds.scores.shape, np.nan, dtype=np.float32)
    rows = np.repeat(np.arange(len(instances)), pools.shape[1])
    scores[rows, pools.ravel()] = np.asarray(predicted, dtype=np.float32)
    return ScoreMatrix(seeds.question_ids, seeds.table_ids, scores)


def apply_rerank(
    seeds: ScoreMatrix, cross_scores: ScoreMatrix, top_k: int, weight: float
) -> ScoreMatrix:
    """Reorder each question's top-k seed tables by a mix of seed and cross-encoder scores.

    The top-k tables keep the score interval they had (from the k-th best seed score up to the
    best), so reranking changes their order but never drops them below tables outside the top k.
    Inside that interval, a table's position is weight * scaled cross-encoder score +
    (1 - weight) * its scaled seed score, both scaled to [0, 1] over the top k.

    Args:
        seeds: Seed score matrix.
        cross_scores: Output of `cross_encoder_scores` (pool of at least `top_k` tables).
        top_k: Number of top seed tables reranked.
        weight: Weight of the cross-encoder; 0 keeps the seed order.

    Returns:
        Reranked ScoreMatrix in [0, 1].
    """
    base = min_max_scale(seeds.scores)
    reranked = base.copy()
    order = np.argsort(-base, axis=1, kind="stable")[:, :top_k]
    for row, columns in enumerate(order):
        seed_part = base[row, columns]
        cross_part = cross_scores.scores[row, columns].astype(np.float64)
        low, high = seed_part.min(), seed_part.max()
        # Scale both parts to [0, 1] within the top k; a flat part (all equal) contributes 0.
        mix = weight * _unit(cross_part) + (1 - weight) * _unit(seed_part)
        reranked[row, columns] = low + (high - low) * mix
    return ScoreMatrix(seeds.question_ids, seeds.table_ids, reranked.astype(np.float32))


def _unit(values: np.ndarray) -> np.ndarray:
    """Scale a vector to [0, 1]; a constant vector becomes all zeros (no division by zero)."""
    spread = values.max() - values.min()
    return (values - values.min()) / spread if spread > 0 else np.zeros_like(values)
