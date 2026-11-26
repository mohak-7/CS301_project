"""Hybrid seed scores: a weighted mix of scaled BM25 and dense (SXFMR) scores.

Implements the hybrid-seed extension (implementation_plan.md, §5.6 and §9). BM25 is strong when
questions reuse schema words (Spider); dense retrieval is strong under paraphrasing (Spider-syn),
so a mix may keep the strengths of both.
"""

from src.retrievers.base import ScoreMatrix
from src.retrievers.graphrag import min_max_scale


def hybrid_scores(bm25: ScoreMatrix, dense: ScoreMatrix, weight: float) -> ScoreMatrix:
    """Mix BM25 and dense scores after scaling each to [0, 1] per question.

    Scaling first puts both on the same range: BM25 scores are unbounded, cosine similarities
    lie in [-1, 1]. Tables BM25 did not return count as 0 on the BM25 side; the dense side
    scores every table, so every table receives a hybrid score.

    Args:
        bm25: BM25 score matrix.
        dense: Dense score matrix with the same rows and columns.
        weight: Weight of BM25; the dense side gets 1 - weight.

    Returns:
        Hybrid ScoreMatrix in [0, 1].

    Raises:
        ValueError: If the two matrices do not cover the same questions and tables.
    """
    if bm25.question_ids != dense.question_ids or bm25.table_ids != dense.table_ids:
        raise ValueError("BM25 and dense matrices must have the same rows and columns")
    mixed = weight * min_max_scale(bm25.scores) + (1 - weight) * min_max_scale(dense.scores)
    return ScoreMatrix(bm25.question_ids, bm25.table_ids, mixed.astype("float32"))


def weighted_mix(matrices: list[ScoreMatrix], weights: list[float]) -> ScoreMatrix:
    """Mix several score matrices after scaling each to [0, 1] per question (§10).

    Args:
        matrices: Score matrices with the same rows and columns.
        weights: One weight per matrix (normally summing to 1).

    Returns:
        Mixed ScoreMatrix.

    Raises:
        ValueError: If the matrices do not cover the same questions and tables.
    """
    first = matrices[0]
    for other in matrices[1:]:
        if other.question_ids != first.question_ids or other.table_ids != first.table_ids:
            raise ValueError("all matrices must have the same rows and columns")
    mixed = sum(w * min_max_scale(m.scores) for m, w in zip(matrices, weights, strict=True))
    return ScoreMatrix(first.question_ids, first.table_ids, mixed.astype("float32"))
