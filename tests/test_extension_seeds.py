"""Unit tests for the extension seeds: hybrid mixing and cross-encoder reranking rules."""

import numpy as np
import pytest

from src.retrievers.base import ScoreMatrix
from src.retrievers.hybrid import hybrid_scores
from src.retrievers.rerank import apply_rerank

IDS = ["a.t1", "a.t2", "b.t1", "b.t2", "c.t1"]


def _matrix(values):
    return ScoreMatrix(["q"], IDS, np.array([values], dtype=np.float32))


def test_hybrid_scales_both_sides_before_mixing():
    bm25 = _matrix([4.0, np.nan, 2.0, 0.0, np.nan])  # unbounded, NaN = not returned
    dense = _matrix([0.1, 0.5, -0.3, 0.9, 0.1])  # cosine similarities
    mixed = hybrid_scores(bm25, dense, weight=0.5).scores[0]
    expected = 0.5 * np.array([1.0, 0.0, 0.5, 0.0, 0.0]) + 0.5 * np.array(
        [1 / 3, 2 / 3, 0, 1, 1 / 3]
    )
    assert mixed == pytest.approx(expected, abs=1e-6)


def test_hybrid_rejects_mismatched_matrices():
    other = ScoreMatrix(["other"], IDS, np.zeros((1, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        hybrid_scores(_matrix([1, 2, 3, 4, 5]), other, weight=0.5)


def test_rerank_reorders_top_k_but_keeps_them_above_the_rest():
    seeds = _matrix([1.0, 0.9, 0.8, 0.2, 0.0])
    cross = _matrix([-5.0, 3.0, 1.0, np.nan, np.nan])  # cross-encoder prefers a.t2
    reranked = apply_rerank(seeds, cross, top_k=3, weight=1.0).scores[0]
    assert reranked[1] > reranked[2] > reranked[0]  # new order inside the top 3
    assert min(reranked[:3]) >= max(reranked[3:])  # outside tables stay below
    assert reranked[3:] == pytest.approx(seeds.scores[0][3:])


def test_rerank_with_zero_weight_keeps_seed_order():
    seeds = _matrix([1.0, 0.9, 0.8, 0.2, 0.0])
    cross = _matrix([-5.0, 3.0, 1.0, np.nan, np.nan])
    reranked = apply_rerank(seeds, cross, top_k=3, weight=0.0).scores[0]
    assert list(np.argsort(-reranked)) == [0, 1, 2, 3, 4]
