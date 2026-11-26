"""Unit tests for Phase 4: table documents, ranking rule and metrics."""

import numpy as np
import pytest
from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate

from src.evaluate import average_precision, question_metrics, summarise
from src.rank_databases import rank_databases, top_tables
from src.retrievers.base import table_documents

KS = {"database_ks": [1, 5], "table_ks": [1, 2], "full_schema_ks": [2]}


def test_table_documents_include_database_id():
    schemas = {
        "concert_singer": [
            {
                "name": "singer",
                "normalized_name": "singer",
                "columns": [{"name": "name", "normalized_name": "name"}],
            }
        ]
    }
    assert table_documents(schemas) == [
        {"id": "concert_singer.singer", "text": "concert_singer singer name"}
    ]


def test_top_tables_skips_unretrieved_and_keeps_tie_order():
    scores = np.array([0.5, np.nan, 0.9, 0.5], dtype=np.float32)
    ids = ["a.t1", "a.t2", "b.t1", "b.t2"]
    ranked = top_tables(scores, ids, cutoff=10)
    assert [identifier for identifier, _ in ranked] == ["b.t1", "a.t1", "b.t2"]
    assert [score for _, score in ranked] == pytest.approx([0.9, 0.5, 0.5])
    assert len(top_tables(scores, ids, cutoff=2)) == 2


def test_rank_databases_uses_mean_score():
    # Database a: one table at 0.9. Database b: 0.8 and 0.2, mean 0.5. Mean, not max or sum.
    ranked = [("a.t1", 0.9), ("b.t1", 0.8), ("b.t2", 0.2)]
    assert rank_databases(ranked) == [("a", 0.9), ("b", 0.5)]


def _instance(gold_tables):
    return {"id": "q", "gold_db": "db", "gold_tables": gold_tables}


def test_per_question_and_pooled_recall_differ():
    # Question 1: 1 gold table, found at rank 1. Question 2: 3 gold tables, 1 found in top 2.
    first = question_metrics(_instance(["x"]), [("db", 1.0)], [("db.x", 1.0)], **KS)
    second = question_metrics(
        _instance(["x", "y", "z"]),
        [("db", 1.0)],
        [("db.x", 1.0), ("db.w", 0.5), ("db.y", 0.1)],
        **KS,
    )
    summary = summarise([first, second], table_ks=[2])
    assert summary["TR@2"] == 100 * (1 + 1 / 3) / 2  # per-question mean: 66.67
    assert summary["pooled TR@2"] == 100 * 2 / 4  # all hits over all gold tables: 50.0
    assert summary["FS@2"] == 50.0
    assert summary["DR@1"] == 100.0


def test_average_precision_matches_ranx_map():
    ranked = ["d1", "d2", "d3", "d4"]
    gold = {"d2", "d4", "d9"}
    run = Run({"q": {doc: 1.0 - i / 10 for i, doc in enumerate(ranked)}})
    expected = ranx_evaluate(Qrels({"q": {doc: 1 for doc in gold}}), run, "map")
    assert np.isclose(average_precision(ranked, gold), expected)
