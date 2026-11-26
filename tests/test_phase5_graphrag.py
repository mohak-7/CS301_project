"""Unit tests for Phase 5: table graph, GraphRAG variants, safeguards and ranking modes."""

import networkx as nx
import numpy as np
import pytest

from src.graph import schema_graph
from src.rank_databases import hierarchical_tables, rank_databases, top_tables
from src.retrievers.base import ScoreMatrix
from src.retrievers.graphrag import (
    GraphRAGParams,
    complete_connectors,
    graphrag_scores,
    min_max_scale,
    neighbour_expansion,
    neighbour_lists,
    personalized_pagerank,
    restart_vectors,
)


def _table(name, foreign_keys=()):
    columns = [{"name": f"{name}_id", "normalized_name": name, "primary_key": True}]
    for column, (ref_table, ref_column) in foreign_keys:
        columns.append(
            {
                "name": column,
                "normalized_name": column,
                "foreign_key": {"table": ref_table, "column": ref_column},
            }
        )
    return {"name": name, "normalized_name": name, "columns": columns}


# concert_singer-like database plus a second database with an isolated table.
SCHEMAS = {
    "cs": [
        _table("singer"),
        _table("concert"),
        _table(
            "singer_in_concert",
            [("singer_id", ("singer", "singer_id")), ("concert_id", ("concert", "concert_id"))],
        ),
    ],
    "other": [_table("lonely")],
}
IDS = ["cs.singer", "cs.concert", "cs.singer_in_concert", "other.lonely"]

# Diamond: top connects to bottom through two equally short paths (left and right).
DIAMOND = {
    "d": [
        _table("top"),
        _table("bottom"),
        _table("left", [("top_id", ("top", "top_id")), ("bottom_id", ("bottom", "bottom_id"))]),
        _table("right", [("top_id", ("top", "top_id")), ("bottom_id", ("bottom", "bottom_id"))]),
    ]
}
DIAMOND_IDS = ["d.top", "d.bottom", "d.left", "d.right"]


def test_graph_links_junction_table_and_keeps_isolated_tables():
    graph = schema_graph(SCHEMAS)
    assert set(graph.nodes) == set(IDS)
    assert graph.has_edge("cs.singer", "cs.singer_in_concert")
    assert graph.has_edge("cs.concert", "cs.singer_in_concert")
    assert not graph.has_edge("cs.singer", "cs.concert")
    assert graph.degree("other.lonely") == 0


def test_min_max_scale_guards_all_zero_rows():
    scores = np.array([[np.nan, np.nan, np.nan, np.nan], [2.0, np.nan, 4.0, 0.0]])
    scaled = min_max_scale(scores)
    assert not np.isnan(scaled).any()
    assert scaled[0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert scaled[1].tolist() == [0.5, 0.0, 1.0, 0.0]


def test_neighbour_expansion_handles_tables_without_neighbours():
    seeds = np.array([[1.0, 0.0, 0.0, 0.4]])
    expanded = neighbour_expansion(seeds, neighbour_lists(schema_graph(SCHEMAS), IDS), beta=0.5)
    assert expanded[0].tolist() == pytest.approx([1.0, 0.0, 0.5, 0.4])


def test_pagerank_matches_networkx_and_skips_zero_restart():
    graph = schema_graph(SCHEMAS)
    seeds = np.array([[0.9, 0.1, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0]])
    restart = restart_vectors(seeds, num_seeds=3)
    ranks = personalized_pagerank(graph, IDS, restart, 0.85, 200, 1e-12)
    personalization = dict(zip(IDS, restart[0], strict=True))
    # Same tight tolerance as ours; networkx's default (1e-6) stops early.
    expected = nx.pagerank(
        graph, alpha=0.85, personalization=personalization, max_iter=1000, tol=1e-14
    )
    assert ranks[0] == pytest.approx([expected[t] for t in IDS], abs=1e-8)
    assert ranks[1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_connectors_score_every_tied_path_below_their_endpoints():
    graph = schema_graph(DIAMOND)
    seeds = np.array([1.0, 0.8, 0.0, 0.0])
    params = GraphRAGParams(
        variant="G3", connector_databases=1, connector_seeds=2, seed_threshold=0.5, gamma=0.9
    )
    completed = complete_connectors(seeds.copy(), seeds, graph, DIAMOND_IDS, params)
    assert completed[2] == completed[3] == pytest.approx(0.9 * 0.8)
    assert completed[2] < min(completed[0], completed[1])


@pytest.mark.parametrize("variant", ["G0", "G1", "G2", "G3"])
def test_all_variants_survive_an_all_zero_query(variant):
    seeds = ScoreMatrix(["q"], IDS, np.full((1, 4), np.nan, dtype=np.float32))
    final = graphrag_scores(seeds, schema_graph(SCHEMAS), GraphRAGParams(variant=variant))
    assert np.isnan(final.scores).all()  # nothing retrieved, no error and no padding (D14)


def test_g0_reproduces_baseline_ranking():
    # Means: cs = (3 + 2) / 2 = 2.5, other = 2. No ties, so float rounding cannot reorder.
    raw = np.array([[3.0, np.nan, 2.0, 2.0]], dtype=np.float32)
    final = graphrag_scores(ScoreMatrix(["q"], IDS, raw), schema_graph(SCHEMAS), GraphRAGParams())
    baseline_tables = top_tables(raw[0], IDS)
    graphrag_tables = top_tables(final.scores[0], IDS)
    assert [t for t, _ in graphrag_tables] == [t for t, _ in baseline_tables]
    assert [d for d, _ in rank_databases(graphrag_tables)] == [
        d for d, _ in rank_databases(baseline_tables)
    ]


def test_ranking_modes():
    ranked = [("a.t1", 0.9), ("b.t1", 0.8), ("a.t2", 0.1), ("b.t2", 0.7)]
    assert rank_databases(ranked)[0][0] == "b"  # means: a = 0.5, b = 0.75
    assert rank_databases(ranked, top_per_database=1)[0][0] == "a"  # best table only
    grouped = hierarchical_tables(ranked, rank_databases(ranked), num_databases=1)
    assert grouped == [("b.t1", 0.8), ("b.t2", 0.7)]
