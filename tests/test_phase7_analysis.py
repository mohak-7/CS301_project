"""Unit tests for Phase 7: bridge tables, join-only tables and the paired test."""

from src.analysis.joins import bridge_tables, join_only_tables
from src.analysis.significance import paired_test
from src.graph import schema_graph
from tests.test_phase5_graphrag import SCHEMAS

DATABASE = SCHEMAS["cs"]


def test_bridge_table_is_the_articulation_point():
    graph = schema_graph(SCHEMAS)
    gold = ["cs.singer", "cs.concert", "cs.singer_in_concert"]
    assert bridge_tables(gold, graph) == {"cs.singer_in_concert"}
    assert bridge_tables(gold[:2], graph) == set()  # fewer than 3 gold tables


def test_join_only_table_with_aliases():
    # The paper's Example 2: the junction table appears only in ON conditions.
    sql = (
        "SELECT s.singer_id FROM singer_in_concert AS sc "
        "JOIN singer AS s ON sc.singer_id = s.singer_id "
        "JOIN concert AS c ON sc.concert_id = c.concert_id WHERE c.concert_id = 2014"
    )
    assert join_only_tables(sql, DATABASE) == {"singer_in_concert"}


def test_double_quoted_values_are_read_as_strings():
    # Spider writes string values in double quotes; they must not be read as columns.
    sql = 'SELECT count(*) FROM singer WHERE singer_id = "Kyle"'
    assert join_only_tables(sql, DATABASE) == set()


def test_unresolvable_query_is_undetermined():
    assert join_only_tables("SELECT missing_column FROM singer", DATABASE) is None


def test_paired_test_with_identical_scores():
    records = {"q": {"metrics": {"TR@5": 1.0}}}
    assert paired_test(records, records, "TR@5") == (0.0, 1.0)
