"""Unit tests for the description extension: prompts, link index and indexed texts."""

from src.describe_tables import build_messages, referenced_by_index, table_prompt
from src.retrievers.descriptions import description_texts
from tests.test_phase5_graphrag import SCHEMAS


def test_prompt_lists_keys_and_links():
    tables = SCHEMAS["cs"]
    junction = next(t for t in tables if t["name"] == "singer_in_concert")
    prompt = table_prompt("cs", junction, [], max_columns=40)
    assert "Table: singer_in_concert (in words: singer_in_concert)" in prompt
    assert "singer_id -> singer.singer_id" in prompt
    assert "Referenced by: (none)" in prompt


def test_referenced_by_index_inverts_foreign_keys():
    index = referenced_by_index(SCHEMAS["cs"])
    assert index["singer"] == ["singer_in_concert.singer_id"]
    assert "singer_in_concert" not in index


def test_long_column_lists_are_truncated():
    table = {
        "name": "wide",
        "normalized_name": "wide",
        "columns": [{"name": f"c{i}", "normalized_name": f"c {i}"} for i in range(50)],
    }
    assert "and 10 more columns" in table_prompt("db", table, [], max_columns=40)


def test_messages_have_system_examples_and_query():
    messages = build_messages("Database: x")
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "Database: x"}
    assert [m["role"] for m in messages[1:-1]] == ["user", "assistant"] * 5


def test_description_texts_prefix_table_name():
    descriptions = {f"cs.{t['name']}": "A description." for t in SCHEMAS["cs"]}
    descriptions["other.lonely"] = "Another."
    texts = description_texts(SCHEMAS, descriptions)
    assert texts["cs.singer"] == "singer: A description."
    assert list(texts) == ["cs.singer", "cs.concert", "cs.singer_in_concert", "other.lonely"]


def test_params_for_g0_without_tuned_settings() -> None:
    """G0 runs for a seed method that has no tuned graph settings (the §10 `desc` seeds)."""
    from src.run_graphrag import params_for

    config = {"bm25": {"G1": {"cutoff": 100, "beta": 0.3}}}
    assert params_for(config, "desc", "G0").cutoff == 100
    assert params_for(config, "bm25", "G1").beta == 0.3
