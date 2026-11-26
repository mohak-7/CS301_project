"""Unit tests for Phase 3 data preparation (schemas, gold tables, Fiben conversion, hardness)."""

from src.difficulty import spider_hardness
from src.extract_gold import extract_gold_tables
from src.prepare_data import _fiben_sqlite_query
from src.schemas import build_database_schema, fiben_sqlite_ddl

# A small tables.json entry modelled on Spider's concert_singer database.
RAW_DB = {
    "db_id": "concert_singer",
    "table_names_original": ["Singer", "Concert", "Singer_in_concert"],
    "table_names": ["singer", "concert", "singer in concert"],
    "column_names_original": [
        [-1, "*"],
        [0, "Singer_ID"],
        [0, "Name"],
        [1, "Concert_ID"],
        [1, "Year"],
        [2, "Concert_ID"],
        [2, "Singer_ID"],
    ],
    "column_names": [
        [-1, "*"],
        [0, "singer id"],
        [0, "name"],
        [1, "concert id"],
        [1, "year"],
        [2, "concert id"],
        [2, "singer id"],
    ],
    "primary_keys": [1, 3, [5, 6]],
    # Only one key is declared; the other must come from the missing-key patch.
    "foreign_keys": [[5, 3]],
}


def _columns(schema, table):
    return {c["name"]: c for t in schema if t["name"] == table for c in t["columns"]}


def test_declared_and_patched_foreign_keys():
    schema = build_database_schema(RAW_DB, use_wordninja=False)
    junction = _columns(schema, "singer_in_concert")
    assert junction["concert_id"]["foreign_key"] == {"table": "concert", "column": "concert_id"}
    # Patched: same name as singer's primary key (and the name is not "id").
    assert junction["singer_id"]["foreign_key"] == {"table": "singer", "column": "singer_id"}
    # Composite primary keys are flattened.
    assert junction["singer_id"]["primary_key"] is True


def test_patch_skips_columns_named_id():
    raw = {
        "db_id": "toy",
        "table_names_original": ["A", "B"],
        "table_names": ["a", "b"],
        "column_names_original": [[-1, "*"], [0, "id"], [1, "id"]],
        "column_names": [[-1, "*"], [0, "id"], [1, "id"]],
        "primary_keys": [1, 2],
        "foreign_keys": [],
    }
    schema = build_database_schema(raw, use_wordninja=False)
    assert all("foreign_key" not in c for t in schema for c in t["columns"])


def test_wordninja_splits_fiben_names():
    raw = {
        "db_id": "fiben",
        "table_names_original": ["ACCOUNTSPAYABLE"],
        "table_names": ["accountspayable"],
        "column_names_original": [[-1, "*"], [0, "HASNAME"]],
        "column_names": [[-1, "*"], [0, "hasname"]],
        "primary_keys": [],
        "foreign_keys": [],
    }
    table = build_database_schema(raw, use_wordninja=True)[0]
    assert table["name"] == "accountspayable"
    assert table["normalized_name"].lower() == "accounts payable"
    assert table["columns"][0]["normalized_name"].lower() == "has name"


def test_gold_tables_include_subqueries_and_ignore_aliases():
    schema = build_database_schema(RAW_DB, use_wordninja=False)
    sql = (
        "SELECT T1.name FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.singer_id = "
        "T2.singer_id WHERE T2.concert_id IN (SELECT concert_id FROM concert WHERE year = 2014)"
    )
    gold = extract_gold_tables(sql, schema)
    assert list(gold) == ["singer", "concert", "singer_in_concert"]  # database table order
    assert gold["concert"] == ["concert_id", "year"]


def test_fiben_ddl_keeps_last_foreign_key_per_table():
    lines = [
        "CREATE TABLE A (AID BIGINT NOT NULL, BID BIGINT, CID BIGINT, PRIMARY KEY (AID));\n",
        "CREATE TABLE B (BID BIGINT NOT NULL, PRIMARY KEY (BID));\n",
        "ALTER TABLE A ADD CONSTRAINT FK1 FOREIGN KEY (BID) REFERENCES B (BID);\n",
        "ALTER TABLE A ADD CONSTRAINT FK2 FOREIGN KEY (CID) REFERENCES B (BID);",
    ]
    ddl = fiben_sqlite_ddl(lines)
    assert "FOREIGN KEY (CID) REFERENCES B (BID));" in ddl
    assert "FOREIGN KEY (BID)" not in ddl


def test_fiben_query_rewrite():
    sql = 'SELECT x FROM FIBEN."T" ORDER BY x FETCH FIRST 5 ROWS ONLY'
    assert _fiben_sqlite_query(sql) == 'SELECT x FROM "T" ORDER BY x LIMIT 5'


def test_spider_hardness_labels():
    schema = build_database_schema(RAW_DB, use_wordninja=False)
    assert spider_hardness("SELECT count(*) FROM singer", schema) == "easy"
    assert spider_hardness("SELECT nonexistent_column FROM singer", schema) is None
