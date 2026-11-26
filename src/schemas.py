"""Convert raw schema files into the authors' `schemas.json` structure.

Implements Phase 3 schema preparation (implementation_plan.md, §2.2, §3.3, D12). Spider and Bird
follow `get_dataset_schemas` in the authors' scripts/prepare_data.py; Fiben first follows UNITE's
scripts/prepare_fiben.py and schema_generator.py, which produced the authors' Fiben input.
"""

import sqlite3
from pathlib import Path
from typing import Any

import wordninja

# {db_id: [table, ...]}. Each table is {"name", "normalized_name", "columns"}, and each column is
# {"name", "normalized_name"} plus optional "primary_key" and "foreign_key" entries.
SchemaCollection = dict[str, list[dict[str, Any]]]


def _flatten_primary_keys(primary_keys: list[int | list[int]]) -> list[int]:
    """Flatten primary-key column indices; composite keys are stored as nested lists.

    Args:
        primary_keys: The `primary_keys` field of a tables.json entry.

    Returns:
        Flat list of column indices that belong to a primary key.
    """
    flat: list[int] = []
    for key in primary_keys:
        flat.extend(key if isinstance(key, list) else [key])
    return flat


def _column_foreign_key(
    raw_db: dict[str, Any], column_index: int, primary_keys: list[int]
) -> dict[str, str] | None:
    """Return the foreign key of one column, including the authors' missing-key patch (§2.2).

    Args:
        raw_db: tables.json entry with lower-cased original names.
        column_index: Index of the column in `column_names_original`.
        primary_keys: Flat list of primary-key column indices.

    Returns:
        {"table", "column"} of the referenced column, or None if the column has no foreign key.
    """
    tables = raw_db["table_names_original"]
    columns = raw_db["column_names_original"]
    foreign_key = None
    # As in the authors' loop, a later declared key overwrites an earlier one.
    for left, right in raw_db["foreign_keys"]:
        if left == column_index:
            foreign_key = {"table": tables[columns[right][0]], "column": columns[right][1]}
    if foreign_key is not None:
        return foreign_key
    # Patch for keys missing from the source files: a column with the same name as another
    # table's primary key is treated as a reference to it. `id` is excluded because almost
    # every table has one and the name carries no link.
    for key_index in primary_keys:
        key_name = columns[key_index][1]
        if key_index != column_index and key_name != "id" and key_name == columns[column_index][1]:
            return {"table": tables[columns[key_index][0]], "column": key_name}
    return None


def _normalise_names(raw_db: dict[str, Any], use_wordninja: bool) -> dict[str, Any]:
    """Lower-case original names and, for Fiben, split them into words (§2.2).

    Args:
        raw_db: One tables.json entry (not modified).
        use_wordninja: Split concatenated names into words. Only Fiben needs this: its names
            are run-together upper-case strings such as ACCOUNTSPAYABLEDETAILS.

    Returns:
        A copy of the entry with normalised name fields.
    """
    db = {**raw_db}
    db["table_names_original"] = [name.lower() for name in raw_db["table_names_original"]]
    db["column_names_original"] = [
        [table_index, name.lower()] for table_index, name in raw_db["column_names_original"]
    ]
    if use_wordninja:
        # The authors split the original-case string, so this is done before lower-casing.
        db["table_names"] = [" ".join(wordninja.split(n)) for n in raw_db["table_names_original"]]
        db["column_names"] = [
            [table_index, " ".join(wordninja.split(name))]
            for table_index, name in raw_db["column_names_original"]
        ]
    return db


def build_database_schema(raw_db: dict[str, Any], use_wordninja: bool) -> list[dict[str, Any]]:
    """Convert one tables.json entry into the authors' per-database table list.

    Args:
        raw_db: One tables.json entry.
        use_wordninja: Whether to split names into words (Fiben only).

    Returns:
        List of tables, each with name, normalised name and columns.
    """
    db = _normalise_names(raw_db, use_wordninja)
    primary_keys = _flatten_primary_keys(db["primary_keys"])
    tables = []
    for table_index, table_name in enumerate(db["table_names_original"]):
        columns = []
        for column_index, (owner, column_name) in enumerate(db["column_names_original"]):
            if owner != table_index:
                continue
            column: dict[str, Any] = {
                "name": column_name,
                "normalized_name": db["column_names"][column_index][1],
            }
            if column_index in primary_keys:
                column["primary_key"] = True
            foreign_key = _column_foreign_key(db, column_index, primary_keys)
            if foreign_key is not None:
                column["foreign_key"] = foreign_key
            columns.append(column)
        tables.append(
            {
                "name": table_name,
                "normalized_name": db["table_names"][table_index],
                "columns": columns,
            }
        )
    return tables


def build_schemas(raw_dbs: list[dict[str, Any]], use_wordninja: bool = False) -> SchemaCollection:
    """Convert a list of tables.json entries into a schema collection.

    Args:
        raw_dbs: tables.json entries (one per database).
        use_wordninja: Whether to split names into words (Fiben only).

    Returns:
        Mapping from db_id to its table list.
    """
    return {raw_db["db_id"]: build_database_schema(raw_db, use_wordninja) for raw_db in raw_dbs}


def fiben_sqlite_ddl(ddl_lines: list[str]) -> str:
    """Rewrite FIBEN.sql into SQLite DDL exactly as UNITE's prepare_fiben.py does.

    Foreign keys are declared in separate ALTER TABLE statements, which SQLite does not support,
    so they are moved into the CREATE TABLE statements. UNITE keeps them in a dictionary keyed
    by table name, so only the last ALTER statement per table survives. We keep this behaviour
    because the authors' Fiben schema was built from UNITE's output.

    Args:
        ddl_lines: Lines of FIBEN.sql.

    Returns:
        SQLite-compatible DDL script.
    """
    last_foreign_key: dict[str, str] = {}
    for line in ddl_lines:
        if line.startswith("ALTER"):
            table_name = line.split("ALTER TABLE ")[1].split()[0]
            statement = line.rstrip().rstrip(";")
            last_foreign_key[table_name] = statement[statement.index("FOREIGN KEY ") :]
    statements = []
    for line in ddl_lines:
        if line.startswith("CREATE"):
            table_name = line.split("CREATE TABLE ")[1].split()[0]
            body = line.rstrip()
            foreign_key = last_foreign_key.get(table_name)
            if foreign_key:
                # Insert before the closing ");". SQLite accepts table constraints without a
                # separating comma, which UNITE relies on.
                body = body[:-2] + f" {foreign_key}" + body[-2:]
            statements.append(body)
    return "\n".join(statements)


def _foreign_key_indices(db: dict[str, Any], raw_keys: list[tuple]) -> list[list[int]]:
    """Map (table, column) foreign-key pairs to column indices, as UNITE's convert_fk_index.

    Args:
        db: tables.json-style entry with original names.
        raw_keys: List of ((table, column), (ref_table, ref_column)) pairs.

    Returns:
        List of [column_index, referenced_column_index] pairs.
    """
    pairs = []
    for (table, column), (ref_table, ref_column) in raw_keys:
        table_id = db["table_names_original"].index(table)
        ref_table_id = db["table_names_original"].index(ref_table)
        column_id = ref_column_id = None
        for index, (owner, name) in enumerate(db["column_names_original"]):
            if owner == ref_table_id and name == ref_column:
                ref_column_id = index
            elif owner == table_id and name == column:
                column_id = index
        if column_id and ref_column_id:
            pairs.append([column_id, ref_column_id])
    return pairs


def fiben_tables_json(ddl_path: Path, db_id: str = "fiben") -> list[dict[str, Any]]:
    """Build Fiben's tables.json entry from FIBEN.sql (UNITE's dump_db_json_schema).

    The DDL is loaded into an in-memory SQLite database and read back with PRAGMA calls, so the
    table order, primary keys and foreign keys match UNITE's output. No cell data is needed.

    Args:
        ddl_path: Path to FIBEN.sql.
        db_id: Database identifier (UNITE uses the file name, "fiben").

    Returns:
        A one-element list holding the tables.json entry.
    """
    connection = sqlite3.connect(":memory:")
    connection.executescript(fiben_sqlite_ddl(ddl_path.read_text().splitlines(keepends=True)))
    db: dict[str, Any] = {
        "db_id": db_id,
        "table_names_original": [],
        "table_names": [],
        "column_names_original": [[-1, "*"]],
        "column_names": [[-1, "*"]],
        "primary_keys": [],
    }
    raw_keys = []
    table_rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for table_index, (table_name,) in enumerate(table_rows.fetchall()):
        db["table_names_original"].append(table_name)
        db["table_names"].append(table_name.lower().replace("_", " "))
        for row in connection.execute(f"PRAGMA foreign_key_list('{table_name}')"):
            raw_keys.append(((table_name, row[3]), (row[2], row[4])))
        for row in connection.execute(f"PRAGMA table_info('{table_name}')"):
            db["column_names_original"].append([table_index, row[1]])
            db["column_names"].append([table_index, row[1].lower().replace("_", " ")])
            # UNITE counts only the first column of a composite key (pk position == 1).
            if row[5] == 1:
                db["primary_keys"].append(len(db["column_names"]) - 1)
    db["foreign_keys"] = _foreign_key_indices(db, raw_keys)
    connection.close()
    return [db]
