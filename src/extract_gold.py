"""Extract the gold tables (and their columns) referenced by a gold SQL query.

Implements gold-table extraction (implementation_plan.md, §2.2 rule 3, §3.4), following
`extract_metadata` in the authors' scripts/prepare_data.py.
"""

from typing import Any

from sqlglot import exp, parse_one


def extract_gold_tables(sql: str, database: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return the tables of `database` referenced in `sql`, with the columns used from each.

    Every table reference is collected, including those in subqueries. Only names that exist in
    the gold database are kept, so aliases and CTE names never become gold tables.

    Args:
        sql: Gold SQL query.
        database: The gold database's table list (schemas.json structure).

    Returns:
        Ordered mapping from table name to the referenced column names, in the database's table
        order (the authors' order).

    Raises:
        sqlglot.errors.ParseError: If the query cannot be parsed. Callers drop such questions
            for Bird and Fiben, as the authors did (§2.2 rule 4).
    """
    parsed = parse_one(sql, read="sqlite")
    referenced_tables = {table.name.lower() for table in parsed.find_all(exp.Table)}
    referenced_columns = {column.name.lower() for column in parsed.find_all(exp.Column)}
    return {
        table["name"]: [c["name"] for c in table["columns"] if c["name"] in referenced_columns]
        for table in database
        if table["name"] in referenced_tables
    }
