"""GQ2: does explicit graph structure preserve multi-table join relationships?

Implements implementation_plan.md, §7.2: full-schema recall by number of gold tables, and recall
of two kinds of gold tables that matter for joins:
- bridge tables: articulation points of the gold tables' induced subgraph (structural);
- join-only tables: tables whose columns appear only in JOIN ... ON conditions (SQL-based).
"""

from collections import Counter
from typing import Any

import networkx as nx
import pandas as pd
from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from src.analysis.load import DATASETS, SEEDS, VARIANTS, Records, method_name
from src.retrievers.base import table_id


def bridge_tables(gold_ids: list[str], graph: nx.Graph) -> set[str]:
    """Return the bridge tables of a question (§7.2, structural definition).

    A bridge table is a gold table whose removal disconnects the other gold tables. Only
    questions with at least 3 gold tables that form a connected subgraph are considered.

    Args:
        gold_ids: Gold table ids of the question.
        graph: Table graph of the collection.

    Returns:
        Set of bridge table ids (empty if the question does not qualify).
    """
    if len(gold_ids) < 3:
        return set()
    induced = graph.subgraph(gold_ids)
    if not nx.is_connected(induced):
        return set()
    return set(nx.articulation_points(induced))


def _prepare_tree(sql: str, database: list[dict[str, Any]]) -> exp.Expression:
    """Parse gold SQL and repair quoting so that `qualify` can resolve every column.

    Spider writes string values in double quotes ("English"), which SQLite syntax reads as
    identifiers. A double-quoted name that is not a table or column of the database is turned
    back into a string. All remaining identifiers are unquoted so they match the lower-cased
    schema (Fiben quotes its upper-case names).

    Args:
        sql: Gold SQL.
        database: Table list of the gold database.

    Returns:
        The repaired syntax tree.
    """
    known = {t["name"] for t in database} | {c["name"] for t in database for c in t["columns"]}
    tree = parse_one(sql, read="sqlite")
    for column in list(tree.find_all(exp.Column)):
        identifier = column.this
        if (
            isinstance(identifier, exp.Identifier)
            and identifier.quoted
            and not column.table
            and identifier.name.lower() not in known
        ):
            column.replace(exp.Literal.string(identifier.name))
    for identifier in tree.find_all(exp.Identifier):
        identifier.set("quoted", False)
    return tree


def _in_join_condition(node: exp.Expression) -> bool:
    """Return True if `node` lies inside the ON condition of a JOIN."""
    while node.parent is not None:
        if isinstance(node.parent, exp.Join) and node.arg_key == "on":
            return True
        node = node.parent
    return False


def join_only_tables(sql: str, database: list[dict[str, Any]]) -> set[str] | None:
    """Return the gold tables whose columns are used only in JOIN ... ON conditions (§7.2).

    Columns are first resolved to their tables with sqlglot's `qualify`, scope by scope, so
    unqualified columns and aliases (T1, T2) are bound to the right base table even when an
    alias is reused in a subquery.

    Args:
        sql: Gold SQL.
        database: Table list of the gold database.

    Returns:
        Set of table names, or None if the query cannot be resolved ("undetermined").
    """
    schema = {t["name"]: {c["name"]: "TEXT" for c in t["columns"]} for t in database}
    try:
        tree = qualify(_prepare_tree(sql, database), schema=schema, dialect="sqlite")
    except (SqlglotError, KeyError, ValueError):
        return None
    usage: dict[str, Counter] = {}
    for scope in traverse_scope(tree):
        for column in scope.columns:
            source = scope.sources.get(column.table)
            if isinstance(source, exp.Table):
                place = "on" if _in_join_condition(column) else "other"
                usage.setdefault(source.name.lower(), Counter())[place] += 1
    return {table for table, counts in usage.items() if counts["on"] and not counts["other"]}


def join_table_sets(
    instances: list[dict[str, Any]], schemas: dict[str, Any], graph: nx.Graph
) -> tuple[dict[str, set[str]], dict[str, set[str]], int]:
    """Find the bridge and join-only gold tables of every question of a dataset.

    Args:
        instances: Test instances.
        schemas: Schema collection.
        graph: Table graph.

    Returns:
        (bridge table ids per question, join-only table ids per question, number of questions
        whose SQL could not be resolved).
    """
    bridges, join_only, undetermined = {}, {}, 0
    for instance in instances:
        gold_ids = [table_id(instance["gold_db"], t) for t in instance["gold_tables"]]
        bridges[instance["id"]] = bridge_tables(gold_ids, graph)
        tables = join_only_tables(instance["gold_sql"], schemas[instance["gold_db"]])
        if tables is None:
            undetermined += 1
            tables = set()
        join_only[instance["id"]] = {
            table_id(instance["gold_db"], t) for t in tables & set(instance["gold_tables"])
        }
    return bridges, join_only, undetermined


def subset_recall(records: Records, targets: dict[str, set[str]], k: int) -> tuple[float, int]:
    """Recall@k over a set of target tables (for example all bridge tables of a dataset).

    Args:
        records: One method's records.
        targets: Target table ids per question.
        k: Cutoff in the method's table ranking.

    Returns:
        (percentage of target tables found in the top k, number of target tables).
    """
    found = total = 0
    for question_id, tables in targets.items():
        ranked = {t for t, _ in records[question_id]["ranked_tables"][:k]}
        found += len(tables & ranked)
        total += len(tables)
    return (100 * found / total if total else float("nan")), total


def join_recall_table(
    results: dict[tuple[str, str, str], Records],
    table_sets: dict[str, tuple[dict[str, set[str]], dict[str, set[str]], int]],
    ks: tuple[int, ...] = (5, 15),
) -> pd.DataFrame:
    """Recall of bridge, join-only and overlapping tables for every method and dataset.

    Args:
        results: Output of `load_all`.
        table_sets: Per dataset, the output of `join_table_sets`.
        ks: Cutoffs.

    Returns:
        One row per (dataset, method, kind) with the table count and recall@k columns.
    """
    rows = []
    for dataset in DATASETS:
        bridges, join_only, _ = table_sets[dataset]
        overlap = {q: bridges[q] & join_only[q] for q in bridges}
        for kind, targets in (("bridge", bridges), ("join-only", join_only), ("both", overlap)):
            for seed in SEEDS:
                for variant in VARIANTS:
                    records = results[(seed, variant, dataset)]
                    row: dict[str, Any] = {
                        "dataset": dataset,
                        "kind": kind,
                        "method": method_name(seed, variant),
                    }
                    for k in ks:
                        row[f"R@{k}"], row["tables"] = subset_recall(records, targets, k)
                    rows.append(row)
    return pd.DataFrame(rows).round(2)


def recall_by_gold_count(results: dict[tuple[str, str, str], Records]) -> pd.DataFrame:
    """Table recall@5 and @15 (per question) grouped by the number of gold tables (1, 2, 3, 4+).

    The same breakdown as the paper's Figure 7(b), with the paper's own metric (table recall),
    so the join analysis needs no metric of our own.

    Args:
        results: Output of `load_all`.

    Returns:
        One row per (dataset, method, gold-table group) with question count and TR@k.
    """
    rows = []
    for (seed, variant, dataset), records in results.items():
        frame = pd.DataFrame([r["metrics"] for r in records.values()])
        frame["gold"] = frame["n_gold"].clip(upper=4).astype(int).astype(str).replace("4", "4+")
        for group, part in frame.groupby("gold"):
            rows.append(
                {
                    "dataset": dataset,
                    "method": method_name(seed, variant),
                    "gold_tables": group,
                    "questions": len(part),
                    "TR@5": round(100 * part["TR@5"].mean(), 2),
                    "TR@15": round(100 * part["TR@15"].mean(), 2),
                }
            )
    return pd.DataFrame(rows)
