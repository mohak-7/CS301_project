"""Build the foreign-key table graph used by GraphRAG.

Implements §5.1 and D6 of implementation_plan.md: a port of `schema2graph` in the authors'
src/utils/helpers.py, keeping only the table-to-table edges, treated as undirected.
"""

import itertools

import networkx as nx

from src.retrievers.base import table_id
from src.schemas import SchemaCollection


def schema_graph(schemas: SchemaCollection) -> nx.Graph:
    """Build the undirected table graph of a whole collection.

    Nodes are table ids ("db_id.table_name"), one per table, including tables without foreign
    keys. As in the authors' code, every group of tables that reference the same key column
    (`table.column`), together with the table that owns that key, is fully connected. This
    covers primary-foreign links and foreign-foreign links (two tables pointing at the same key).
    The foreign keys already include the authors' missing-key patch (§2.2). Databases share no
    edges, so each database is a separate set of connected pieces.

    Args:
        schemas: Schema collection (schemas.json structure).

    Returns:
        Undirected graph with a `db_id` attribute on every node.
    """
    graph = nx.Graph()
    for db_id, tables in schemas.items():
        key_groups: dict[str, set[str]] = {}
        for table in tables:
            graph.add_node(table_id(db_id, table["name"]), db_id=db_id)
            for column in table["columns"]:
                foreign_key = column.get("foreign_key")
                if foreign_key:
                    key = f"{foreign_key['table']}.{foreign_key['column']}"
                    key_groups.setdefault(key, set()).update([foreign_key["table"], table["name"]])
        for group in key_groups.values():
            for first, second in itertools.combinations(sorted(group), 2):
                graph.add_edge(table_id(db_id, first), table_id(db_id, second))
    return graph
