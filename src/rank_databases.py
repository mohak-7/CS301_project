"""Turn table scores into a top-k table list and a database ranking.

Implements the paper's ranking rule (implementation_plan.md, §2.4, D8, D14): keep the top 100
tables of the whole collection, then score each database by the average of its kept tables.
"""

import numpy as np

from src.retrievers.base import split_table_id

RankedTables = list[tuple[str, float]]  # [(table_id, score)], best first
RankedDatabases = list[tuple[str, float]]  # [(db_id, score)], best first


def top_tables(scores: np.ndarray, table_ids: list[str], cutoff: int = 100) -> RankedTables:
    """Return the best `cutoff` tables of one question.

    Tables with NaN scores (not returned by the retriever) are never used to fill the list
    (D14), so the list can be shorter than `cutoff`. Ties keep collection order (stable sort),
    which makes the ranking deterministic.

    Args:
        scores: Scores of all tables for one question, shape (tables,).
        table_ids: Table ids in the same order as `scores`.
        cutoff: Maximum number of tables to keep.

    Returns:
        List of (table_id, score), best first.
    """
    retrieved = np.flatnonzero(~np.isnan(scores))
    order = retrieved[np.argsort(-scores[retrieved], kind="stable")][:cutoff]
    return [(table_ids[index], float(scores[index])) for index in order]


def rank_databases(
    ranked_tables: RankedTables, top_per_database: int | None = None
) -> RankedDatabases:
    """Rank databases by the mean score of their tables in the top-k list (D8).

    This is the authors' rule in scripts/retrival.py: sum of a database's retrieved table scores
    divided by the number of its retrieved tables. Ties keep the order in which databases
    first appear in the table list.

    Args:
        ranked_tables: Output of `top_tables`.
        top_per_database: If set, average only each database's best this-many tables (an R-seed
            ablation, D13); None applies the paper's rule.

    Returns:
        List of (db_id, mean score), best first.
    """
    per_database: dict[str, list[float]] = {}
    for identifier, score in ranked_tables:
        db_id, _ = split_table_id(identifier)
        per_database.setdefault(db_id, []).append(score)
    # Scores arrive best first, so slicing keeps each database's best tables.
    averages = [
        (db_id, float(np.mean(scores[:top_per_database]))) for db_id, scores in per_database.items()
    ]
    return sorted(averages, key=lambda item: -item[1])


def hierarchical_tables(
    ranked_tables: RankedTables, ranked_databases: RankedDatabases, num_databases: int = 5
) -> RankedTables:
    """Group a table list by database, in database rank order (R-hier mode, D13).

    DBCopilot lists the tables of its candidate schemas database by database. This applies the
    same ordering to a retriever's table list: the tables of the best database first (in score
    order), then those of the second database, and so on for the top `num_databases`.

    Args:
        ranked_tables: Output of `top_tables` (global order).
        ranked_databases: Database ranking used for grouping.
        num_databases: Number of databases whose tables are listed.

    Returns:
        Grouped table list; tables of lower-ranked databases are left out.
    """
    by_database: dict[str, RankedTables] = {}
    for identifier, score in ranked_tables:
        db_id, _ = split_table_id(identifier)
        by_database.setdefault(db_id, []).append((identifier, score))
    grouped: RankedTables = []
    for db_id, _ in ranked_databases[:num_databases]:
        grouped.extend(by_database.get(db_id, []))
    return grouped
