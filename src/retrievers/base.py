"""Shared retriever definitions: table documents, question queries and score matrices.

Implements the common retriever interface (implementation_plan.md, §2.4, §4.1).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.schemas import SchemaCollection


@dataclass
class ScoreMatrix:
    """Scores of every table in a collection for every question of a dataset.

    Attributes:
        question_ids: Instance ids, one per row.
        table_ids: "db_id.table_name" ids, one per column, in collection order.
        scores: float32 array of shape (questions, tables). NaN marks a table the retriever did
            not return; BM25 returns only tables sharing a term with the question (D14).
    """

    question_ids: list[str]
    table_ids: list[str]
    scores: np.ndarray

    def save(self, path: Path) -> None:
        """Save the matrix as a compressed .npz file.

        Args:
            path: Destination file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            question_ids=np.array(self.question_ids),
            table_ids=np.array(self.table_ids),
            scores=self.scores,
        )

    @classmethod
    def load(cls, path: Path) -> "ScoreMatrix":
        """Load a matrix saved with `save`.

        Args:
            path: Source .npz file.

        Returns:
            The loaded ScoreMatrix.
        """
        data = np.load(path)
        return cls(list(data["question_ids"]), list(data["table_ids"]), data["scores"])


def table_id(db_id: str, table_name: str) -> str:
    """Return the collection-wide table id used by the authors ("db_id.table_name")."""
    return f"{db_id}.{table_name}"


def split_table_id(identifier: str) -> tuple[str, str]:
    """Split a table id into (db_id, table_name).

    Args:
        identifier: "db_id.table_name"; database ids contain no dots.

    Returns:
        The database id and the table name.
    """
    db_id, table_name = identifier.split(".", 1)
    return db_id, table_name


def table_documents(schemas: SchemaCollection) -> list[dict[str, str]]:
    """Build one retrieval document per table, in collection order (§2.4).

    The text is "<db_id> <normalised table name> <normalised column names>", as in the authors'
    `generate_collection`. The database id is part of the text, so words in a database name
    (for example `concert_singer`) can match the question.

    Args:
        schemas: Schema collection.

    Returns:
        List of {"id": "db_id.table_name", "text": document text}.
    """
    documents = []
    for db_id, tables in schemas.items():
        for table in tables:
            column_names = " ".join(column["normalized_name"] for column in table["columns"])
            documents.append(
                {
                    "id": table_id(db_id, table["name"]),
                    "text": f"{db_id} {table['normalized_name']} {column_names}",
                }
            )
    return documents


def question_queries(instances: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build retriever queries from processed instances.

    Args:
        instances: Processed test or train instances.

    Returns:
        List of {"id": instance id, "text": question}.
    """
    return [{"id": instance["id"], "text": instance["question"]} for instance in instances]


def results_to_matrix(
    results: dict[str, dict[str, float]], question_ids: list[str], table_ids: list[str]
) -> ScoreMatrix:
    """Convert retriv search results into a ScoreMatrix.

    Args:
        results: {question_id: {table_id: score}} as returned by retriv's `bsearch`.
        question_ids: Row order.
        table_ids: Column order (collection order).

    Returns:
        ScoreMatrix with NaN for tables missing from a question's results.
    """
    column_index = {identifier: index for index, identifier in enumerate(table_ids)}
    scores = np.full((len(question_ids), len(table_ids)), np.nan, dtype=np.float32)
    for row, question_id in enumerate(question_ids):
        for identifier, score in results.get(question_id, {}).items():
            scores[row, column_index[identifier]] = score
    return ScoreMatrix(question_ids, table_ids, scores)
