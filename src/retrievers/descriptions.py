"""Dense retrieval over LLM-written table descriptions.

Implements the description channel of the extension (implementation_plan.md, §10). Each table's
description is embedded once; a question is matched to every table by cosine similarity. The
descriptions are prose, so a sentence encoder handles them better than bare column lists, and
they fit its input length (the schema text of wide tables does not, §4.5).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrievers.base import ScoreMatrix, table_id
from src.retrievers.rerank import resolve_device
from src.schemas import SchemaCollection


def load_descriptions(path: Path) -> dict[str, str]:
    """Load table descriptions written by src/describe_tables.py, keyed by table id."""
    with path.open() as handle:
        return {row["table_id"]: row["description"] for row in map(json.loads, handle)}


def description_texts(schemas: SchemaCollection, descriptions: dict[str, str]) -> dict[str, str]:
    """Return the text indexed for each table: its normalised name followed by its description.

    The table name is kept in front of the description because questions often mention it.

    Args:
        schemas: Schema collection.
        descriptions: Descriptions keyed by table id.

    Returns:
        Mapping from table id to "<table name>: <description>", in collection order.

    Raises:
        KeyError: If a table has no description (the generation run is incomplete).
    """
    texts = {}
    for db_id, tables in schemas.items():
        for table in tables:
            identifier = table_id(db_id, table["name"])
            texts[identifier] = f"{table['normalized_name']}: {descriptions[identifier]}"
    return texts


def embed(texts: list[str], encoder: SentenceTransformer, batch_size: int) -> np.ndarray:
    """Encode texts into L2-normalised embeddings (rows), so dot products are cosines."""
    return encoder.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )


def description_scores(
    instances: list[dict[str, Any]],
    texts: dict[str, str],
    settings: dict[str, Any],
) -> ScoreMatrix:
    """Score every table for every question by question-description cosine similarity.

    Args:
        instances: Processed instances (rows of the result).
        texts: Output of `description_texts` (columns of the result, in its order).
        settings: Encoder settings (model, max_length, batch_size, device).

    Returns:
        ScoreMatrix of cosine similarities; every table has a score.
    """
    encoder = SentenceTransformer(settings["model"], device=resolve_device(settings["device"]))
    encoder.max_seq_length = settings["max_length"]
    table_ids = list(texts)
    tables = embed([texts[t] for t in table_ids], encoder, settings["batch_size"])
    questions = embed([i["question"] for i in instances], encoder, settings["batch_size"])
    scores = (questions @ tables.T).astype(np.float32)
    return ScoreMatrix([i["id"] for i in instances], table_ids, scores)
