"""Dense table retrieval (SXFMR) with `retriv` and `all-mpnet-base-v2`.

Implements the SXFMR baseline and dense seeding (implementation_plan.md, §2.4, §4.3, §5.2).
"""

from pathlib import Path
from typing import Any

import retriv
from retriv import DenseRetriever

from src.retrievers.base import (
    ScoreMatrix,
    question_queries,
    results_to_matrix,
    table_documents,
)
from src.schemas import SchemaCollection


def build_dense_index(
    documents: list[dict[str, str]],
    index_name: str,
    retriv_dir: Path,
    model: str,
    max_length: int,
    normalize: bool,
    batch_size: int,
) -> DenseRetriever:
    """Build a retriv dense index (exact search) over table documents.

    Args:
        documents: Output of `table_documents`.
        index_name: Name of the retriv index.
        retriv_dir: Directory where retriv stores its indexes and embeddings.
        model: Hugging Face model id of the sentence encoder.
        max_length: Maximum number of tokens per input.
        normalize: L2-normalise embeddings.
        batch_size: Encoding batch size.

    Returns:
        The indexed retriever.
    """
    retriv.set_base_path(str(retriv_dir))
    retriever = DenseRetriever(
        index_name=index_name,
        model=model,
        normalize=normalize,
        max_length=max_length,
        use_ann=False,
    )
    return retriever.index(documents, batch_size=batch_size, show_progress=False)


def dense_scores(
    schemas: SchemaCollection,
    instances: list[dict[str, Any]],
    index_name: str,
    retriv_dir: Path,
    model: str = "sentence-transformers/all-mpnet-base-v2",
    max_length: int = 128,
    normalize: bool = True,
    batch_size: int = 64,
) -> ScoreMatrix:
    """Score every table of the collection for every question by embedding similarity.

    Uses exact search (no approximate index), as the authors did (`use_ann=False`). Inputs are
    truncated to `max_length` tokens; the authors kept retriv's default of 128, which cuts the
    text of tables with many columns. retriv runs the encoder on the CPU (its GPU option is
    CUDA only).

    Args:
        schemas: Schema collection to index.
        instances: Processed instances whose questions are scored.
        index_name: Name of the retriv index (one per collection).
        retriv_dir: Directory where retriv stores its indexes and embeddings.
        model: Hugging Face model id of the sentence encoder.
        max_length: Maximum number of tokens per input.
        normalize: L2-normalise embeddings, so the score is cosine similarity.
        batch_size: Encoding batch size.

    Returns:
        ScoreMatrix of shape (questions, tables); every table has a score.
    """
    documents = table_documents(schemas)
    retriever = build_dense_index(
        documents, index_name, retriv_dir, model, max_length, normalize, batch_size
    )
    results = retriever.bsearch(
        question_queries(instances),
        cutoff=len(documents),
        batch_size=batch_size,
        show_progress=False,
    )
    question_ids = [instance["id"] for instance in instances]
    return results_to_matrix(results, question_ids, [d["id"] for d in documents])
