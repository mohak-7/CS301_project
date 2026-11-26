"""BM25 table retrieval with `retriv`, as in the authors' baselines.

Implements the BM25 baseline and BM25 seeding (implementation_plan.md, §2.4, §4.3, §5.2).
"""

from pathlib import Path
from typing import Any

import retriv
from retriv import SparseRetriever

from src.retrievers.base import (
    ScoreMatrix,
    question_queries,
    results_to_matrix,
    table_documents,
)
from src.schemas import SchemaCollection


def build_bm25_index(
    documents: list[dict[str, str]], index_name: str, retriv_dir: Path, k1: float, b: float
) -> SparseRetriever:
    """Build a retriv BM25 index over table documents.

    Args:
        documents: Output of `table_documents`.
        index_name: Name of the retriv index.
        retriv_dir: Directory where retriv stores its indexes.
        k1: BM25 term-frequency saturation.
        b: BM25 length normalisation.

    Returns:
        The indexed retriever.
    """
    retriv.set_base_path(str(retriv_dir))
    retriever = SparseRetriever(index_name=index_name, hyperparams={"k1": k1, "b": b})
    return retriever.index(documents, show_progress=False)


def bm25_scores(
    schemas: SchemaCollection,
    instances: list[dict[str, Any]],
    index_name: str,
    retriv_dir: Path,
    k1: float = 1.2,
    b: float = 0.75,
) -> ScoreMatrix:
    """Score every table of the collection for every question with BM25.

    All tables are requested (cutoff = collection size) so that Phase 5 can reuse the scores;
    the baseline's top-100 cut is applied later (D8). `retriv` returns only tables that share
    a term with the question, so the others stay NaN in the matrix (D14).

    Args:
        schemas: Schema collection to index.
        instances: Processed instances whose questions are scored.
        index_name: Name of the retriv index (one per collection).
        retriv_dir: Directory where retriv stores its indexes.
        k1: BM25 term-frequency saturation (retriv default).
        b: BM25 length normalisation (retriv default).

    Returns:
        ScoreMatrix of shape (questions, tables).
    """
    documents = table_documents(schemas)
    retriever = build_bm25_index(documents, index_name, retriv_dir, k1, b)
    results = retriever.bsearch(
        question_queries(instances), cutoff=len(documents), show_progress=False
    )
    question_ids = [instance["id"] for instance in instances]
    return results_to_matrix(results, question_ids, [d["id"] for d in documents])
