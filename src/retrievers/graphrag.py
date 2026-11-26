"""GraphRAG table scoring: semantic seeds combined with foreign-key graph structure.

Implements Steps 1 and 2 of implementation_plan.md, §5.2 and §5.3: min-max scaled semantic
seeds (G0), neighbour expansion (G1), personalized PageRank over the whole collection (G2) and
connector completion along shortest foreign-key paths (G3), with the §5.3 safeguards.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import sparse

from src.rank_databases import rank_databases, top_tables
from src.retrievers.base import ScoreMatrix, split_table_id

VARIANTS = ("G0", "G1", "G2", "G3")


@dataclass(frozen=True)
class GraphRAGParams:
    """Hyperparameters of the GraphRAG variants (§5.5); values come from the config files.

    Attributes:
        variant: One of G0, G1, G2, G3.
        beta: G1 weight of the best neighbour's score.
        alpha: G2/G3 weight of the semantic score (1 - alpha goes to PageRank).
        num_seeds: G2/G3 number of top semantic tables in the restart vector (N).
        damping: G2/G3 PageRank damping factor.
        pagerank_iterations: Maximum power-iteration steps.
        pagerank_tolerance: L1 convergence tolerance per question.
        connector_databases: G3 number of top databases searched for connectors (B).
        connector_seeds: G3 seed tables per database (M).
        seed_threshold: G3 minimum scaled semantic score of a seed (tau).
        max_path_length: G3 maximum foreign-key path length in edges (L_max).
        gamma: G3 connector factor; connectors score gamma * min(score(u), score(v)).
        cutoff: Tables kept before database ranking (D8).
    """

    variant: str = "G0"
    beta: float = 0.3
    alpha: float = 0.7
    num_seeds: int = 10
    damping: float = 0.85
    pagerank_iterations: int = 100
    pagerank_tolerance: float = 1e-8
    connector_databases: int = 3
    connector_seeds: int = 3
    seed_threshold: float = 0.3
    max_path_length: int = 2
    gamma: float = 0.95
    cutoff: int = 100


def min_max_scale(scores: np.ndarray) -> np.ndarray:
    """Scale each row of a score array to [0, 1] (§5.2).

    Tables a retriever did not return (NaN) count as 0 before scaling. A row whose scores are
    all equal (for example a complete BM25 miss, where every score is 0) would divide by zero,
    so it becomes all zeros instead.

    Args:
        scores: Array of shape (questions, tables), possibly containing NaN.

    Returns:
        Scaled array of the same shape, with no NaN.
    """
    filled = np.nan_to_num(scores.astype(np.float64), nan=0.0)
    row_min = filled.min(axis=1, keepdims=True)
    row_range = filled.max(axis=1, keepdims=True) - row_min
    safe_range = np.where(row_range > 0, row_range, 1.0)
    return np.where(row_range > 0, (filled - row_min) / safe_range, 0.0)


def neighbour_lists(graph: nx.Graph, table_ids: list[str]) -> list[np.ndarray]:
    """Return, for every table, the column indices of its foreign-key neighbours.

    Args:
        graph: Table graph (src.graph.schema_graph).
        table_ids: Column order of the score matrix.

    Returns:
        One integer array per table; empty for tables without foreign keys.
    """
    position = {identifier: index for index, identifier in enumerate(table_ids)}
    return [
        np.array(sorted(position[n] for n in graph.neighbors(t)), dtype=np.int64) for t in table_ids
    ]


def neighbour_expansion(seeds: np.ndarray, neighbours: list[np.ndarray], beta: float) -> np.ndarray:
    """G1: add beta times the best neighbour's seed score to each table's seed score (§5.3).

    Args:
        seeds: Scaled semantic scores, shape (questions, tables).
        neighbours: Output of `neighbour_lists`.
        beta: Weight of the neighbour term.

    Returns:
        G1 scores, same shape as `seeds`.
    """
    best_neighbour = np.zeros_like(seeds)
    for column, adjacent in enumerate(neighbours):
        # Tables without foreign keys keep a neighbour term of 0 (the `default=0.0` safeguard).
        if adjacent.size:
            best_neighbour[:, column] = seeds[:, adjacent].max(axis=1)
    return seeds + beta * best_neighbour


def restart_vectors(seeds: np.ndarray, num_seeds: int) -> np.ndarray:
    """Build per-question PageRank restart vectors from the top-N seed scores (§5.3, G2).

    v(t) = s(t) / sum of s over the top-N seeds, and 0 for all other tables. A question whose
    top-N seeds all score 0 gets an all-zero vector; PageRank is then skipped for it.

    Args:
        seeds: Scaled semantic scores, shape (questions, tables).
        num_seeds: Number of top tables kept per question (N).

    Returns:
        Restart vectors, same shape as `seeds`; each row sums to 1 or is all zero.
    """
    vectors = np.zeros_like(seeds)
    top = np.argsort(-seeds, axis=1, kind="stable")[:, :num_seeds]
    rows = np.arange(seeds.shape[0])[:, None]
    vectors[rows, top] = seeds[rows, top]
    mass = vectors.sum(axis=1, keepdims=True)
    return np.divide(vectors, mass, out=np.zeros_like(vectors), where=mass > 0)


def personalized_pagerank(
    graph: nx.Graph,
    table_ids: list[str],
    restart: np.ndarray,
    damping: float,
    iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Personalized PageRank for many questions at once over the whole collection (§5.3, G2).

    Runs one power iteration per question on the global table graph, so each database receives
    probability mass in proportion to how strongly its tables match the question. This follows
    networkx's `pagerank` semantics: random-walk steps along edges with probability `damping`,
    otherwise a jump to the restart vector; tables without edges send their mass to the restart
    vector. Rows with an all-zero restart vector stay all zero instead of raising an error.

    Args:
        graph: Table graph.
        table_ids: Column order of `restart`.
        restart: Restart vectors, shape (questions, tables), rows summing to 1 or 0.
        damping: Probability of following an edge.
        iterations: Maximum number of iterations.
        tolerance: Stop when every row changes by less than this (L1 norm).

    Returns:
        PageRank scores, same shape as `restart`.
    """
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=table_ids, format="csr", dtype=float)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    dangling = degree == 0
    # Row-normalise so each table splits its mass equally among its neighbours.
    transition = sparse.diags(np.divide(1.0, degree, where=~dangling, out=np.zeros_like(degree)))
    transition = (transition @ adjacency).tocsr()
    ranks = restart.copy()
    for _ in range(iterations):
        dangling_mass = ranks[:, dangling].sum(axis=1, keepdims=True)
        updated = damping * (ranks @ transition + dangling_mass * restart) + (1 - damping) * restart
        converged = np.abs(updated - ranks).sum(axis=1).max() < tolerance
        ranks = updated
        if converged:
            break
    return ranks


def pagerank_scores(
    seeds: np.ndarray, graph: nx.Graph, table_ids: list[str], params: GraphRAGParams
) -> np.ndarray:
    """G2: mix scaled semantic scores with scaled personalized PageRank (§5.3).

    PageRank scores sum to 1, while seeds lie in [0, 1], so PageRank is min-max scaled per
    question before mixing; otherwise the structural term would be too small to matter.

    Args:
        seeds: Scaled semantic scores, shape (questions, tables).
        graph: Table graph.
        table_ids: Column order of `seeds`.
        params: GraphRAG hyperparameters (alpha, num_seeds, damping, iteration settings).

    Returns:
        G2 scores, same shape as `seeds`.
    """
    restart = restart_vectors(seeds, params.num_seeds)
    ranks = personalized_pagerank(
        graph,
        table_ids,
        restart,
        params.damping,
        params.pagerank_iterations,
        params.pagerank_tolerance,
    )
    return params.alpha * seeds + (1 - params.alpha) * min_max_scale(ranks)


def connector_paths(graph: nx.Graph, source: str, target: str, max_length: int) -> list[list[str]]:
    """Return all shortest foreign-key paths between two tables, if short enough (§5.3, G3).

    `nx.all_shortest_paths` is used because schemas with a diamond shape have several equally
    short paths; scoring all of them makes the result independent of graph insertion order.

    Args:
        graph: Table graph.
        source: First table id.
        target: Second table id.
        max_length: Maximum path length in edges (L_max).

    Returns:
        Paths as lists of table ids (endpoints included); empty if none qualifies.
    """
    if not nx.has_path(graph, source, target):
        return []
    paths = list(nx.all_shortest_paths(graph, source, target))
    return paths if len(paths[0]) - 1 <= max_length else []


def complete_connectors(
    scores: np.ndarray,
    seeds: np.ndarray,
    graph: nx.Graph,
    table_ids: list[str],
    params: GraphRAGParams,
) -> np.ndarray:
    """G3: raise the scores of tables on short paths between strong seeds (§5.3).

    For each of the top-B databases (ranked by the paper's rule on the G2 scores), take its
    top-M seed tables with scaled semantic score >= tau. For every pair (u, v), every table w
    strictly inside a shortest path of at most L_max edges gets
    score(w) = max(score(w), gamma * min(score(u), score(v))), which places it just behind the
    weaker of the two tables it connects.

    Args:
        scores: G2 scores of one question, shape (tables,).
        seeds: Scaled semantic scores of the same question, shape (tables,).
        graph: Table graph.
        table_ids: Order of `scores` and `seeds`.
        params: GraphRAG hyperparameters (B, M, tau, L_max, gamma, cutoff).

    Returns:
        G3 scores of the question, shape (tables,).
    """
    position = {identifier: index for index, identifier in enumerate(table_ids)}
    completed = scores.copy()
    ranked_databases = rank_databases(top_tables(scores, table_ids, params.cutoff))
    for db_id, _ in ranked_databases[: params.connector_databases]:
        candidates = [
            index
            for index in np.argsort(-seeds, kind="stable")
            if split_table_id(table_ids[index])[0] == db_id
            and seeds[index] >= params.seed_threshold
        ][: params.connector_seeds]
        for first, second in zip(*np.triu_indices(len(candidates), k=1), strict=True):
            u, v = candidates[first], candidates[second]
            bridge_score = params.gamma * min(scores[u], scores[v])
            for path in connector_paths(graph, table_ids[u], table_ids[v], params.max_path_length):
                for inner in path[1:-1]:
                    completed[position[inner]] = max(completed[position[inner]], bridge_score)
    return completed


def graphrag_scores(
    seed_matrix: ScoreMatrix, graph: nx.Graph, params: GraphRAGParams
) -> ScoreMatrix:
    """Compute GraphRAG scores for every question and table (§5.2, §5.3).

    Tables whose final score is not positive are marked as not retrieved (NaN), so they are
    never used to fill the top-k list (D14). For G0 this reproduces the baseline exactly:
    min-max scaling is linear, so neither the table order nor the database order changes.

    Args:
        seed_matrix: Full semantic score matrix (BM25 or SXFMR) from Phase 4.
        graph: Table graph of the same collection.
        params: GraphRAG hyperparameters; `params.variant` selects G0-G3.

    Returns:
        ScoreMatrix with the GraphRAG scores.
    """
    if params.variant not in VARIANTS:
        raise ValueError(f"unknown variant {params.variant}; expected one of {VARIANTS}")
    table_ids = seed_matrix.table_ids
    seeds = min_max_scale(seed_matrix.scores)
    if params.variant == "G0":
        final = seeds
    elif params.variant == "G1":
        final = neighbour_expansion(seeds, neighbour_lists(graph, table_ids), params.beta)
    else:
        final = pagerank_scores(seeds, graph, table_ids, params)
        if params.variant == "G3":
            final = np.vstack(
                [
                    complete_connectors(final[row], seeds[row], graph, table_ids, params)
                    for row in range(final.shape[0])
                ]
            )
    final = np.where(final > 0, final, np.nan).astype(np.float32)
    return ScoreMatrix(seed_matrix.question_ids, table_ids, final)
