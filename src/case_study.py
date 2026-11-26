"""Show the top databases and tables that each method returns for selected questions.

Implements the Phase 5 sanity check on the paper's Figures 8 and 9 (implementation_plan.md,
§5 deliverables) and supports the case studies of §7.2.
"""

import argparse
import logging
from typing import Any

from src.common import PROJECT_ROOT, load_config, read_json, read_jsonl, setup_logging
from src.graph import schema_graph
from src.rank_databases import rank_databases, top_tables
from src.retrievers.base import ScoreMatrix, split_table_id
from src.retrievers.graphrag import VARIANTS, GraphRAGParams, graphrag_scores
from src.run_baselines import compute_scores

logger = logging.getLogger(__name__)


def find_questions(instances: list[dict[str, Any]], phrases: list[str]) -> list[int]:
    """Return the row index of the first instance whose question contains each phrase.

    Args:
        instances: Processed instances.
        phrases: Case-insensitive substrings of the wanted questions.

    Returns:
        Row indices, one per phrase.

    Raises:
        ValueError: If a phrase matches no question.
    """
    rows = []
    for phrase in phrases:
        matches = [
            i for i, inst in enumerate(instances) if phrase.lower() in inst["question"].lower()
        ]
        if not matches:
            raise ValueError(f"no question contains: {phrase}")
        rows.append(matches[0])
    return rows


def describe(scores: Any, table_ids: list[str], top_k: int, cutoff: int) -> str:
    """Format the top database and its top tables for one question (R-main ranking).

    Args:
        scores: Score row of one question.
        table_ids: Column order of `scores`.
        top_k: Number of tables shown.
        cutoff: Tables kept before database ranking (D8).

    Returns:
        Text such as "world_1: {country, countrylanguage, city}".
    """
    ranked_tables = top_tables(scores, table_ids, cutoff)
    if not ranked_tables:
        return "(nothing retrieved)"
    best_db = rank_databases(ranked_tables)[0][0]
    tables = [split_table_id(t)[1] for t, _ in ranked_tables if split_table_id(t)[0] == best_db]
    return f"{best_db}: {{{', '.join(tables[:top_k])}}}"


def main() -> None:
    """Print each method's routing for the selected questions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="spider")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument(
        "--phrases",
        nargs="+",
        default=["most popular on the Asian continent", "flights fly from Aberdeen to Ashley"],
        help="substrings of the questions to show (default: the paper's Figures 8 and 9)",
    )
    parser.add_argument("--config", default="configs/graphrag.yaml")
    parser.add_argument("--baseline-config", default="configs/baselines.yaml")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    setup_logging()
    baseline_config = load_config(args.baseline_config)
    params = load_config(args.config)["params"]
    processed = PROJECT_ROOT / baseline_config["processed_dir"] / args.dataset
    instances = read_jsonl(processed / f"{args.split}.jsonl")
    graph = schema_graph(read_json(processed / "schemas.json"))
    rows = find_questions(instances, args.phrases)
    for seed in ("bm25", "sxfmr"):
        full = compute_scores(seed, args.dataset, baseline_config, split=args.split)
        subset = ScoreMatrix(
            [full.question_ids[r] for r in rows], full.table_ids, full.scores[rows]
        )
        for variant in VARIANTS:
            final = graphrag_scores(subset, graph, GraphRAGParams(variant=variant, **params))
            for position, row in enumerate(rows):
                instance = instances[row]
                logger.info(
                    "%s | %s-%s | %s",
                    instance["question"],
                    seed,
                    variant,
                    describe(final.scores[position], final.table_ids, args.top_k, params["cutoff"]),
                )
    for row in rows:
        gold = f"{instances[row]['gold_db']}: {{{', '.join(instances[row]['gold_tables'])}}}"
        logger.info("%s | gold | %s", instances[row]["question"], gold)


if __name__ == "__main__":
    main()
