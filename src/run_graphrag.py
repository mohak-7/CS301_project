"""Run a GraphRAG variant with a ranking mode on one or more datasets and evaluate it.

Implements Step 3 of implementation_plan.md, §5.4 (ranking modes R-main, R-seed, R-hier, D13)
on top of the scoring in src/retrievers/graphrag.py. Seeds are the Phase 4 score matrices.
"""

import argparse
import logging
import time
from typing import Any

import networkx as nx
import pandas as pd

from src.common import PROJECT_ROOT, load_config, read_json, read_jsonl, setup_logging, write_jsonl
from src.evaluate import question_metrics, summarise
from src.graph import schema_graph
from src.rank_databases import hierarchical_tables, rank_databases, top_tables
from src.retrievers.base import ScoreMatrix
from src.retrievers.graphrag import VARIANTS, GraphRAGParams, graphrag_scores
from src.seeds import SEED_METHODS, seed_scores

logger = logging.getLogger(__name__)
MODES = ("R-main", "R-seed", "R-seed-top3", "R-hier")


def rank_question(
    final_scores: Any,
    seed_scores: Any,
    table_ids: list[str],
    mode: str,
    params: GraphRAGParams,
    ranking: dict[str, int],
) -> tuple[list, list]:
    """Rank databases and tables of one question under a ranking mode (§5.4, D13).

    Args:
        final_scores: GraphRAG scores of the question, shape (tables,), NaN = not retrieved.
        seed_scores: Semantic seed scores of the question (Phase 4 matrix row).
        table_ids: Column order of both score rows.
        mode: R-main, R-seed, R-seed-top3 or R-hier.
        params: GraphRAG hyperparameters (for the cutoff).
        ranking: Ranking settings (hier_databases, seed_top_tables).

    Returns:
        (ranked databases, ranked tables), both best first.
    """
    ranked_tables = top_tables(final_scores, table_ids, params.cutoff)
    if mode == "R-main":
        return rank_databases(ranked_tables), ranked_tables
    if mode == "R-hier":
        ranked_databases = rank_databases(ranked_tables)
        grouped = hierarchical_tables(ranked_tables, ranked_databases, ranking["hier_databases"])
        return ranked_databases, grouped
    # R-seed modes rank databases from the semantic seeds, so graph expansion cannot change
    # which database wins; tables are still ranked by GraphRAG.
    seed_tables = top_tables(seed_scores, table_ids, params.cutoff)
    top_per_database = ranking["seed_top_tables"] if mode == "R-seed-top3" else None
    return rank_databases(seed_tables, top_per_database), ranked_tables


def evaluate_variant(
    final: ScoreMatrix,
    seeds: ScoreMatrix,
    instances: list[dict[str, Any]],
    mode: str,
    params: GraphRAGParams,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rank and evaluate every question of a dataset.

    Args:
        final: GraphRAG score matrix.
        seeds: Semantic seed score matrix (same rows and columns).
        instances: Processed instances in matrix row order.
        mode: Ranking mode.
        params: GraphRAG hyperparameters.
        settings: Dict with `ranking` (configs/graphrag.yaml) and `metrics`
            (configs/baselines.yaml) sections.

    Returns:
        Prediction records with ranked databases, ranked tables and metrics.
    """
    metrics = settings["metrics"]
    records = []
    for row, instance in enumerate(instances):
        ranked_databases, ranked_tables = rank_question(
            final.scores[row], seeds.scores[row], final.table_ids, mode, params, settings["ranking"]
        )
        scores = question_metrics(
            instance,
            ranked_databases,
            ranked_tables,
            metrics["database_ks"],
            metrics["table_ks"],
            metrics["full_schema_ks"],
        )
        records.append(
            {
                "id": instance["id"],
                "ranked_databases": ranked_databases,
                "ranked_tables": ranked_tables,
                "metrics": scores,
            }
        )
    return records


def load_inputs(
    dataset: str, seed_method: str, split: str, baseline_config: dict[str, Any]
) -> tuple[list[dict[str, Any]], ScoreMatrix, nx.Graph]:
    """Load the instances, seed score matrix and table graph of a dataset.

    Args:
        dataset: Dataset name.
        seed_method: One of SEED_METHODS (src/seeds.py).
        split: "test" or "train".
        baseline_config: configs/baselines.yaml.

    Returns:
        (instances, seed matrix, table graph). Questions without gold tables are removed from
        both the instances and the matrix.
    """
    processed = PROJECT_ROOT / baseline_config["processed_dir"] / dataset
    instances = read_jsonl(processed / f"{split}.jsonl")
    seeds = seed_scores(seed_method, dataset, split, baseline_config)
    if seeds.question_ids != [instance["id"] for instance in instances]:
        raise ValueError(f"seed matrix rows do not match {dataset}/{split}")
    # Bird's training set has 34 questions on `retail_world` whose tables (Territories, Region)
    # are missing from its schema file, so they have no gold tables and cannot be scored.
    # No test set has such questions (checked in Phase 5).
    keep = [row for row, instance in enumerate(instances) if instance["gold_tables"]]
    if len(keep) < len(instances):
        logger.info(
            "%s/%s: skipped %d questions without gold tables",
            dataset,
            split,
            len(instances) - len(keep),
        )
        instances = [instances[row] for row in keep]
        seeds = ScoreMatrix(
            [seeds.question_ids[row] for row in keep], seeds.table_ids, seeds.scores[keep]
        )
    return instances, seeds, schema_graph(read_json(processed / "schemas.json"))


def run(
    dataset: str, seed_method: str, mode: str, params: GraphRAGParams, args: argparse.Namespace
) -> dict[str, Any]:
    """Run one configuration on one dataset and write its per-question results.

    Args:
        dataset: Dataset name.
        seed_method: "bm25" or "sxfmr".
        mode: Ranking mode.
        params: GraphRAG hyperparameters.
        args: Command-line arguments (split, configs, output name).

    Returns:
        Summary row of the run.
    """
    baseline_config = load_config(args.baseline_config)
    settings = {
        "metrics": baseline_config["metrics"],
        "ranking": load_config(args.config)["ranking"],
    }
    instances, seeds, graph = load_inputs(dataset, seed_method, args.split, baseline_config)
    started = time.perf_counter()
    final = graphrag_scores(seeds, graph, params)
    elapsed = time.perf_counter() - started
    records = evaluate_variant(final, seeds, instances, mode, params, settings)
    name = f"graphrag_{seed_method}_{params.variant}_{mode}"
    # Per-question records are only needed for the Phase 7 analysis, which uses the test set and
    # the main ranking mode. Other runs (training questions, ablation modes) log summaries only;
    # a training run covers about 16,000 questions (about 40 MB of records).
    if args.split == "test" and mode == "R-main":
        out_dir = PROJECT_ROOT / baseline_config["results_dir"] / args.output_group / name
        write_jsonl(out_dir / f"{dataset}_{args.split}.jsonl", records)
    summary = summarise([r["metrics"] for r in records], baseline_config["metrics"]["table_ks"])
    return {"method": name, "dataset": dataset, "split": args.split, "seconds": elapsed, **summary}


def main() -> None:
    """Run the requested GraphRAG configurations and log a summary table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/graphrag.yaml", help="GraphRAG settings")
    parser.add_argument("--baseline-config", default="configs/baselines.yaml")
    parser.add_argument("--seeds", nargs="+", default=["bm25", "sxfmr"], choices=list(SEED_METHODS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--modes", nargs="+", default=["R-main"], choices=list(MODES))
    parser.add_argument("--datasets", nargs="+", default=["spider"])
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output-group", default="graphrag_dev", help="results subfolder")
    parser.add_argument("--summary-file", default=None, help="CSV for the summary table")
    args = parser.parse_args()

    setup_logging()
    config_params = load_config(args.config)["params"]
    rows = [
        run(dataset, seed, mode, params_for(config_params, seed, variant), args)
        for dataset in args.datasets
        for seed in args.seeds
        for variant in args.variants
        for mode in args.modes
    ]
    summary = pd.DataFrame(rows).round(2)
    columns = ["method", "dataset", "DR@1", "DR@5", "TR@5", "TR@15", "FS@5", "FS@15", "seconds"]
    logger.info("summary:\n%s", summary[columns].to_string(index=False))
    if args.summary_file:
        output = PROJECT_ROOT / args.summary_file
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output, index=False)
        logger.info("wrote %s", output.relative_to(PROJECT_ROOT))


def params_for(config_params: dict[str, Any], seed_method: str, variant: str) -> GraphRAGParams:
    """Select the hyperparameters of one variant for one seed retriever.

    configs/final.yaml holds tuned settings per seed retriever and variant (§6.1);
    configs/graphrag.yaml holds one flat set of untuned defaults. G0 has no hyperparameters.

    Args:
        config_params: The `params` section of the configuration file.
        seed_method: A seed method of src/seeds.py ("bm25", "sxfmr", "hybrid_rr", ...).
        variant: G0, G1, G2 or G3.

    Returns:
        GraphRAGParams for this run.

    Raises:
        KeyError: If a G1-G3 run asks for a seed method with no tuned settings.
    """
    nested = all(isinstance(value, dict) for value in config_params.values())
    if not nested:
        return GraphRAGParams(variant=variant, **config_params)
    if variant == "G0":
        # G0 only needs the shared cutoff, so seed methods without tuned graph settings (the
        # description-only seeds of §10) can still be evaluated on their own.
        any_tuned = next(iter(config_params.values()))
        return GraphRAGParams(variant="G0", cutoff=any_tuned["G1"]["cutoff"])
    return GraphRAGParams(variant=variant, **config_params[seed_method][variant])


if __name__ == "__main__":
    main()
