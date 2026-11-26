"""Tune GraphRAG hyperparameters on training questions and freeze them in configs/final.yaml.

Implements Phase 6 tuning (implementation_plan.md, §5.5, §6.1, D7): a staged grid search
(G1, then G2, then G3 with G2's settings fixed) on random samples of Spider and Bird training
questions, ranked by per-question table recall@5 with database recall@1 as the tie-breaker.
No test question is read.
"""

import argparse
import datetime
import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.common import PROJECT_ROOT, load_config, set_seed, setup_logging
from src.evaluate import question_metrics, summarise
from src.rank_databases import rank_databases, top_tables
from src.retrievers.base import ScoreMatrix
from src.retrievers.graphrag import GraphRAGParams, graphrag_scores
from src.run_graphrag import load_inputs

logger = logging.getLogger(__name__)
TuningSet = list[tuple[list[dict[str, Any]], ScoreMatrix, Any]]  # (instances, seeds, graph)


def sample_training_sets(
    seed_method: str, config: dict[str, Any], baseline_config: dict[str, Any]
) -> TuningSet:
    """Draw a fixed random sample of training questions from each tuning dataset.

    Args:
        seed_method: "bm25" or "sxfmr".
        config: Tuning configuration.
        baseline_config: configs/baselines.yaml.

    Returns:
        One (instances, seed matrix, table graph) triple per dataset.
    """
    tuning_set = []
    for dataset in config["datasets"]:
        instances, seeds, graph = load_inputs(dataset, seed_method, "train", baseline_config)
        # The same rows are drawn for both seed methods, because the generator is re-seeded.
        rows = np.sort(
            np.random.default_rng(config["seed"]).choice(
                len(instances), config["sample_size"], replace=False
            )
        )
        subset = ScoreMatrix(
            [seeds.question_ids[r] for r in rows], seeds.table_ids, seeds.scores[rows]
        )
        tuning_set.append(([instances[r] for r in rows], subset, graph))
    return tuning_set


def score_setting(
    params: GraphRAGParams, tuning_set: TuningSet, metrics: dict[str, Any]
) -> dict[str, float]:
    """Evaluate one setting with the paper's ranking rule (R-main) on every tuning dataset.

    Args:
        params: GraphRAG hyperparameters.
        tuning_set: Output of `sample_training_sets`.
        metrics: Metric cutoffs from configs/baselines.yaml.

    Returns:
        Per-question TR@5 and DR@1, averaged over the datasets (equal weight per dataset).
    """
    per_dataset = []
    for instances, seeds, graph in tuning_set:
        final = graphrag_scores(seeds, graph, params)
        results = []
        for row, instance in enumerate(instances):
            ranked_tables = top_tables(final.scores[row], final.table_ids, params.cutoff)
            results.append(
                question_metrics(
                    instance,
                    rank_databases(ranked_tables),
                    ranked_tables,
                    metrics["database_ks"],
                    metrics["table_ks"],
                    metrics["full_schema_ks"],
                )
            )
        per_dataset.append(summarise(results, metrics["table_ks"]))
    return {key: float(np.mean([s[key] for s in per_dataset])) for key in ("TR@5", "DR@1", "FS@15")}


def search_stage(
    variant: str,
    grid: dict[str, list[Any]],
    base: dict[str, Any],
    tuning_set: TuningSet,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate every combination of one stage's grid, other settings held at `base`.

    Args:
        variant: G1, G2 or G3.
        grid: Values to try for each hyperparameter of this stage.
        base: Fixed settings (and the best settings of earlier stages).
        tuning_set: Output of `sample_training_sets`.
        metrics: Metric cutoffs.

    Returns:
        One row per combination: the settings and their scores.
    """
    names = list(grid)
    rows = []
    for values in itertools.product(*(grid[name] for name in names)):
        setting = dict(zip(names, values, strict=True))
        scores = score_setting(
            GraphRAGParams(variant=variant, **base, **setting), tuning_set, metrics
        )
        rows.append({"variant": variant, **setting, **scores})
    return rows


def best_setting(
    rows: list[dict[str, Any]], grid: dict[str, list[Any]], objective: str, tie_breaker: str
) -> dict[str, Any]:
    """Pick the best combination: highest objective, then highest tie-breaker, then grid order.

    Args:
        rows: Output of `search_stage`.
        grid: The stage's grid (to know which keys are settings).
        objective: Metric to maximise.
        tie_breaker: Metric used when the objective ties.

    Returns:
        The chosen settings of this stage.
    """
    best = max(rows, key=lambda row: (round(row[objective], 6), round(row[tie_breaker], 6)))
    return {name: best[name] for name in grid}


def tune_seed_method(
    seed_method: str, config: dict[str, Any], baseline_config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Run the three tuning stages for one seed retriever.

    Args:
        seed_method: "bm25" or "sxfmr".
        config: Tuning configuration.
        baseline_config: configs/baselines.yaml.

    Returns:
        (final settings per variant, all evaluated rows).
    """
    tuning_set = sample_training_sets(seed_method, config, baseline_config)
    metrics, fixed, grid = baseline_config["metrics"], config["fixed"], config["grid"]
    objective, tie_breaker = config["objective"], config["tie_breaker"]
    g0_scores = score_setting(GraphRAGParams(variant="G0", **fixed), tuning_set, metrics)
    log = [{"variant": "G0", **g0_scores}]
    logger.info("%s G0: TR@5 %.2f, DR@1 %.2f", seed_method, g0_scores["TR@5"], g0_scores["DR@1"])

    def run_stage(variant: str, base: dict[str, Any]) -> dict[str, Any]:
        rows = search_stage(variant, grid[variant], base, tuning_set, metrics)
        log.extend(rows)
        chosen = best_setting(rows, grid[variant], objective, tie_breaker)
        best = next(row for row in rows if all(row[k] == v for k, v in chosen.items()))
        logger.info(
            "%s %s: best %s -> TR@5 %.2f, DR@1 %.2f",
            seed_method,
            variant,
            chosen,
            best["TR@5"],
            best["DR@1"],
        )
        return chosen

    g1 = run_stage("G1", fixed)
    g2 = run_stage("G2", fixed)
    # G3 starts from G2's best settings and tunes only the connector settings (§5.5).
    g3 = {**g2, **run_stage("G3", {**fixed, **g2})}
    return {"G1": g1, "G2": g2, "G3": g3}, log


def main() -> None:
    """Tune every seed retriever, save the search log and write configs/final.yaml."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tuning.yaml")
    parser.add_argument("--baseline-config", default="configs/baselines.yaml")
    parser.add_argument("--ranking-config", default="configs/graphrag.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    baseline_config = load_config(args.baseline_config)
    set_seed(config["seed"])
    results_dir = PROJECT_ROOT / config["results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    output = PROJECT_ROOT / config["output"]
    # With `merge`, settings of other seed methods already frozen in the output are kept
    # unchanged; only the seed methods tuned in this run are added (extension, §9).
    final: dict[str, Any] = {"params": {}}
    if config.get("merge") and output.exists():
        final = yaml.safe_load(output.read_text())
    for seed_method in config["seed_methods"]:
        chosen, log = tune_seed_method(seed_method, config, baseline_config)
        pd.DataFrame(log).round(4).to_csv(results_dir / f"{seed_method}.csv", index=False)
        final["params"][seed_method] = {
            variant: {**config["fixed"], **settings} for variant, settings in chosen.items()
        }
    final["ranking"] = load_config(args.ranking_config)["ranking"]
    frozen = {
        "date": datetime.date.today().isoformat(),
        "note": "Tuned on training questions only (D7); frozen before any GraphRAG test run.",
        "sample_size": config["sample_size"],
        "datasets": config["datasets"],
    }
    if config.get("merge") and "frozen" in final:
        final.setdefault("frozen_extension", {})[",".join(config["seed_methods"])] = frozen
    else:
        final["frozen"] = frozen
    output.write_text(
        "# Frozen GraphRAG settings, written by src/tune.py (implementation_plan.md, §6.1).\n"
        + yaml.safe_dump(final, sort_keys=False)
    )
    logger.info("wrote %s", output.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
