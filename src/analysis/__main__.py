"""Run the full Phase 7 analysis and write tables, figures and case studies.

Implements implementation_plan.md, Phase 7. Outputs: results/analysis/*.csv,
results/analysis/case_studies.md and results/figures/*.{png,csv}.
"""

import argparse
import logging
from typing import Any

import pandas as pd

from src.analysis.breakdowns import (
    by_database_size,
    by_difficulty,
    dataset_overview,
    mismatch_drop,
    recall_curve,
)
from src.analysis.figures import line_panels
from src.analysis.joins import join_recall_table, join_table_sets, recall_by_gold_count
from src.analysis.load import (
    DATASETS,
    SEEDS,
    Records,
    load_all,
    load_dataset,
    method_name,
)
from src.analysis.significance import significance_table
from src.common import PROJECT_ROOT, load_config, setup_logging
from src.evaluate import summarise
from src.graph import schema_graph
from src.retrievers.base import split_table_id

logger = logging.getLogger(__name__)


def summaries(results: dict[tuple[str, str, str], Records]) -> pd.DataFrame:
    """Dataset-level metrics of every method, recomputed from the per-question records."""
    rows = []
    for (seed, variant, dataset), records in results.items():
        summary = summarise([r["metrics"] for r in records.values()], [5, 10, 15, 20])
        rows.append({"method": method_name(seed, variant), "dataset": dataset, **summary})
    return pd.DataFrame(rows).round(2)


def variant_index(method: str) -> int:
    """Return 0 for a seed retriever and 1-3 for GraphRAG G1-G3 (series order in figures)."""
    return 0 if "-" not in method else int(method[-1])


def _top_tables(record: dict[str, Any], k: int) -> list[str]:
    """Return the first k table ids of a record."""
    return [t for t, _ in record["ranked_tables"][:k]]


def select_cases(
    results: dict[tuple[str, str, str], Records],
    instances: list[dict[str, Any]],
    join_sets: tuple[dict, dict, int],
    dataset: str,
    seed: str,
    variant: str,
) -> dict[str, dict[str, Any] | None]:
    """Pick one success and one failure of a GraphRAG variant by a fixed rule (§7.2).

    Success: the first question (in file order) where the variant has a bridge or join-only
    table in its top 5 that the seed retriever misses. Failure: the first question where the
    seed retriever has every gold table in its top 5 and the variant does not.

    Args:
        results: Output of `load_all`.
        instances: Test instances of the dataset.
        join_sets: Output of `join_table_sets` for the dataset.
        dataset: Dataset name.
        seed: Seed retriever.
        variant: GraphRAG variant.

    Returns:
        {"success": instance or None, "failure": instance or None}.
    """
    bridges, join_only, _ = join_sets
    seed_records, variant_records = (
        results[(seed, "G0", dataset)],
        results[(seed, variant, dataset)],
    )
    cases: dict[str, dict[str, Any] | None] = {"success": None, "failure": None}
    for instance in instances:
        question_id = instance["id"]
        before, after = (
            set(_top_tables(seed_records[question_id], 5)),
            set(_top_tables(variant_records[question_id], 5)),
        )
        targets = bridges[question_id] | join_only[question_id]
        if cases["success"] is None and targets & (after - before):
            cases["success"] = instance
        gold = {f"{instance['gold_db']}.{t}" for t in instance["gold_tables"]}
        if cases["failure"] is None and gold <= before and not gold <= after:
            cases["failure"] = instance
    return cases


def case_markdown(
    results: dict[tuple[str, str, str], Records],
    dataset: str,
    seed: str,
    variant: str,
    label: str,
    instance: dict[str, Any],
) -> str:
    """Format one case: the question, the gold schema and both methods' top-5 tables."""
    lines = [f"#### {label} ({dataset}, {method_name(seed, variant)} vs. {seed})", ""]
    lines.append(f"Question: {instance['question']}")
    lines.append(f"Gold: ⟨{instance['gold_db']}, {{{', '.join(instance['gold_tables'])}}}⟩")
    for name, key in (
        (seed, (seed, "G0", dataset)),
        (method_name(seed, variant), (seed, variant, dataset)),
    ):
        tables = ", ".join(
            ".".join(split_table_id(t)) for t in _top_tables(results[key][instance["id"]], 5)
        )
        lines.append(f"- {name} top 5: {tables}")
    return "\n".join(lines) + "\n"


def _count_ticks(rows: pd.DataFrame, x: str, label: Any = str) -> dict[float, str]:
    """Tick labels that show each x value with its question count, e.g. "4+ (n=12)"."""
    counts = rows.groupby(x)["questions"].first()
    return {value: f"{label(value)}\n(n={count})" for value, count in counts.items()}


def write_figures(
    curve: pd.DataFrame, sizes: pd.DataFrame, full_schema: pd.DataFrame, figures_dir: Any
) -> None:
    """Write the three Phase 7 figures (recall@k, mAP by database size, TR@5 by gold count)."""
    for frame in (curve, sizes, full_schema):
        frame["variant_index"] = frame["method"].map(variant_index)
        frame["seed"] = frame["method"].str.split("-").str[0]
    line_panels(
        curve,
        [(f"Spider, {s.upper()} seeds", curve[curve.seed == s], None) for s in SEEDS],
        "k", "TR", ("tables retrieved (k)", "table recall@k (%)"),
        figures_dir / "recall_at_k_spider", marker_every=5,
    )  # fmt: skip
    # Database sizes are spaced evenly as categories: only sizes with at least 10 questions are
    # kept, so the kept values are not evenly spaced and their labels would overlap.
    size_values = sorted(sizes["tables_in_database"].unique())
    sizes["size_order"] = sizes["tables_in_database"].map({v: i for i, v in enumerate(size_values)})
    size_ticks = _count_ticks(sizes, "size_order", lambda order: size_values[int(order)])
    line_panels(
        sizes,
        [(f"Spider, {s.upper()} seeds", sizes[sizes.seed == s], size_ticks) for s in SEEDS],
        "size_order", "mAP", ("tables in the gold database", "table mAP (%)"),
        figures_dir / "map_by_database_size_spider",
    )  # fmt: skip
    full_schema = full_schema[full_schema.dataset.isin(["spider", "bird"])].copy()
    full_schema["gold_order"] = full_schema["gold_tables"].map({"1": 1, "2": 2, "3": 3, "4+": 4})
    names = {1: "1", 2: "2", 3: "3", 4: "4+"}
    panels = []
    for dataset in ("spider", "bird"):
        rows = full_schema[full_schema.dataset == dataset]
        ticks = _count_ticks(rows, "gold_order", names.get)
        panels += [
            (f"{dataset.capitalize()}, {s.upper()} seeds", rows[rows.seed == s], ticks)
            for s in SEEDS
        ]
    line_panels(
        full_schema, panels, "gold_order", "TR@5",
        ("gold tables per question", "table recall@5 (%)"),
        figures_dir / "table_recall_by_gold_tables",
    )  # fmt: skip


def main() -> None:
    """Run every Phase 7 analysis and write its outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/analysis")
    parser.add_argument("--figures", default="results/figures")
    args = parser.parse_args()

    setup_logging()
    output, figures_dir = PROJECT_ROOT / args.output, PROJECT_ROOT / args.figures
    output.mkdir(parents=True, exist_ok=True)
    results = load_all()
    loaded = {d: load_dataset(d, PROJECT_ROOT / "data" / "processed") for d in DATASETS}
    instances = {d: loaded[d][0] for d in DATASETS}
    schemas = {d: loaded[d][1] for d in DATASETS}

    summary = summaries(results)
    summary.to_csv(output / "summary.csv", index=False)
    significance = significance_table(results)
    significance.to_csv(output / "significance.csv", index=False)
    logger.info("GQ1: %d paired t-tests", len(significance))

    join_sets = {
        d: join_table_sets(instances[d], schemas[d], schema_graph(schemas[d])) for d in DATASETS
    }
    counts = [
        {
            "dataset": d,
            "questions with bridge tables": sum(bool(v) for v in join_sets[d][0].values()),
            "bridge tables": sum(len(v) for v in join_sets[d][0].values()),
            "join-only tables": sum(len(v) for v in join_sets[d][1].values()),
            "undetermined queries": join_sets[d][2],
        }
        for d in DATASETS
    ]
    pd.DataFrame(counts).to_csv(output / "join_table_counts.csv", index=False)
    join_recall_table(results, join_sets).to_csv(output / "join_recall.csv", index=False)
    full_schema = recall_by_gold_count(results)
    full_schema.to_csv(output / "table_recall_by_gold_tables.csv", index=False)

    by_difficulty(results, instances).to_csv(output / "by_difficulty.csv", index=False)
    sizes = by_database_size(results, instances, schemas)
    sizes.to_csv(output / "map_by_database_size.csv", index=False)
    mismatch_drop(summary, load_config("configs/reference.yaml")).to_csv(
        output / "mismatch_drop.csv", index=False
    )
    dataset_overview(summary).to_csv(output / "dataset_overview.csv", index=False)
    curve = recall_curve(results, instances)
    write_figures(curve, sizes, full_schema, figures_dir)

    cases = []
    for dataset, seed, variant in (
        ("spider", "bm25", "G1"),
        ("bird", "bm25", "G1"),
        ("spider", "sxfmr", "G2"),
    ):
        chosen = select_cases(
            results, instances[dataset], join_sets[dataset], dataset, seed, variant
        )
        for label, instance in chosen.items():
            if instance is not None:
                cases.append(
                    case_markdown(results, dataset, seed, variant, label.capitalize(), instance)
                )
    (output / "case_studies.md").write_text(
        "### Case studies (selected by a fixed rule, §7.2)\n\n" + "\n".join(cases)
    )
    logger.info("wrote analysis to %s and figures to %s", args.output, args.figures)


if __name__ == "__main__":
    main()
