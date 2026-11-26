"""Assemble the Phase 6 results tables in the layout of the paper's Tables 3 and 4.

Implements §6.4 of implementation_plan.md: reported rows (from configs/reference.yaml) next to
our runs; a pooled-recall comparison with DBCopilot (D11); and the ranking-mode ablation (D13).
Writes Markdown and CSV files to results/tables/.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.common import PROJECT_ROOT, load_config, setup_logging

logger = logging.getLogger(__name__)
METRICS = ["DR@1", "DR@5", "TR@5", "TR@15"]
POOLED = {"TR@5": "pooled TR@5", "TR@15": "pooled TR@15"}
REPORTED_NAMES = {
    "bm25": "BM25",
    "sxfmr": "SXFMR",
    "crush_bm25": "CRUSH (BM25)",
    "crush_sxfmr": "CRUSH (SXFMR)",
    "bm25_finetuned": "BM25 (fine-tuned)",
    "dtr": "DTR",
    "dbcopilot": "DBCopilot",
}
OUR_METHODS = [
    "bm25", "sxfmr",
    "graphrag_bm25_G1", "graphrag_bm25_G2", "graphrag_bm25_G3",
    "graphrag_sxfmr_G1", "graphrag_sxfmr_G2", "graphrag_sxfmr_G3",
]  # fmt: skip


def our_results(tables_dir: Path) -> pd.DataFrame:
    """Load our baseline and GraphRAG (R-main) summaries into one frame.

    Args:
        tables_dir: results/tables.

    Returns:
        Frame with columns method, dataset and the metric columns.
    """
    baselines = pd.read_csv(tables_dir / "baselines_summary.csv")
    graphrag = pd.read_csv(tables_dir / "graphrag_test_summary.csv")
    main_mode = graphrag[graphrag["method"].str.endswith("_R-main")].copy()
    main_mode["method"] = main_mode["method"].str.removesuffix("_R-main")
    return pd.concat([baselines, main_mode], ignore_index=True)


def results_table(
    ours: pd.DataFrame, reference: dict, datasets: list[str], pooled: bool
) -> pd.DataFrame:
    """Build one results table: reported rows first, then our rows.

    Args:
        ours: Output of `our_results`.
        reference: Paper numbers (configs/reference.yaml).
        datasets: Column groups of the table.
        pooled: Use pooled table recall for our rows (to compare with DBCopilot's row, D11);
            otherwise per-question table recall (as in the paper's baseline rows).

    Returns:
        Frame indexed by (source, method) with one column per (dataset, metric).
    """
    rows = {}
    for key, name in REPORTED_NAMES.items():
        if pooled and key != "dbcopilot":
            continue
        rows[("reported", name)] = {(d, m): reference[d][key][m] for d in datasets for m in METRICS}
    for method in OUR_METHODS:
        values = {}
        for dataset in datasets:
            row = ours[(ours["method"] == method) & (ours["dataset"] == dataset)].iloc[0]
            for metric in METRICS:
                column = POOLED.get(metric, metric) if pooled else metric
                values[(dataset, metric)] = round(float(row[column]), 2)
        rows[("ours", method)] = values
    return pd.DataFrame.from_dict(rows, orient="index")


def to_markdown(table: pd.DataFrame, title: str, note: str) -> str:
    """Format a results table as Markdown, marking the best value of each column in bold.

    Args:
        table: Output of `results_table`.
        title: Heading line.
        note: Explanation printed under the table.

    Returns:
        Markdown text.
    """
    header = "| Source | Method | " + " | ".join(f"{d} {m}" for d, m in table.columns) + " |"
    lines = [f"### {title}", "", header, "|---|---|" + "---|" * len(table.columns)]
    best = table.max()
    for (source, method), row in table.iterrows():
        cells = [f"**{v:.2f}**" if v == best[c] else f"{v:.2f}" for c, v in row.items()]
        lines.append(f"| {source} | {method} | " + " | ".join(cells) + " |")
    return "\n".join([*lines, "", note, ""])


def ranking_mode_table(tables_dir: Path, datasets: list[str]) -> pd.DataFrame:
    """Build the ranking-mode ablation: DR@1 and TR@5 of each variant under each mode (D13).

    Args:
        tables_dir: results/tables.
        datasets: Datasets to include.

    Returns:
        Frame indexed by (method, mode) with (dataset, metric) columns.
    """
    graphrag = pd.read_csv(tables_dir / "graphrag_test_summary.csv")
    graphrag[["base", "mode"]] = graphrag["method"].str.extract(r"^(graphrag_\w+_G\d)_(R-.+)$")
    table = graphrag[graphrag["dataset"].isin(datasets)].pivot_table(
        index=["base", "mode"], columns="dataset", values=["DR@1", "TR@5"]
    )
    table = table.swaplevel(axis=1)[[(d, m) for d in datasets for m in ("DR@1", "TR@5")]]
    table.columns = [f"{dataset} {metric}" for dataset, metric in table.columns]
    table.index = [f"{base.removeprefix('graphrag_')} {mode}" for base, mode in table.index]
    return table.round(2)


def main() -> None:
    """Write the main, pooled and ranking-mode tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="configs/reference.yaml")
    parser.add_argument("--tables-dir", default="results/tables")
    args = parser.parse_args()

    setup_logging()
    tables_dir = PROJECT_ROOT / args.tables_dir
    reference = load_config(args.reference)
    ours = our_results(tables_dir)
    groups = {
        "regular": ["spider", "bird", "fiben"],
        "robustness": ["spider_syn", "spider_realistic"],
    }
    sections = []
    for name, datasets in groups.items():
        per_question = results_table(ours, reference, datasets, pooled=False)
        per_question.to_csv(tables_dir / f"main_{name}.csv")
        sections.append(
            to_markdown(
                per_question,
                f"Schema routing, {name} test sets (per-question table recall)",
                "Reported rows: the paper's Tables 3 and 4. The DBCopilot row uses pooled table "
                "recall (D11); compare it with the pooled table below.",
            )
        )
        pooled = results_table(ours, reference, datasets, pooled=True)
        pooled.to_csv(tables_dir / f"pooled_{name}.csv")
        sections.append(
            to_markdown(
                pooled,
                f"Pooled table recall vs. DBCopilot, {name} test sets",
                "Our rows use pooled table recall, the definition behind DBCopilot's reported "
                "numbers (§2.3, D11). Database recall is identical under both definitions.",
            )
        )
    modes = ranking_mode_table(tables_dir, [d for ds in groups.values() for d in ds])
    modes.to_csv(tables_dir / "ranking_modes.csv")
    sections.append("### Ranking-mode ablation (D13)\n\n" + modes.to_markdown() + "\n")
    (tables_dir / "main_results.md").write_text("\n".join(sections))
    logger.info("wrote %s", (tables_dir / "main_results.md").relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
