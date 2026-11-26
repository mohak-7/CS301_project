"""Summary figure: pooled table recall@5 after each step of our method, against DBCopilot.

Implements the summary figure of the report (implementation_plan.md, §8 and §10). Each bar is a
fixed configuration, the same on every dataset (no per-dataset choice on test data): the dense
baseline, GraphRAG on dense seeds (G2, Phase 6), hybrid reranked seeds with G1 (Phase 9) and
description-based reranked seeds with G1 (Phase 10). The steps are ordered, so they use one blue
ramp, light to dark (ordinal steps 250/400/550/700 of the reference palette); DBCopilot's reported
value is a black tick. A legend and a CSV table view accompany the figure.
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.figures import SURFACE, TEXT, TEXT_SECONDARY, _style_axis  # noqa: E402
from src.common import PROJECT_ROOT, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)
STEPS = [
    ("sxfmr", "Dense baseline (SXFMR)"),
    ("sxfmr-G2", "+ graph (G2)"),
    ("hybrid_rr-G1", "+ hybrid seeds and reranking"),
    ("hybrid_desc_rr-G1", "+ LLM table descriptions = GraphRAG"),
]
STEP_COLOURS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
REFERENCE = "DBCopilot (reported)"
DATASETS = [
    ("spider", "Spider"), ("bird", "Bird"), ("fiben", "Fiben"),
    ("spider_syn", "Spider-syn"), ("spider_realistic", "Spider-realistic"),
]  # fmt: skip
METRIC = "pooled TR@5"
BAR_WIDTH = 0.19
GAP = 0.02  # surface gap between adjacent bars


def main() -> None:
    """Write results/figures/progress_ptr5.{png,csv}."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/tables/descriptions_summary.csv")
    parser.add_argument("--output", default="results/figures/progress_ptr5")
    args = parser.parse_args()
    setup_logging()

    summary = pd.read_csv(PROJECT_ROOT / args.summary).drop_duplicates(["dataset", "method"])
    wanted = [key for key, _ in STEPS] + [REFERENCE]
    table = summary[summary["method"].isin(wanted)].pivot(
        index="dataset", columns="method", values=METRIC
    )
    table = table.loc[[key for key, _ in DATASETS], wanted]

    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    figure.patch.set_facecolor(SURFACE)
    _style_axis(axis, "", "", "Pooled table recall@5")
    for index, ((key, label), colour) in enumerate(zip(STEPS, STEP_COLOURS, strict=True)):
        offsets = [i + (index - 1.5) * BAR_WIDTH for i in range(len(DATASETS))]
        axis.bar(offsets, table[key], width=BAR_WIDTH - GAP, color=colour, label=label)
    span = 2 * BAR_WIDTH * 2
    for position, value in enumerate(table[REFERENCE]):
        axis.hlines(value, position - span / 2, position + span / 2, color=TEXT, linewidth=2,
                    label=REFERENCE if position == 0 else None)  # fmt: skip
    axis.set_xticks(range(len(DATASETS)), [name for _, name in DATASETS])
    axis.set_ylim(0, 100)
    handles, labels = axis.get_legend_handles_labels()
    # The reference line is drawn first but listed last, so the legend reads in step order.
    order = [labels.index(label) for _, label in STEPS] + [labels.index(REFERENCE)]
    axis.legend(
        [handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=7,
        ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), labelcolor=TEXT_SECONDARY,
    )  # fmt: skip
    figure.tight_layout()
    output = PROJECT_ROOT / args.output
    figure.savefig(output.with_suffix(".png"), dpi=200, facecolor=SURFACE)
    table.round(2).to_csv(output.with_suffix(".csv"))
    logger.info("wrote %s.png and .csv", output.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
