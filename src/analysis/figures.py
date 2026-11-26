"""Static report figures for Phase 7 (PNG, each with a CSV of the plotted data).

Implements the figures of implementation_plan.md, §7.2-§7.3. Colours are the first slots of the
validated reference categorical palette, in fixed order (G0 = seed retriever, then G1, G2); each
series also has its own marker shape, and every figure has a legend and a CSV table view, so
identity never depends on colour alone. G3 is not drawn: it is within 0.5 points of G2 on every
dataset (§6.5) and its line would hide G2's; its values are in the CSV files.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (slots 1-3)
MARKERS = ["o", "s", "^"]
SURFACE, TEXT, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
VARIANT_LABELS = ["seed retriever (G0)", "GraphRAG G1", "GraphRAG G2"]


def _style_axis(axis: plt.Axes, title: str, x_label: str, y_label: str) -> None:
    """Apply recessive axes: light horizontal grid, no top/right spines, secondary-ink text."""
    axis.set_facecolor(SURFACE)
    axis.set_title(title, color=TEXT, fontsize=11, loc="left")
    axis.set_xlabel(x_label, color=TEXT_SECONDARY, fontsize=9)
    axis.set_ylabel(y_label, color=TEXT_SECONDARY, fontsize=9)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def line_panels(
    data: pd.DataFrame,
    panels: list[tuple[str, pd.DataFrame, dict[float, str] | None]],
    x: str,
    y: str,
    labels: tuple[str, str],
    path: Path,
    columns: int = 2,
    marker_every: int = 1,
) -> None:
    """Draw one line chart per panel with three series (G0-G2) and save PNG + CSV.

    Args:
        data: All plotted rows, including G3 (saved as the CSV table view).
        panels: (panel title, rows of that panel, optional x tick labels); rows have
            `variant_index` 0-3, `x` and `y`.
        x: Column on the x axis.
        y: Column on the y axis.
        labels: (x-axis label, y-axis label).
        path: Output path without extension.
        columns: Panels per row.
        marker_every: Draw a marker on every n-th point (dense curves stay readable).
    """
    rows_needed = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(
        rows_needed, columns, figsize=(4.8 * columns, 3.4 * rows_needed), sharey=True, squeeze=False
    )
    figure.patch.set_facecolor(SURFACE)
    for axis, (title, rows, ticks) in zip(axes.flat, panels, strict=True):
        for index in range(len(SERIES_COLOURS)):
            series = rows[rows["variant_index"] == index].sort_values(x)
            axis.plot(
                series[x], series[y], color=SERIES_COLOURS[index], linewidth=2,
                marker=MARKERS[index], markersize=6, markevery=marker_every,
                markeredgecolor=SURFACE, markeredgewidth=1, label=VARIANT_LABELS[index],
            )  # fmt: skip
        if ticks:
            axis.set_xticks(list(ticks), list(ticks.values()))
        _style_axis(axis, title, *labels)
    handles, names = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=3, frameon=False, fontsize=8,
                  labelcolor=TEXT_SECONDARY)  # fmt: skip
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=200, facecolor=SURFACE)
    plt.close(figure)
    data.to_csv(path.with_suffix(".csv"), index=False)
