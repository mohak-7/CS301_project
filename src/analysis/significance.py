"""GQ1: paired significance tests of GraphRAG variants against their seed retriever.

Implements implementation_plan.md, §7.1 and D10: the authors' test (`scipy.stats.ttest_rel`) on
per-question scores, only between methods we ran ourselves. DBCopilot has no per-question
results, so it cannot be tested.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.load import DATASETS, SEEDS, Records, method_name

METRICS = ["DR@1", "DR@5", "TR@5", "TR@15"]


def paired_test(baseline: Records, candidate: Records, metric: str) -> tuple[float, float]:
    """Return the mean difference (in points) and the paired t-test p-value for one metric.

    Args:
        baseline: Records of the seed retriever.
        candidate: Records of a GraphRAG variant on the same questions.
        metric: Per-question metric name.

    Returns:
        (mean of candidate - baseline, in percentage points; two-sided p-value). The p-value is
        1.0 when every difference is zero, where the t statistic is undefined.
    """
    ids = list(baseline)
    before = np.array([baseline[i]["metrics"][metric] for i in ids])
    after = np.array([candidate[i]["metrics"][metric] for i in ids])
    differences = after - before
    if not differences.any():
        return 0.0, 1.0
    return 100 * float(differences.mean()), float(stats.ttest_rel(after, before).pvalue)


def significance_table(results: dict[tuple[str, str, str], Records]) -> pd.DataFrame:
    """Test every GraphRAG variant against its seed retriever on every dataset and metric.

    Args:
        results: Output of `load_all`.

    Returns:
        One row per (dataset, method, metric) with the difference, p-value and significance
        marks (* p < 0.05, ** p < 0.01, as in the paper's tables). No correction for multiple
        comparisons is applied, matching the paper; the number of tests is reported.
    """
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            baseline = results[(seed, "G0", dataset)]
            for variant in ("G1", "G2", "G3"):
                candidate = results[(seed, variant, dataset)]
                for metric in METRICS:
                    difference, p_value = paired_test(baseline, candidate, metric)
                    mark = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method_name(seed, variant),
                            "metric": metric,
                            "difference": round(difference, 2),
                            "p_value": p_value,
                            "significance": mark,
                        }
                    )
    return pd.DataFrame(rows)
