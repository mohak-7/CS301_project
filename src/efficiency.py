"""Measure each method's efficiency in the units of the paper's Table 5.

Uses the Spider test set by default. Implements §6.3 of implementation_plan.md. Each method
runs in a fresh process, so peak memory is measured per method. Build time covers index
construction (and, for GraphRAG, the table graph); query time covers scoring all questions in
batches of 64 and ranking them.
"""

import argparse
import json
import logging
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.common import PROJECT_ROOT, load_config, read_json, read_jsonl, setup_logging
from src.graph import schema_graph
from src.rank_databases import rank_databases, top_tables
from src.retrievers.base import question_queries, results_to_matrix, table_documents
from src.retrievers.bm25 import build_bm25_index
from src.retrievers.dense import build_dense_index
from src.retrievers.graphrag import graphrag_scores
from src.run_graphrag import params_for

logger = logging.getLogger(__name__)
METHODS = [
    "bm25", "sxfmr",
    "graphrag_bm25_G1", "graphrag_bm25_G2", "graphrag_bm25_G3",
    "graphrag_sxfmr_G1", "graphrag_sxfmr_G2", "graphrag_sxfmr_G3",
]  # fmt: skip
MODEL_DIR = "cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2"


def directory_size_mb(path: Path) -> float:
    """Return the total size of the files under `path`, in MB (0 if it does not exist).

    Links are followed and each distinct target file is counted once: the Hugging Face cache
    keeps model files in a shared blob folder and links to them from the model's folder.
    """
    if not path.exists():
        return 0.0
    files = {f.resolve() for f in path.rglob("*") if f.is_file()}
    return sum(f.stat().st_size for f in files) / 1e6


def build_retriever(
    seed_method: str, documents: list[dict[str, str]], index_name: str, config: dict[str, Any]
) -> Any:
    """Build a fresh retriv index for the seed retriever.

    Args:
        seed_method: "bm25" or "sxfmr".
        documents: Table documents.
        index_name: Name of a new, temporary index.
        config: configs/baselines.yaml.

    Returns:
        The indexed retriv retriever.
    """
    settings = config["methods"][seed_method]
    retriv_dir = PROJECT_ROOT / config["retriv_dir"] / "efficiency"
    if seed_method == "bm25":
        return build_bm25_index(documents, index_name, retriv_dir, settings["k1"], settings["b"])
    return build_dense_index(
        documents,
        index_name,
        retriv_dir,
        settings["model"],
        settings["max_length"],
        settings["normalize"],
        settings["batch_size"],
    )


def measure(method: str, dataset: str, batch_size: int) -> dict[str, Any]:
    """Build, query and rank with one method, recording time, disk and peak memory.

    Args:
        method: One of METHODS.
        dataset: Dataset whose test questions are used.
        batch_size: Query batch size (64, as in the paper's Table 5).

    Returns:
        Dict with build seconds, query seconds, queries per second, disk MB and peak RAM MB.
    """
    config = load_config("configs/baselines.yaml")
    seed_method = "bm25" if "bm25" in method else "sxfmr"
    processed = PROJECT_ROOT / config["processed_dir"] / dataset
    schemas = read_json(processed / "schemas.json")
    instances = read_jsonl(processed / "test.jsonl")
    documents = table_documents(schemas)
    retriv_dir = PROJECT_ROOT / config["retriv_dir"] / "efficiency"
    index_name = f"{dataset}_{method}"

    started = time.perf_counter()
    retriever = build_retriever(seed_method, documents, index_name, config)
    graph = schema_graph(schemas) if method.startswith("graphrag") else None
    build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    results = retriever.bsearch(
        question_queries(instances),
        cutoff=len(documents),
        batch_size=batch_size,
        show_progress=False,
    )
    matrix = results_to_matrix(results, [i["id"] for i in instances], [d["id"] for d in documents])
    if graph is not None:
        variant = method.rsplit("_", 1)[1]
        params = params_for(load_config("configs/final.yaml")["params"], seed_method, variant)
        matrix = graphrag_scores(matrix, graph, params)
    for row in range(len(instances)):
        rank_databases(top_tables(matrix.scores[row], matrix.table_ids, config["cutoff"]))
    query_seconds = time.perf_counter() - started

    disk = directory_size_mb(retriv_dir / "collections" / index_name)
    if seed_method == "sxfmr":
        disk += directory_size_mb(PROJECT_ROOT / MODEL_DIR)  # the paper counts the model on disk
    shutil.rmtree(retriv_dir / "collections" / index_name, ignore_errors=True)
    return {
        "method": method,
        "build_s": round(build_seconds, 1),
        "query_s": round(query_seconds, 2),
        "QPS": round(len(instances) / query_seconds, 1),
        "disk_MB": round(disk, 1),
        # macOS reports ru_maxrss in bytes.
        "peak_RAM_MB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
    }


def main() -> None:
    """Measure one method (child process) or all methods (each in its own process)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, help="measure one method and print JSON")
    parser.add_argument("--dataset", default="spider")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="results/tables/efficiency.csv")
    args = parser.parse_args()

    setup_logging()
    if args.method:
        sys.stdout.write(json.dumps(measure(args.method, args.dataset, args.batch_size)) + "\n")
        return
    rows = []
    for method in METHODS:
        command = [
            sys.executable,
            "-m",
            "src.efficiency",
            "--method",
            method,
            "--dataset",
            args.dataset,
        ]
        output = subprocess.run(
            command, capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
        )
        rows.append(json.loads(output.stdout.strip().splitlines()[-1]))
        logger.info("%s", rows[-1])
    table = pd.DataFrame(rows)
    table.to_csv(PROJECT_ROOT / args.output, index=False)
    logger.info("efficiency on %s:\n%s", args.dataset, table.to_string(index=False))


if __name__ == "__main__":
    main()
