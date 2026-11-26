"""Build the processed benchmark files: schemas.json, test.jsonl and train.jsonl.

Implements Phase 3 data preparation (implementation_plan.md, §2.1, §2.2, §3.3–§3.5, D12, D15),
following the authors' scripts/prepare_data.py on public sources.
"""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlglot.errors import SqlglotError
from tqdm import tqdm

from src.common import PROJECT_ROOT, load_config, read_json, setup_logging, write_json, write_jsonl
from src.difficulty import spider_hardness
from src.extract_gold import extract_gold_tables
from src.schemas import SchemaCollection, build_schemas, fiben_tables_json

logger = logging.getLogger(__name__)

# A raw question as (question, gold_sql, gold_db, difficulty or None).
RawQuestion = tuple[str, str, str, str | None]


def build_instances(
    dataset: str,
    raw_questions: list[RawQuestion],
    schemas: SchemaCollection,
    spider_labels: bool,
) -> list[dict[str, Any]]:
    """Turn raw questions into processed instances with gold tables and difficulty.

    Questions whose SQL cannot be parsed are dropped, as in the authors' script for Bird and
    Fiben (§2.2 rule 4). For the Spider family the authors' script has no fallback; we drop and
    log instead, and the size check in check_data.py shows whether any were lost.

    Args:
        dataset: Dataset name, used in instance ids.
        raw_questions: Raw questions in source-file order.
        schemas: Schema collection the gold databases belong to.
        spider_labels: Compute Spider hardness labels instead of using the given difficulty.

    Returns:
        Processed instances; `source_index` keeps the position in the source file.
    """
    instances, dropped = [], 0
    for source_index, (question, sql, db_id, difficulty) in enumerate(
        tqdm(raw_questions, desc=dataset)
    ):
        try:
            gold_columns = extract_gold_tables(sql, schemas[db_id])
        except SqlglotError:
            dropped += 1
            continue
        if spider_labels:
            difficulty = spider_hardness(sql, schemas[db_id])
        instances.append(
            {
                "id": f"{dataset}-{source_index}",
                "source_index": source_index,
                "question": question,
                "gold_sql": sql,
                "gold_db": db_id,
                "gold_tables": list(gold_columns),
                "gold_columns": gold_columns,
                "difficulty": difficulty,
            }
        )
    logger.info("%s: kept %d, dropped %d unparsable", dataset, len(instances), dropped)
    return instances


def spider_sources(raw_dir: Path) -> dict[str, Any]:
    """Load Spider's schema and its dev and train questions.

    Args:
        raw_dir: Root of the raw data.

    Returns:
        Dict with "schemas", "test" and "train" raw-question lists.
    """
    dev = read_json(raw_dir / "spider" / "dev.json")
    # Spider-Syn's train file keeps the original Spider question in `SpiderQuestion`. It covers
    # train_spider.json (7000 questions, as in the paper's Table 2); used for tuning only.
    train = read_json(raw_dir / "spider_syn" / "train_spider.json")
    return {
        "schemas": build_schemas(read_json(raw_dir / "spider" / "tables.json")),
        "test": [(r["question"], r["query"], r["db_id"], None) for r in dev],
        "train": [(r["SpiderQuestion"], r["query"], r["db_id"], None) for r in train],
    }


def spider_syn_sources(raw_dir: Path) -> dict[str, Any]:
    """Load Spider-syn: Spider's schema with synonym-substituted dev questions.

    Args:
        raw_dir: Root of the raw data.

    Returns:
        Dict with "schemas" and "test".
    """
    dev = read_json(raw_dir / "spider_syn" / "dev.json")
    return {
        "schemas": build_schemas(read_json(raw_dir / "spider" / "tables.json")),
        "test": [(r["SpiderSynQuestion"], r["query"], r["db_id"], None) for r in dev],
    }


def spider_realistic_sources(raw_dir: Path) -> dict[str, Any]:
    """Load Spider-realistic: Spider's schema with column mentions removed from questions.

    Args:
        raw_dir: Root of the raw data.

    Returns:
        Dict with "schemas" and "test".
    """
    dev = read_json(raw_dir / "spider_realistic" / "spider-realistic.json")
    return {
        "schemas": build_schemas(read_json(raw_dir / "spider" / "tables.json")),
        "test": [(r["question"], r["query"], r["db_id"], None) for r in dev],
    }


def bird_sources(raw_dir: Path) -> dict[str, Any]:
    """Load Bird: training and dev databases form one collection of 80 (§2.1).

    The dev questions are the official June 2024 revision; no complete 2023 copy is public (D15).
    Bird's `evidence` field is not used: routing takes only the question, as in the paper.

    Args:
        raw_dir: Root of the raw data.

    Returns:
        Dict with "schemas", "test" and "train".
    """
    bird_dir = raw_dir / "bird"
    raw_tables = read_json(bird_dir / "train_tables.json") + read_json(bird_dir / "dev_tables.json")
    dev = read_json(bird_dir / "dev_2024.json")
    train = read_json(bird_dir / "train.json")
    return {
        "schemas": build_schemas(raw_tables),
        "test": [(r["question"], r["SQL"], r["db_id"], r["difficulty"]) for r in dev],
        "train": [(r["question"], r["SQL"], r["db_id"], None) for r in train],
    }


def _fiben_sqlite_query(sql: str) -> str:
    """Rewrite a Fiben (DB2) query for SQLite, as UNITE's prepare_fiben.py does.

    Args:
        sql: Original FIBEN query.

    Returns:
        Query without the FIBEN schema prefix and with FETCH FIRST n ROWS ONLY as LIMIT n.
    """
    query = sql.replace("FIBEN.", "")
    if "FETCH" in query:
        limit = query.split("FETCH FIRST")[1].split("ROWS ONLY")[0].strip()
        query = query[: query.index("FETCH")] + f"LIMIT {limit}"
    return query


def fiben_sources(raw_dir: Path) -> dict[str, Any]:
    """Load Fiben: one financial database (152 tables) and 300 questions.

    Names are split into words with wordninja (the authors do this for Fiben only).

    Args:
        raw_dir: Root of the raw data.

    Returns:
        Dict with "schemas" and "test".
    """
    fiben_dir = raw_dir / "fiben"
    raw_tables = fiben_tables_json(fiben_dir / "FIBEN.sql")
    queries = read_json(fiben_dir / "FIBEN_Queries.json")
    return {
        "schemas": build_schemas(raw_tables, use_wordninja=True),
        "test": [
            (q["question"].strip(), _fiben_sqlite_query(q["SQL"]).strip(), "fiben", None)
            for q in queries
        ],
    }


DATASETS: dict[str, tuple[Callable[[Path], dict[str, Any]], bool]] = {
    # name: (source loader, compute Spider hardness labels)
    "spider": (spider_sources, True),
    "spider_syn": (spider_syn_sources, True),
    "spider_realistic": (spider_realistic_sources, True),
    "bird": (bird_sources, False),
    "fiben": (fiben_sources, False),
}


def prepare_dataset(name: str, raw_dir: Path, processed_dir: Path) -> None:
    """Prepare one dataset and write its processed files.

    Args:
        name: Dataset name (a key of DATASETS).
        raw_dir: Root of the raw data.
        processed_dir: Root of the processed data.
    """
    loader, spider_labels = DATASETS[name]
    sources = loader(raw_dir)
    out_dir = processed_dir / name
    write_json(out_dir / "schemas.json", sources["schemas"])
    for split in ("test", "train"):
        if split in sources:
            instances = build_instances(name, sources[split], sources["schemas"], spider_labels)
            write_jsonl(out_dir / f"{split}.jsonl", instances)


def main() -> None:
    """Prepare all (or selected) datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", help="data config")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    for name in args.datasets:
        prepare_dataset(
            name, PROJECT_ROOT / config["raw_dir"], PROJECT_ROOT / config["processed_dir"]
        )


if __name__ == "__main__":
    main()
