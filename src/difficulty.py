"""Compute Spider hardness labels (easy / medium / hard / extra) for gold SQL queries.

Implements difficulty labelling (implementation_plan.md, §3.5) with the official Spider
evaluation code, vendored unmodified in third_party/test_suite_sql_eval (Apache-2.0).
"""

import importlib
import logging
import sys
import types
from functools import cache
from typing import Any

import nltk

from src.common import PROJECT_ROOT

logger = logging.getLogger(__name__)
VENDOR_DIR = PROJECT_ROOT / "third_party" / "test_suite_sql_eval"
NLTK_DIR = PROJECT_ROOT / "data" / "nltk_data"


@cache
def _load_spider_evaluator() -> tuple[Any, Any, Any]:
    """Import the vendored Spider evaluation modules once.

    `evaluation.py` imports `exec_eval`, which runs SQL against databases. Hardness does not
    need it, and we do not download databases (D15), so a placeholder module is registered.

    Returns:
        The (Evaluator class, Schema class, get_sql function) of the Spider evaluation code.
    """
    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    sys.modules.setdefault("exec_eval", types.ModuleType("exec_eval"))
    sys.modules["exec_eval"].eval_exec_match = None
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    evaluation = importlib.import_module("evaluation")
    process_sql = importlib.import_module("process_sql")
    return evaluation.Evaluator, process_sql.Schema, process_sql.get_sql


def spider_hardness(sql: str, database: list[dict[str, Any]]) -> str | None:
    """Return the Spider hardness label of one query.

    Args:
        sql: Gold SQL query.
        database: The query's database (schemas.json structure, lower-cased original names).

    Returns:
        "easy", "medium", "hard" or "extra", or None if Spider's parser rejects the query.
    """
    evaluator_class, schema_class, get_sql = _load_spider_evaluator()
    schema = schema_class({t["name"]: [c["name"] for c in t["columns"]] for t in database})
    try:
        parsed = get_sql(schema, sql)
    except Exception:  # The vendored parser raises bare assertion and lookup errors.
        logger.debug("Spider parser rejected: %s", sql)
        return None
    return evaluator_class().eval_hardness(parsed)
