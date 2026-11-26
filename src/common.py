"""Shared helpers: random seed, configuration loading, logging and JSON I/O.

Implements the central reproducibility rules of implementation_plan.md, §S.3.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SEED = 42  # Same seed as the authors' scripts (scripts/retrival.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy random number generators.

    Args:
        seed: Seed value shared by all experiments.
    """
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file, absolute or relative to the project root.

    Returns:
        Parsed configuration as a nested dictionary.
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open() as handle:
        return yaml.safe_load(handle)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a concise log format for pipeline stage summaries.

    Args:
        level: Logging level for the root logger.
    """
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Model downloads log every HTTP request at INFO level; keep only warnings from them.
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def read_json(path: Path) -> Any:
    """Read a JSON file.

    Args:
        path: File to read.

    Returns:
        The decoded JSON value.
    """
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    """Write a value as indented JSON, creating parent directories.

    Args:
        path: Destination file.
        data: JSON-serialisable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file.

    Args:
        path: File with one JSON object per line.

    Returns:
        List of decoded records.
    """
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as JSON Lines, creating parent directories.

    Args:
        path: Destination file.
        records: JSON-serialisable dictionaries, one per line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
