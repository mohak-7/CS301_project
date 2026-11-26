"""Table retrievers for the baselines and for GraphRAG seeding (implementation_plan.md, §2.4).

Two settings are applied before any retriever module is imported:
- Hugging Face models are cached inside the project (cache/huggingface), not the user's home.
- PyTorch is imported before `retriv`. On macOS, `retriv`'s dependencies load a second OpenMP
  runtime; if that happens before PyTorch is loaded, PyTorch crashes with a segmentation fault.
"""

import os

from src.common import PROJECT_ROOT

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "cache" / "huggingface"))

import torch  # noqa: E402, F401  (import order matters; see module docstring)
