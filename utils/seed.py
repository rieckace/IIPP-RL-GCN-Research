"""Reproducibility helpers for research experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for a reproducible run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_rng(seed: int) -> random.Random:
    """Return an isolated Python RNG for deterministic experiment choices."""
    return random.Random(seed)
