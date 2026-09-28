"""Deterministic train/test splits over Who&When files."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Sequence, Tuple



def list_subset_files(root: os.PathLike | str) -> List[str]:
    """Return the subset's JSON filenames sorted numerically (``1.json`` first)."""
    files = [f for f in os.listdir(root) if f.endswith(".json")]
    return sorted(files, key=lambda x: int("".join(filter(str.isdigit, x)) or 0))


def make_split(
    root: os.PathLike | str, train_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Split a subset's files into train / test filename lists.

    All methods in the paper share one split per subset: ``train_ratio`` of the
    trajectories are available for training (only the supervised baselines and
    MASC's projection layers use them) and the rest are held out for testing.

    The filenames are sorted before shuffling so the split depends only on
    ``seed`` -- ``os.listdir`` order is filesystem-dependent and would otherwise
    make the split unreproducible across machines.
    """
    files = sorted(list_subset_files(root))
    rng = random.Random(seed)
    rng.shuffle(files)
    cut = int(len(files) * train_ratio)
    return files[:cut], files[cut:]


def save_split(path: os.PathLike | str, files: Sequence[str]) -> None:
    """Write a filename list as JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(list(files), handle, indent=2)


def load_split(path: os.PathLike | str) -> List[str]:
    """Read a filename list written by :func:`save_split`."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
