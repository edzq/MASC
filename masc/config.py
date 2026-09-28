"""YAML-backed experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from masc.encoder import DEFAULT_ENCODER


@dataclass
class Config:
    """Everything needed to train and evaluate the detector.

    Defaults reproduce the configuration reported in the paper: an
    ``all-MiniLM-L6-v2`` encoder (hidden size 384) and a frozen
    LLaMA-3.1-8B-Instruct backbone. Per-subset values for ``epochs``, ``lr``,
    ``alpha`` and ``beta`` come from Table 6 of the paper and live in the YAML
    files under ``configs/``; the defaults here are the Hand-Crafted ones.
    """

    # --- data ---
    data_root: str = "data/Who&When"
    subset: str = "Algorithm-Generated"
    split_dir: str = "data/splits"
    #: 'w/ GT' in the paper: give the detector the task's reference answer.
    include_ground_truth: bool = False
    #: Defaults to True for Algorithm-Generated, False for Hand-Crafted.
    include_query: Optional[bool] = None

    # --- models ---
    encoder_model: str = DEFAULT_ENCODER
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    freeze_backbone: bool = True
    device_map: Optional[str] = "auto"
    dtype: str = "bfloat16"
    num_heads: int = 4

    # --- training ---
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.0
    lambda_proto: float = 0.1
    #: Stop early once the mean epoch loss drops below this value.
    loss_threshold: float = 0.1
    seed: int = 42

    # --- scoring ---
    scoring_mode: str = "teacher_forcing"
    alpha: float = 1.0
    beta: float = 0.1
    threshold: float = 0.5
    #: 'at_error' scores the prefix ending at the annotated error (paper
    #: protocol); 'full' scores the whole trajectory for localization.
    test_truncation: str = "at_error"

    # --- io ---
    output_dir: str = "outputs/masc"

    @property
    def subset_dir(self) -> Path:
        """Directory holding the chosen subset's trajectory files."""
        return Path(self.data_root) / self.subset

    @property
    def torch_dtype(self):
        """Resolve :attr:`dtype` to a ``torch`` dtype."""
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "Config":
        """Load a config file, then apply non-``None`` keyword overrides.

        Unknown keys in the YAML raise, so a typo in a config fails loudly
        instead of being silently ignored.
        """
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")

        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view, for logging alongside results."""
        return asdict(self)
