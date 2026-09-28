"""MASC: Metacognitive Self-Correction for Multi-Agent Systems.

Reference implementation of the step-level, unsupervised error detector
described in "Metacognitive Self-Correction for Multi-Agent System via
Prototype-Guided Next-Execution Reconstruction" (Findings of ACL 2026).

Typical use:

    from masc import SentenceEncoder, PrototypeReconstructor
    from masc.data import load_who_and_when
    from masc.scoring import score_trajectory

``SentenceEncoder`` and ``PrototypeReconstructor`` are resolved lazily so that
torch-free helpers such as :mod:`masc.splits` stay importable without a full
deep-learning stack installed.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from masc.encoder import SentenceEncoder
    from masc.model import PrototypeReconstructor

__all__ = ["SentenceEncoder", "PrototypeReconstructor"]
__version__ = "1.0.0"

_LAZY = {"SentenceEncoder": "masc.encoder", "PrototypeReconstructor": "masc.model"}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
