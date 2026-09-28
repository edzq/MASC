"""Loading Who&When trajectories and MAS execution traces."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence

import torch

from masc.encoder import SentenceEncoder
from masc.splits import list_subset_files

Subset = Literal["Hand-Crafted", "Algorithm-Generated"]
Truncation = Literal["before_error", "at_error", "full"]

#: Who&When names the acting agent under a different key per subset.
AGENT_KEY = {"Hand-Crafted": "role", "Algorithm-Generated": "name"}


@dataclass
class Trajectory:
    """One multi-agent trajectory, flattened into the detector's context.

    Attributes:
        steps: The strings that are encoded into the context. Element 0 is the
            task query when ``include_query`` is set, followed by agent outputs.
        labels: ``1`` for the annotated decisive error step, ``0`` elsewhere.
            Same length as ``steps``. Position 0 is never scored (it has no
            history), so evaluation drops it -- see
            :func:`masc.metrics.align_scores_and_labels`.
        mistake_step: Index of the decisive error *within the history*, as
            annotated in Who&When.
        source: File the trajectory came from.
        has_query: Whether ``steps[0]`` is the query rather than an agent output.
    """

    steps: List[str]
    labels: List[int]
    mistake_step: int
    source: str
    has_query: bool

    def __len__(self) -> int:
        return len(self.steps)


def _build_query(record: dict, include_ground_truth: bool) -> str:
    """Assemble the query string, optionally under the 'w/ GT' protocol.

    Who&When's ``w/ GT`` condition gives the detector access to the reference
    answer of the task; ``w/o GT`` relies on the agent logs alone. Here that
    switch appends the reference answer to the query that heads the context.
    """
    question = record.get("question", "")
    if isinstance(question, dict):  # MAS traces nest the task
        question = question.get("task", "")
    if include_ground_truth:
        ground_truth = record.get("ground_truth", "")
        if ground_truth:
            return f"{question}\nReference answer: {ground_truth}"
    return str(question)


def load_who_and_when(
    root: os.PathLike | str,
    subset: Subset,
    files: Optional[Sequence[str]] = None,
    truncation: Truncation = "at_error",
    include_query: Optional[bool] = None,
    include_ground_truth: bool = False,
) -> List[Trajectory]:
    """Load Who&When trajectories as detector contexts.

    Args:
        root: Directory holding the subset's JSON files.
        subset: ``"Hand-Crafted"`` or ``"Algorithm-Generated"``.
        files: Filenames to load; defaults to every JSON file in ``root``.
        truncation: Which prefix of the history to keep.

            * ``"before_error"`` -- steps strictly before the annotated error.
              These are the *normal* trajectories used for unsupervised
              training, so no error ever enters the training signal.
            * ``"at_error"`` -- prefix up to and including the error step. This
              is the detection protocol reported in the paper: the detector must
              flag the error the moment it is produced.
            * ``"full"`` -- the entire history, for measuring how early the
              first flag fires over a complete trajectory.
        include_query: Whether to prepend the task query to the context.
            Defaults to ``True`` for ``Algorithm-Generated`` and ``False`` for
            ``Hand-Crafted``, where the first history entry is already the
            human's question.
        include_ground_truth: Enables the ``w/ GT`` protocol; requires
            ``include_query``.

    Returns:
        One :class:`Trajectory` per file, in the order given by ``files``.
    """
    root = Path(root)
    if files is None:
        files = list_subset_files(root)
    if include_query is None:
        include_query = subset == "Algorithm-Generated"
    if include_ground_truth and not include_query:
        raise ValueError(
            "include_ground_truth requires include_query: the reference answer "
            "is appended to the query that heads the context."
        )

    trajectories: List[Trajectory] = []
    for filename in files:
        with open(root / filename, "r", encoding="utf-8") as handle:
            record = json.load(handle)

        history = record.get("history", [])
        mistake_step = int(record.get("mistake_step", -1))

        steps: List[str] = []
        labels: List[int] = []
        if include_query:
            steps.append(_build_query(record, include_ground_truth))
            labels.append(0)

        if truncation == "before_error":
            last = mistake_step  # exclusive: the error step is left out
        elif truncation == "at_error":
            last = mistake_step + 1
        else:
            last = len(history)
        last = min(last, len(history))

        for i in range(last):
            steps.append(history[i].get("content", ""))
            labels.append(1 if i == mistake_step else 0)

        trajectories.append(
            Trajectory(
                steps=steps,
                labels=labels,
                mistake_step=mistake_step,
                source=filename,
                has_query=include_query,
            )
        )
    return trajectories


def load_mas_traces(
    root: os.PathLike | str,
    num_files: Optional[int] = None,
    drop_last_step: bool = True,
) -> List[Trajectory]:
    """Load execution traces emitted by a MAS framework (``0.json``, ``1.json``, ...).

    These traces come from running MASC inside a multi-agent framework rather
    than from Who&When, so they carry no error annotation: ``labels`` is empty
    and ``mistake_step`` is ``-1``. Use them to fit the detector on a
    framework's own normal behaviour before plugging it in.

    Args:
        root: Directory of trace files named ``{index}.json``.
        num_files: Load indices ``0..num_files-1``. Defaults to every
            numerically-named JSON file present.
        drop_last_step: Drop the final history entry, which in these traces is
            the aggregated final answer rather than an intermediate step.
    """
    root = Path(root)
    if num_files is None:
        indices = sorted(
            int(p.stem) for p in root.glob("*.json") if p.stem.isdigit()
        )
    else:
        indices = list(range(num_files))

    trajectories: List[Trajectory] = []
    for index in indices:
        path = root / f"{index}.json"
        if not path.exists():
            print(f"[masc.data] trace not found, skipping: {path}")
            continue
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)

        history = record.get("history", [])
        last = len(history) - 1 if drop_last_step else len(history)
        steps = [_build_query(record, include_ground_truth=False)]
        steps += [history[i].get("content", "") for i in range(max(0, last))]

        trajectories.append(
            Trajectory(
                steps=steps,
                labels=[],
                mistake_step=-1,
                source=path.name,
                has_query=True,
            )
        )
    return trajectories


def filter_short(
    trajectories: Iterable[Trajectory], min_steps: int = 2
) -> List[Trajectory]:
    """Drop trajectories with fewer than ``min_steps`` context items.

    Such trajectories contain no next-step target, so they can neither be
    trained on nor scored. Filtering *before* encoding keeps trajectories and
    their embeddings index-aligned, which evaluation relies on.
    """
    return [t for t in trajectories if len(t) >= min_steps]


def embed_trajectories(
    encoder: SentenceEncoder, trajectories: Iterable[Trajectory]
) -> List[torch.Tensor]:
    """Encode each trajectory into an ``(L, D)`` tensor, one per input item."""
    return [encoder.encode(t.steps) for t in trajectories]
