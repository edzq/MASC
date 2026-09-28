"""Anomaly scoring at inference time (paper Sec. 3.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from masc.model import PrototypeReconstructor

ScoringMode = Literal["teacher_forcing", "autoregressive"]


@dataclass
class TrajectoryScores:
    """Per-step anomaly scores for one trajectory.

    Every list has length ``L - 1`` for a trajectory of ``L`` context items:
    entry ``i`` scores context position ``i + 1``, because position 0 has no
    history to be reconstructed from.

    Attributes:
        raw: ``s(t) = alpha * ||x_hat_t - x_t||^2 + beta * (1 - cos(x_hat_t, p))``.
        normalized: ``sigmoid(raw)``, the score fed to AUC-ROC.
        predictions: ``1`` where ``normalized > threshold``, else ``0``.
        recon: The reconstruction term alone, for ablation/analysis.
        proto: The prototype-misalignment term alone, for ablation/analysis.
    """

    raw: List[float] = field(default_factory=list)
    normalized: List[float] = field(default_factory=list)
    predictions: List[int] = field(default_factory=list)
    recon: List[float] = field(default_factory=list)
    proto: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.raw)


def minmax_normalize(scores: Sequence[float]) -> List[float]:
    """Min-max rescale to ``[0, 1]``; returns all ``0.5`` for a constant input."""
    arr = np.asarray(scores, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return [0.5] * len(arr)
    return ((arr - lo) / (hi - lo)).tolist()


@torch.no_grad()
def score_trajectory(
    model: PrototypeReconstructor,
    embeddings: torch.Tensor,
    device: str = "cuda",
    mode: ScoringMode = "teacher_forcing",
    alpha: float = 0.9,
    beta: float = 0.1,
    threshold: float = 0.5,
) -> TrajectoryScores:
    """Score every step of one trajectory.

    Args:
        model: A trained :class:`PrototypeReconstructor`.
        embeddings: ``(L, D)`` context embeddings for a single trajectory, as
            produced by :func:`masc.data.embed_trajectories`.
        device: Device to run on.
        mode: ``"teacher_forcing"`` conditions each prediction on the *realised*
            history, which is the online deployment setting reported in the
            paper: at step ``t`` the system has genuinely observed steps
            ``< t``. ``"autoregressive"`` instead extends the history with its
            own predictions, which is useful for probing how far the model can
            roll forward without observations.
        alpha: Weight of the reconstruction term (Eq. 12).
        beta: Weight of the prototype-misalignment term (Eq. 12).
        threshold: Decision threshold on the sigmoid-normalised score, i.e. the
            ``delta`` that gates self-correction (Eq. 13).

    Returns:
        A :class:`TrajectoryScores` with ``L - 1`` entries. A trajectory with
        fewer than two items yields a single maximally-anomalous entry, since
        there is no history to reconstruct from.
    """
    model.eval()
    seq = embeddings.to(device)
    length = seq.size(0)

    if length < 2:
        return TrajectoryScores(
            raw=[1.0], normalized=[1.0], predictions=[1], recon=[1.0], proto=[1.0]
        )

    prototype = F.normalize(model.prototype.to(seq.device), dim=-1)  # (1, D)
    scores = TrajectoryScores()

    # Autoregressive mode seeds the history with the true first item and then
    # appends its own predictions; teacher forcing re-slices the true sequence.
    history = seq[0:1, :].unsqueeze(0)  # (1, 1, D)

    for t in range(length - 1):
        if mode == "teacher_forcing":
            history = seq[: t + 1, :].unsqueeze(0)  # (1, t+1, D)

        pred_all, _ = model(history, train_mode=False)
        pred_next = pred_all[:, -1, :]  # (1, D)
        true_next = seq[t + 1, :].unsqueeze(0)  # (1, D)

        recon_term = F.mse_loss(pred_next, true_next, reduction="mean").item()
        similarity = torch.matmul(F.normalize(pred_next, dim=-1), prototype.T)
        proto_term = 1.0 - similarity.squeeze().item()

        raw = alpha * recon_term + beta * proto_term
        normalized = torch.sigmoid(torch.tensor(raw, dtype=torch.float32)).item()

        scores.raw.append(raw)
        scores.normalized.append(normalized)
        scores.predictions.append(int(normalized > threshold))
        scores.recon.append(recon_term)
        scores.proto.append(proto_term)

        if mode == "autoregressive":
            history = torch.cat([history, pred_next.unsqueeze(1)], dim=1)

    return scores
