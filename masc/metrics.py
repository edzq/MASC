"""Detection metrics (paper Sec. 4.1, 'Metrics')."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from masc.data import Trajectory
from masc.scoring import TrajectoryScores


def align_scores_and_labels(
    trajectory: Trajectory, scores: TrajectoryScores
) -> Tuple[List[int], List[float], List[int]]:
    """Line up per-step scores with the annotated labels for one trajectory.

    ``scores`` has one entry per context position ``1..L-1``: position 0 is
    never scored because it has no history to reconstruct from. So the first
    label is dropped, and the two sequences are then index-aligned.

    Returns:
        ``(labels, normalized_scores, predictions)``, all the same length.
    """
    labels = trajectory.labels[1:]
    size = min(len(labels), len(scores))
    return labels[:size], scores.normalized[:size], scores.predictions[:size]


def detection_metrics(
    labels: Sequence[int], scores: Sequence[float], predictions: Sequence[int]
) -> Dict[str, float]:
    """Step-level detection metrics over the flattened test set.

    AUC-ROC and AUPRC are threshold-free and computed from the continuous
    scores; accuracy / precision / recall / F1 come from the thresholded
    predictions. AUC is undefined when the labels are single-class, in which
    case it is reported as ``nan``.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if len(set(labels)) > 1:
        metrics["auc_roc"] = roc_auc_score(labels, scores)
        metrics["auprc"] = average_precision_score(labels, scores)
    else:
        metrics["auc_roc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics


def localization_accuracy(
    trajectories: Sequence[Trajectory], all_scores: Sequence[TrajectoryScores]
) -> Dict[str, float]:
    """Trajectory-level accuracy of *where* the error is placed.

    Two variants, both over trajectories that produced at least one score:

    * ``first_flag`` -- the first step the detector flags is the annotated error
      step. This is the online question: does the intervention fire on the right
      step? Trajectories that never flag count as misses.
    * ``argmax`` -- the highest-scoring step is the annotated error step. This is
      the offline attribution question and ignores the threshold entirely.
    """
    first_hits, argmax_hits, total = 0, 0, 0
    for trajectory, scores in zip(trajectories, all_scores):
        if len(scores) == 0:
            continue
        total += 1
        # Scores start at context position 1; with a query at position 0 the
        # offsets cancel and score index i corresponds to history step i.
        offset = 0 if trajectory.has_query else 1
        if 1 in scores.predictions:
            if scores.predictions.index(1) + offset == trajectory.mistake_step:
                first_hits += 1
        best = max(range(len(scores)), key=lambda i: scores.raw[i])
        if best + offset == trajectory.mistake_step:
            argmax_hits += 1

    if total == 0:
        return {"first_flag_accuracy": float("nan"), "argmax_accuracy": float("nan")}
    return {
        "first_flag_accuracy": first_hits / total,
        "argmax_accuracy": argmax_hits / total,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """One-line, aligned rendering of a metrics dict."""
    return " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
