#!/usr/bin/env python
"""Score an LLM-as-detector log under the early-prediction-penalty protocol.

A stricter alternative to ``evaluate.py --metric step_level``. A judge that flags
an error *before* the annotated one has raised a false alarm on every step in
between, so those steps are all counted as predicted-positive rather than just
the one step the judge named. A judge that flags late (or not at all) is charged
a single false positive at its predicted step.

For the lenient variant, where only the named step counts, use ``evaluate.py``.

Example:
    python baselines/llm_as_detector/evaluate_step_level.py \\
        --data_path ../../data/Who\\&When/Algorithm-Generated \\
        --eval_file outputs_no_answer/all_at_once_gpt-4o-mini_alg_generated.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PREDICTION_BLOCK = re.compile(r"Prediction for ([^:]+\.json):(.*?)(?=Prediction for|\Z)", re.DOTALL)
AGENT_PATTERN = re.compile(r"Agent Name:\s*([\w_]+)", re.IGNORECASE)
STEP_PATTERN = re.compile(r"Step Number:\s*(\d+)", re.IGNORECASE)


def read_predictions(eval_file: str | os.PathLike) -> Dict[str, Dict[str, str]]:
    """Parse a prediction log, keeping the first prediction per trajectory."""
    if not os.path.exists(eval_file):
        raise SystemExit(f"evaluation file not found: {eval_file}")
    with open(eval_file, "r", encoding="utf-8") as handle:
        data = handle.read()

    predictions: Dict[str, Dict[str, str]] = {}
    skipped = 0
    for block in PREDICTION_BLOCK.finditer(data):
        name = block.group(1).strip()
        content = block.group(2).strip()
        agent = AGENT_PATTERN.search(content)
        step = STEP_PATTERN.search(content)
        if not (agent and step):
            skipped += 1
            continue
        # Step-by-Step can emit several blocks per trajectory; the first one is
        # the earliest step it flagged, which is the decision under test.
        predictions.setdefault(
            name, {"predicted_agent": agent.group(1), "predicted_step": step.group(1)}
        )

    print(f"--- {eval_file} ---")
    print(f"parsed predictions for {len(predictions)} trajectories")
    if skipped:
        print(f"skipped {skipped} blocks with no parsable agent/step")
    return predictions


def build_labels_with_penalty(
    data_path: str | os.PathLike, predictions: Dict[str, Dict[str, str]]
) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    """Build flattened truth/prediction sequences with the early-flag penalty.

    Each trajectory contributes ``mistake_step + 1`` steps: ``mistake_step``
    correct ones followed by the error. An early flag at step ``p < mistake_step``
    marks every step in ``[p, mistake_step)`` as predicted-positive; a flag at or
    after the error marks that one step (clamped into range).

    Returns:
        ``(y_true, y_pred, actual_steps, predicted_steps, trajectory_lengths)``.
    """
    data_path = Path(data_path)
    y_true: List[int] = []
    y_pred: List[int] = []
    actual_steps, predicted_steps, lengths = [], [], []

    for name, prediction in predictions.items():
        labeled_file = data_path / name
        if not labeled_file.exists():
            print(f"no reference file for prediction '{name}'")
            continue
        with open(labeled_file, "r", encoding="utf-8") as handle:
            record = json.load(handle)

        actual_step = int(record["mistake_step"])
        predicted_step = int(prediction["predicted_step"])
        length = len(record["history"])

        actual_steps.append(actual_step)
        predicted_steps.append(predicted_step)
        lengths.append(length)

        # Truth: steps 0..actual_step-1 are correct, actual_step is the error.
        truth = [0] * actual_step + [1]
        predicted = [0] * len(truth)

        if predicted_step < actual_step:
            # Early flag: charge a false positive for every step from the flag up
            # to (but excluding) the real error.
            for i in range(predicted_step, actual_step):
                predicted[i] = 1
        else:
            # On time or late: a single positive, clamped to the sequence.
            predicted[min(predicted_step, len(predicted) - 1)] = 1

        y_true.extend(truth)
        y_pred.extend(predicted)

    assert len(y_true) == len(y_pred), "truth/prediction sequences fell out of sync"
    return y_true, y_pred, actual_steps, predicted_steps, lengths


def evaluate_classification(y_true: List[int], y_pred: List[int]) -> None:
    """Print the step-level classification report for binary predictions."""
    if not y_true:
        raise SystemExit("no steps to evaluate; check --data_path and --eval_file")
    print(f"\nscored {len(y_true)} steps, {sum(y_true)} annotated errors")
    print(f"accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"f1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
    # Predictions are binary, so these are not comparable to a detector's
    # threshold-free scores; they are reported for completeness.
    print(f"auc_roc:   {roc_auc_score(y_true, y_pred):.4f}")
    print(f"auprc:     {average_precision_score(y_true, y_pred):.4f}")
    print("\nconfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nclassification report:")
    print(
        classification_report(
            y_true, y_pred, target_names=["Correct Step", "Error Step"], zero_division=0
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_path", default="../../data/Who&When/Algorithm-Generated",
                        help="Directory of the reference Who&When JSON files.")
    parser.add_argument("--eval_file", required=True, help="Prediction log to score.")
    args = parser.parse_args()

    if not Path(args.data_path).is_dir():
        raise SystemExit(f"data directory not found: {args.data_path}")

    predictions = read_predictions(args.eval_file)
    if not predictions:
        raise SystemExit("no predictions parsed; is this the right log file?")
    y_true, y_pred, *_ = build_labels_with_penalty(args.data_path, predictions)
    evaluate_classification(y_true, y_pred)


if __name__ == "__main__":
    main()
