#!/usr/bin/env python
"""Score an LLM-as-detector prediction log against the Who&When annotations.

``run_inference.py`` writes one or more ``Prediction for <file>.json:`` blocks
per trajectory, each naming an agent and a step. This script parses that log and
reports either trajectory-level attribution accuracy or step-level detection
metrics.

Note that Step-by-Step can emit several predictions for one trajectory (it flags
each step it judges erroneous), so a prediction log maps each file to a *list* of
predictions. The trajectory-level metrics reduce that list to its first entry --
the earliest step the judge flagged -- while ``--metric step_level`` uses all of
them.

Example:
    python baselines/llm_as_detector/evaluate.py \\
        --data_path ../../data/Who\\&When/Algorithm-Generated \\
        --eval_file outputs_no_answer/step_by_step_gpt-4o-mini_alg_generated.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PREDICTION_BLOCK = re.compile(r"Prediction for ([^:]+\.json):(.*?)(?=Prediction for|\Z)", re.DOTALL)
AGENT_PATTERN = re.compile(r"Agent Name:\s*([\w_]+)", re.IGNORECASE)
STEP_PATTERN = re.compile(r"Step Number:\s*(\d+)", re.IGNORECASE)


def read_predictions(eval_file: str | os.PathLike) -> Dict[str, List[Dict[str, str]]]:
    """Parse a prediction log into ``{filename: [{agent, step}, ...]}``."""
    if not os.path.exists(eval_file):
        raise SystemExit(f"evaluation file not found: {eval_file}")
    with open(eval_file, "r", encoding="utf-8") as handle:
        data = handle.read()

    predictions: Dict[str, List[Dict[str, str]]] = {}
    parsed, skipped = 0, 0
    for block in PREDICTION_BLOCK.finditer(data):
        name = block.group(1).strip()
        content = block.group(2).strip()
        agent = AGENT_PATTERN.search(content)
        step = STEP_PATTERN.search(content)
        if not (agent and step):
            skipped += 1
            continue
        predictions.setdefault(name, []).append(
            {"predicted_agent": agent.group(1), "predicted_step": step.group(1)}
        )
        parsed += 1

    print(f"--- {eval_file} ---")
    print(f"parsed {parsed} prediction blocks over {len(predictions)} trajectories")
    if skipped:
        print(f"skipped {skipped} blocks with no parsable agent/step")
    return predictions


def read_actual_data(labeled_json: Path) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(mistake_agent, mistake_step)`` as strings, or ``(None, None)``."""
    try:
        with open(labeled_json, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        print(f"could not read {labeled_json}: {error}")
        return None, None
    agent = record.get("mistake_agent")
    step = record.get("mistake_step")
    if agent is None or step is None:
        print(f"missing mistake_agent/mistake_step in {labeled_json}")
        return None, None
    return str(agent), str(step)


def attribution_accuracy(predictions, data_path: Path, total_files: int) -> Dict[str, float]:
    """Trajectory-level accuracy of the predicted agent and step.

    Accuracy is over *all* reference files, so trajectories the judge produced no
    parsable prediction for count as misses -- matching the Who&When protocol.
    """
    correct_agent, correct_step, evaluated = 0, 0, 0
    for name, preds in predictions.items():
        labeled_file = data_path / name
        if not labeled_file.exists():
            print(f"no reference file for prediction '{name}'")
            continue
        actual_agent, actual_step = read_actual_data(labeled_file)
        if actual_agent is None:
            continue
        evaluated += 1
        first = preds[0]
        if actual_agent in first["predicted_agent"]:
            correct_agent += 1
        if actual_step in first["predicted_step"]:
            correct_step += 1

    print(f"evaluated {evaluated} of {total_files} reference trajectories")
    denominator = total_files or 1
    return {
        "agent_accuracy": 100.0 * correct_agent / denominator,
        "step_accuracy": 100.0 * correct_step / denominator,
    }


def step_level_metrics(predictions, data_path: Path) -> Dict[str, float]:
    """Step-level detection metrics over the prefix ending at the annotated error.

    Each trajectory contributes ``mistake_step + 1`` steps with a single positive
    label; a step counts as predicted-positive if the judge flagged it. The
    predictions are binary, so AUC-ROC/AUPRC here are computed on hard labels and
    are not comparable to a detector's threshold-free scores.
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    for name, preds in predictions.items():
        labeled_file = data_path / name
        if not labeled_file.exists():
            print(f"no reference file for prediction '{name}'")
            continue
        with open(labeled_file, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        actual_step = int(record["mistake_step"])
        length = actual_step + 1

        truth = [0] * length
        truth[actual_step] = 1
        predicted = [0] * length
        for pred in preds:
            index = int(pred["predicted_step"])
            if 0 <= index < length:
                predicted[index] = 1
        y_true.extend(truth)
        y_pred.extend(predicted)

    if not y_true:
        raise SystemExit("no steps to evaluate; check --data_path and --eval_file")
    print(f"scored {len(y_true)} steps, {sum(y_true)} annotated errors")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_pred),
        "auprc": average_precision_score(y_true, y_pred),
    }


def plot_predicted_vs_actual(predictions, data_path: Path, out_path: Path) -> None:
    """Bar chart of actual error step, first predicted step, and trajectory length."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    actual, predicted, lengths = [], [], []
    for name, preds in predictions.items():
        labeled_file = data_path / name
        if not labeled_file.exists():
            continue
        with open(labeled_file, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        actual.append(int(record["mistake_step"]) + 1)
        predicted.append(int(preds[0]["predicted_step"]) + 1)
        lengths.append(len(record["history"]))

    x = np.arange(len(actual))
    width = 0.25
    plt.figure(figsize=(18, 6))
    plt.bar(x - width, actual, width, label="Actual error step")
    plt.bar(x, predicted, width, label="Predicted error step")
    plt.bar(x + width, lengths, width, label="Trajectory length")
    plt.xlabel("Trajectory index")
    plt.ylabel("Step")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_path", default="../../data/Who&When/Algorithm-Generated",
                        help="Directory of the reference Who&When JSON files.")
    parser.add_argument("--eval_file", required=True, help="Prediction log to score.")
    parser.add_argument(
        "--metric", default="step_level", choices=["step_level", "attribution", "both"],
        help="'attribution' is the Who&When agent/step accuracy; 'step_level' is "
             "per-step detection.",
    )
    parser.add_argument("--plot", type=Path, help="Also write a predicted-vs-actual plot here.")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_dir():
        raise SystemExit(f"data directory not found: {data_path}")
    total_files = len(list(data_path.glob("*.json")))

    predictions = read_predictions(args.eval_file)
    if not predictions:
        raise SystemExit("no predictions parsed; is this the right log file?")

    if args.metric in {"attribution", "both"}:
        metrics = attribution_accuracy(predictions, data_path, total_files)
        print("[attribution] " + " | ".join(f"{k}: {v:.2f}%" for k, v in metrics.items()))
    if args.metric in {"step_level", "both"}:
        metrics = step_level_metrics(predictions, data_path)
        print("[step level]  " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    if args.plot:
        plot_predicted_vs_actual(predictions, data_path, args.plot)


if __name__ == "__main__":
    main()
