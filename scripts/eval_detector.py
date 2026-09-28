#!/usr/bin/env python
"""Evaluate a trained MASC detector on held-out Who&When trajectories.

Reports step-level AUC-ROC / AUPRC and thresholded accuracy, plus
trajectory-level localization accuracy.

Example:
    python scripts/eval_detector.py --config configs/algorithm_generated.yaml \\
        --checkpoint outputs/masc/algorithm_generated/detector.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from masc.config import Config
from masc.data import embed_trajectories, filter_short, load_who_and_when
from masc.encoder import SentenceEncoder
from masc.metrics import (
    align_scores_and_labels,
    detection_metrics,
    format_metrics,
    localization_accuracy,
)
from masc.model import PrototypeReconstructor, load_heads
from masc.scoring import score_trajectory
from masc.splits import load_split


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to detector.pt.")
    parser.add_argument("--subset", choices=["Hand-Crafted", "Algorithm-Generated"])
    parser.add_argument("--base_model")
    parser.add_argument("--encoder_model")
    parser.add_argument("--alpha", type=float, help="Weight of the reconstruction term.")
    parser.add_argument("--beta", type=float, help="Weight of the prototype term.")
    parser.add_argument("--threshold", type=float, help="Decision threshold delta.")
    parser.add_argument("--scoring_mode", choices=["teacher_forcing", "autoregressive"])
    parser.add_argument(
        "--test_truncation",
        choices=["at_error", "full"],
        help="'at_error' is the paper protocol; 'full' scores whole trajectories.",
    )
    parser.add_argument("--include_ground_truth", action="store_true", default=None)
    parser.add_argument("--output_dir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in {"config", "checkpoint", "device"} and v is not None
    }
    cfg = Config.from_yaml(args.config, **overrides)

    test_files = load_split(Path(cfg.split_dir) / cfg.subset / "test_files.json")
    trajectories = load_who_and_when(
        cfg.subset_dir,
        subset=cfg.subset,
        files=test_files,
        truncation=cfg.test_truncation,
        include_query=cfg.include_query,
        include_ground_truth=cfg.include_ground_truth,
    )
    trajectories = filter_short(trajectories, min_steps=2)
    print(f"[data] {cfg.subset}: scoring {len(trajectories)} test trajectories")

    encoder = SentenceEncoder(cfg.encoder_model, device=args.device)
    data = embed_trajectories(encoder, trajectories)

    model = PrototypeReconstructor(
        emb_dim=data[0].size(-1),
        base_model=cfg.base_model,
        freeze_backbone=True,
        device_map=cfg.device_map,
        torch_dtype=cfg.torch_dtype,
        num_heads=cfg.num_heads,
    )
    if cfg.device_map is None:
        model.to(args.device)
    load_heads(model, args.checkpoint)

    all_scores, labels, scores, predictions = [], [], [], []
    per_trajectory = []
    for trajectory, embeddings in zip(trajectories, data):
        result = score_trajectory(
            model,
            embeddings,
            device=args.device,
            mode=cfg.scoring_mode,
            alpha=cfg.alpha,
            beta=cfg.beta,
            threshold=cfg.threshold,
        )
        all_scores.append(result)
        y, s, p = align_scores_and_labels(trajectory, result)
        labels.extend(y)
        scores.extend(s)
        predictions.extend(p)
        per_trajectory.append(
            {
                "file": trajectory.source,
                "mistake_step": trajectory.mistake_step,
                "scores": result.normalized,
                "predictions": result.predictions,
                "labels": y,
            }
        )

    detection = detection_metrics(labels, scores, predictions)
    localization = localization_accuracy(trajectories, all_scores)

    print(f"\n[steps] {len(labels)} scored steps, {sum(labels)} annotated errors")
    print(f"[detection]    {format_metrics(detection)}")
    print(f"[localization] {format_metrics(localization)}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg.to_dict(),
        "checkpoint": str(args.checkpoint),
        "num_trajectories": len(trajectories),
        "num_steps": len(labels),
        "detection": detection,
        "localization": localization,
        "per_trajectory": per_trajectory,
    }
    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[done] results written to {output_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
