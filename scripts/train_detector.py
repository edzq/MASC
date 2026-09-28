#!/usr/bin/env python
"""Train the MASC detector on normal trajectories (unsupervised).

Only the projection layers, the prototype and the prototype-update attention are
trained; the sentence encoder and the LLM backbone stay frozen. Training sees
only the steps that precede the annotated error, so no error label ever enters
the objective.

Example:
    python scripts/train_detector.py --config configs/algorithm_generated.yaml
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
from masc.model import PrototypeReconstructor, save_heads, train_epoch
from masc.splits import load_split


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    # Overrides; anything left as None keeps the config's value.
    parser.add_argument("--subset", choices=["Hand-Crafted", "Algorithm-Generated"])
    parser.add_argument("--base_model", help="Frozen LLM backbone id or path.")
    parser.add_argument("--encoder_model", help="Frozen sentence encoder id or path.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--lambda_proto", type=float)
    parser.add_argument("--output_dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--include_ground_truth",
        action="store_true",
        default=None,
        help="Use the 'w/ GT' protocol: expose the task's reference answer.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in {"config", "device"} and v is not None
    }
    cfg = Config.from_yaml(args.config, **overrides)
    torch.manual_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2)

    train_files = load_split(Path(cfg.split_dir) / cfg.subset / "train_files.json")
    trajectories = load_who_and_when(
        cfg.subset_dir,
        subset=cfg.subset,
        files=train_files,
        truncation="before_error",  # normal steps only
        include_query=cfg.include_query,
        include_ground_truth=cfg.include_ground_truth,
    )
    trajectories = filter_short(trajectories, min_steps=2)
    print(
        f"[data] {cfg.subset}: {len(train_files)} train files -> "
        f"{len(trajectories)} usable normal trajectories"
    )
    if not trajectories:
        raise SystemExit(
            "no trajectory has two or more normal steps; nothing to train on"
        )

    encoder = SentenceEncoder(cfg.encoder_model, device=args.device)
    data = embed_trajectories(encoder, trajectories)
    emb_dim = data[0].size(-1)
    print(f"[data] embedding dim: {emb_dim}")

    model = PrototypeReconstructor(
        emb_dim=emb_dim,
        base_model=cfg.base_model,
        freeze_backbone=cfg.freeze_backbone,
        device_map=cfg.device_map,
        torch_dtype=cfg.torch_dtype,
        num_heads=cfg.num_heads,
    )
    if cfg.device_map is None:
        model.to(args.device)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"[model] {cfg.base_model} frozen; {trainable:,} trainable parameters")

    history = []
    for epoch in range(cfg.epochs):
        loss = train_epoch(
            model,
            data,
            optimizer,
            device=args.device,
            lambda_proto=cfg.lambda_proto,
        )
        history.append(loss)
        print(f"[train] epoch {epoch:>3} | loss {loss:.4f}")
        if loss < cfg.loss_threshold:
            print(f"[train] loss below {cfg.loss_threshold}; stopping early")
            break

    checkpoint = output_dir / "detector.pt"
    save_heads(model, checkpoint)
    with open(output_dir / "train_loss.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"[done] checkpoint written to {checkpoint}")


if __name__ == "__main__":
    main()
