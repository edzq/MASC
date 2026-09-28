#!/usr/bin/env python
"""Fit / run the MASC detector on a multi-agent framework's own execution traces.

Use this when plugging MASC into a MAS framework rather than benchmarking it on
Who&When. ``--mode fit`` learns the detector from traces of the framework's
normal behaviour; ``--mode score`` scores traces and reports, per step, whether
the Eq. 13 gate would fire and hand the step to a correction agent.

Traces are JSON files named ``0.json``, ``1.json``, ... with a ``question``
object carrying a ``task`` field and a ``history`` list of ``{"content": ...}``
entries -- the format emitted by the G-Designer-style frameworks used in the
paper. They carry no error annotations, so no detection metrics are reported.

Examples:
    python scripts/run_mas_detector.py --mode fit \\
        --trace_dir traces/mmlu/FullConnected --output_dir outputs/masc/mmlu
    python scripts/run_mas_detector.py --mode score \\
        --trace_dir traces/mmlu/FullConnected \\
        --checkpoint outputs/masc/mmlu/detector.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from masc.config import Config
from masc.data import embed_trajectories, filter_short, load_mas_traces
from masc.encoder import SentenceEncoder
from masc.model import PrototypeReconstructor, load_heads, save_heads, train_epoch
from masc.scoring import score_trajectory


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["fit", "score"], required=True)
    parser.add_argument("--trace_dir", required=True, help="Directory of {index}.json traces.")
    parser.add_argument("--config", default="configs/mas_trace.yaml")
    parser.add_argument("--checkpoint", help="Required for --mode score.")
    parser.add_argument(
        "--num_traces",
        type=int,
        help="Load traces 0..N-1 only; defaults to every trace in the directory.",
    )
    parser.add_argument("--base_model")
    parser.add_argument("--encoder_model")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--threshold", type=float, help="Correction gate delta.")
    parser.add_argument("--output_dir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.mode == "score" and not args.checkpoint:
        raise SystemExit("--mode score requires --checkpoint")

    overrides = {
        k: v
        for k, v in vars(args).items()
        if k in {f for f in Config.__dataclass_fields__} and v is not None
    }
    cfg = Config.from_yaml(args.config, **overrides)
    torch.manual_seed(cfg.seed)

    traces = filter_short(load_mas_traces(args.trace_dir, args.num_traces), min_steps=2)
    print(f"[data] {len(traces)} usable traces from {args.trace_dir}")
    if not traces:
        raise SystemExit("no trace has two or more steps; nothing to do")

    encoder = SentenceEncoder(cfg.encoder_model, device=args.device)
    data = embed_trajectories(encoder, traces)

    model = PrototypeReconstructor(
        emb_dim=data[0].size(-1),
        base_model=cfg.base_model,
        freeze_backbone=cfg.freeze_backbone,
        device_map=cfg.device_map,
        torch_dtype=cfg.torch_dtype,
        num_heads=cfg.num_heads,
    )
    if cfg.device_map is None:
        model.to(args.device)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "fit":
        optimizer = torch.optim.AdamW(
            model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        for epoch in range(cfg.epochs):
            loss = train_epoch(
                model, data, optimizer, device=args.device, lambda_proto=cfg.lambda_proto
            )
            print(f"[fit] epoch {epoch:>3} | loss {loss:.4f}")
            if loss < cfg.loss_threshold:
                print(f"[fit] loss below {cfg.loss_threshold}; stopping early")
                break
        checkpoint = output_dir / "detector.pt"
        save_heads(model, checkpoint)
        print(f"[done] checkpoint written to {checkpoint}")
        return

    load_heads(model, args.checkpoint)
    results, flagged_steps, total_steps = [], 0, 0
    for trace, embeddings in zip(traces, data):
        scores = score_trajectory(
            model,
            embeddings,
            device=args.device,
            mode=cfg.scoring_mode,
            alpha=cfg.alpha,
            beta=cfg.beta,
            threshold=cfg.threshold,
        )
        # Score index i covers context position i+1, i.e. history step i, since
        # the task query heads every MAS trace context.
        triggers = [i for i, flag in enumerate(scores.predictions) if flag]
        flagged_steps += len(triggers)
        total_steps += len(scores)
        results.append(
            {
                "file": trace.source,
                "scores": scores.normalized,
                "trigger_steps": triggers,
            }
        )

    rate = flagged_steps / total_steps if total_steps else float("nan")
    print(
        f"[score] {flagged_steps}/{total_steps} steps exceed delta={cfg.threshold} "
        f"({rate:.1%}); these are the steps handed to the correction agent"
    )
    out = output_dir / "mas_scores.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"config": cfg.to_dict(), "traces": results}, handle, indent=2)
    print(f"[done] per-step scores written to {out}")


if __name__ == "__main__":
    main()
