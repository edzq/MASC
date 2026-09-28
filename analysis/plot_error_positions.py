#!/usr/bin/env python
"""Plot where decisive errors fall inside Who&When trajectories.

Motivates the prototype prior: a large share of errors occur early, when the
interaction history is still too sparse for reconstruction alone to be reliable
(paper Sec. 1 and Sec. 3.2).

Produces three figures over both subsets: a binned bar chart of relative error
position, a histogram overlay of the same quantity, and a scatter of trajectory
length against error step.

Example:
    python analysis/plot_error_positions.py --out_dir figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never shown interactively
import matplotlib.pyplot as plt
import numpy as np

SUBSETS = {"HC": "Hand-Crafted", "Alg": "Algorithm-Generated"}
COLORS = {"HC": "#4C72B0", "Alg": "#DD8452"}


def collect(subset_dir: Path) -> Dict[str, List[float]]:
    """Return error step, trajectory length and their ratio for every file."""
    steps, lengths, ratios = [], [], []
    for path in sorted(subset_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        history = record.get("history", [])
        if not history:
            continue
        step = int(record["mistake_step"])
        steps.append(step)
        lengths.append(len(history))
        ratios.append(step / len(history))
    return {"steps": steps, "lengths": lengths, "ratios": ratios}


def plot_binned(data: Dict[str, Dict], out_path: Path, num_bins: int = 5) -> None:
    """Grouped bars: how many errors fall in each fifth of a trajectory."""
    bins = np.linspace(0, 1, num_bins + 1)
    x = np.arange(num_bins)
    width = 0.8 / max(1, len(data))

    plt.figure(figsize=(8, 6))
    for i, (key, values) in enumerate(data.items()):
        hist, _ = np.histogram(values["ratios"], bins=bins)
        offset = (i - (len(data) - 1) / 2) * width
        plt.bar(x + offset, hist, width, label=key, color=COLORS[key])
    plt.xticks(
        x, [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(num_bins)], fontsize=16
    )
    plt.yticks(fontsize=14)
    plt.xlabel("Relative error position", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def plot_histogram(data: Dict[str, Dict], out_path: Path, num_bins: int = 10) -> None:
    """Overlaid histograms of relative error position, one per subset."""
    plt.figure(figsize=(7, 5))
    for key, values in data.items():
        plt.hist(
            values["ratios"], bins=num_bins, alpha=0.6, edgecolor="black",
            color=COLORS[key], label=key,
        )
    plt.xlabel("Relative error position (mistake_step / trajectory length)", fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def plot_scatter(data: Dict[str, Dict], out_path: Path) -> None:
    """Scatter of trajectory length against error step."""
    plt.figure(figsize=(6, 4))
    for key, values in data.items():
        plt.scatter(
            values["lengths"], values["steps"], alpha=0.6, color=COLORS[key], label=key
        )
    plt.xlabel("Trajectory length", fontsize=12)
    plt.ylabel("Error step", fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_root", default="data/Who&When")
    parser.add_argument("--out_dir", type=Path, default=Path("figures"))
    parser.add_argument("--num_bins", type=int, default=5)
    args = parser.parse_args()

    data = {}
    for key, subset in SUBSETS.items():
        subset_dir = Path(args.data_root) / subset
        if not subset_dir.is_dir():
            print(f"skipping missing subset: {subset_dir}")
            continue
        data[key] = collect(subset_dir)
        print(f"{subset}: {len(data[key]['ratios'])} trajectories")
    if not data:
        raise SystemExit("no subsets found; check --data_root")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_binned(data, args.out_dir / "error_step_stats.pdf", args.num_bins)
    plot_histogram(data, args.out_dir / "error_position_hist.pdf")
    plot_scatter(data, args.out_dir / "length_vs_error_step.pdf")


if __name__ == "__main__":
    main()
