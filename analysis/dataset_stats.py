#!/usr/bin/env python
"""Descriptive statistics of the Who&When subsets (paper Appendix B).

Reports trajectory-length statistics, where the annotated error falls in
absolute and relative terms, and the ``is_corrected`` breakdown. Optionally
writes a per-trajectory plot of trajectory length against error step.

Example:
    python analysis/dataset_stats.py --data_root data/Who\\&When
    python analysis/dataset_stats.py --subset Algorithm-Generated --plot out.pdf
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

SUBSETS = ["Hand-Crafted", "Algorithm-Generated"]


def collect(subset_dir: Path) -> Dict[str, List]:
    """Read every trajectory in a subset directory into parallel lists."""
    lengths, error_steps, ratios, corrected = [], [], [], []
    files = sorted(
        (p for p in subset_dir.glob("*.json")),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
    )
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        history = record.get("history", [])
        if not isinstance(history, list) or not history:
            print(f"  skipping {path.name}: no usable history")
            continue
        mistake_step = int(record.get("mistake_step", -1))
        lengths.append(len(history))
        error_steps.append(mistake_step)
        ratios.append(mistake_step / len(history))
        # Hand-Crafted records spell this "is_corrected", Algorithm-Generated
        # ones "is_correct"; both mean "did the system reach the right answer".
        flag = record.get("is_corrected", record.get("is_correct"))
        corrected.append(flag)
    return {
        "files": [p.name for p in files],
        "lengths": lengths,
        "error_steps": error_steps,
        "ratios": ratios,
        "corrected": corrected,
    }


def report(subset: str, stats: Dict[str, List]) -> None:
    lengths, steps, ratios = stats["lengths"], stats["error_steps"], stats["ratios"]
    if not lengths:
        print(f"{subset}: no trajectories found")
        return
    print(f"\n{subset}  ({len(lengths)} trajectories)")
    print(
        f"  trajectory length   min {min(lengths):>3}  max {max(lengths):>3}  "
        f"mean {statistics.mean(lengths):6.2f}  median {statistics.median(lengths):6.1f}"
    )
    print(
        f"  error step          min {min(steps):>3}  max {max(steps):>3}  "
        f"mean {statistics.mean(steps):6.2f}  median {statistics.median(steps):6.1f}"
    )
    print(
        f"  relative position   min {min(ratios):.3f}  max {max(ratios):.3f}  "
        f"mean {statistics.mean(ratios):.3f}"
    )
    print(f"  errors at step 0    {sum(1 for s in steps if s == 0)}")
    true_count = sum(1 for c in stats["corrected"] if c is True)
    false_count = sum(1 for c in stats["corrected"] if c is False)
    missing = len(lengths) - true_count - false_count
    print(
        f"  solved correctly    true {true_count}  false {false_count}  "
        f"missing/invalid {missing}"
    )


def plot(subset: str, stats: Dict[str, List], out_path: Path) -> None:
    """Plot trajectory length against annotated error step, one point per file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    samples = np.arange(1, len(stats["lengths"]) + 1)
    plt.figure(figsize=(14, 7))
    plt.plot(samples, stats["lengths"], color="blue", label="Trajectory length",
             marker="o", markersize=4)
    plt.plot(samples, stats["error_steps"], color="red", label="Error step",
             linestyle="--", marker="s", markersize=4)
    plt.title(f"Trajectory length vs. error step ({subset})", fontsize=14)
    plt.xlabel("Trajectory index", fontsize=12)
    plt.ylabel("Step", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  plot written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_root", default="data/Who&When")
    parser.add_argument("--subset", choices=SUBSETS, help="Default: both subsets.")
    parser.add_argument("--plot", type=Path, help="Write a length-vs-error-step plot here.")
    args = parser.parse_args()

    subsets = [args.subset] if args.subset else SUBSETS
    for subset in subsets:
        subset_dir = Path(args.data_root) / subset
        if not subset_dir.is_dir():
            print(f"skipping missing subset: {subset_dir}")
            continue
        stats = collect(subset_dir)
        report(subset, stats)
        if args.plot:
            suffix = f".{subset}" if len(subsets) > 1 else ""
            plot(subset, stats, args.plot.with_suffix(f"{suffix}{args.plot.suffix}"))


if __name__ == "__main__":
    main()
