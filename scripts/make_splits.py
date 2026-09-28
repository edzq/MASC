#!/usr/bin/env python
"""Generate the train/test split files shared by MASC and every baseline.

Example:
    python scripts/make_splits.py --data_root data/Who\\&When --out data/splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from masc.splits import make_split, save_split

SUBSETS = ["Hand-Crafted", "Algorithm-Generated"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data/Who&When")
    parser.add_argument("--out", default="data/splits")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.2,
        help="Fraction of trajectories available for training (paper: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for subset in SUBSETS:
        root = Path(args.data_root) / subset
        if not root.is_dir():
            print(f"skipping missing subset: {root}")
            continue
        train, test = make_split(root, args.train_ratio, args.seed)
        out = Path(args.out) / subset
        save_split(out / "train_files.json", train)
        save_split(out / "test_files.json", test)
        print(f"{subset}: {len(train)} train / {len(test)} test -> {out}")


if __name__ == "__main__":
    main()
