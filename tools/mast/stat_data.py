#!/usr/bin/env python
"""Count correct vs. incorrect traces in a directory of MAST trace JSON files.

Auxiliary utility, not part of the paper's reported experiments. Each trace is
expected to carry ``other_data.correct``; traces missing the field are reported
separately rather than silently counted as failures.

Example:
    python tools/mast/stat_data.py --root data/MAST/traces/AG2 --recursive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, List


def iter_json_files(root: Path, recursive: bool) -> Iterator[Path]:
    """Yield JSON files directly under ``root``, or anywhere beneath it."""
    return root.rglob("*.json") if recursive else root.glob("*.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="Directory of trace JSON files.")
    parser.add_argument(
        "--recursive", action="store_true", help="Also descend into subdirectories."
    )
    parser.add_argument("--list", action="store_true", help="Print each incorrect trace path.")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"directory not found: {args.root}")

    correct: List[Path] = []
    incorrect: List[Path] = []
    missing: List[Path] = []
    unreadable: List[Path] = []

    for path in sorted(iter_json_files(args.root, args.recursive)):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (json.JSONDecodeError, OSError) as error:
            print(f"could not read {path}: {error}")
            unreadable.append(path)
            continue

        if not isinstance(record, dict):
            print(f"not a trace object, skipping: {path}")
            missing.append(path)
            continue
        other = record.get("other_data")
        flag = other.get("correct") if isinstance(other, dict) else None
        if flag is True:
            correct.append(path)
        elif flag is False:
            incorrect.append(path)
        else:
            missing.append(path)

    total = len(correct) + len(incorrect)
    print(f"\nscanned {args.root}")
    print(f"  correct        {len(correct)}")
    print(f"  incorrect      {len(incorrect)}")
    if total:
        print(f"  failure rate   {len(incorrect) / total:.1%}")
    if missing:
        print(f"  missing 'other_data.correct'  {len(missing)}")
    if unreadable:
        print(f"  unreadable                    {len(unreadable)}")

    if args.list:
        print("\nincorrect traces:")
        for path in incorrect:
            print(f"  {path}")


if __name__ == "__main__":
    main()
