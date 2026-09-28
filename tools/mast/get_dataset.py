#!/usr/bin/env python
"""Export MAST / MAD trajectories to per-trace text files.

Auxiliary utility, not part of the paper's reported experiments -- MASC is
evaluated on Who&When and AgentErrorBench. It is kept because MAST (Cemri et al.,
2025) provides a complementary taxonomy of MAS failure modes (paper Appendix A)
and these traces are useful for inspecting them.

Downloads ``mcemri/MAD`` from the Hugging Face Hub, keeps the traces whose MAST
annotation flags at least one failure mode, and writes them grouped by
``<output_dir>/<mas_name>/<benchmark>/<trace_id>.txt``.

Example:
    python tools/mast/get_dataset.py --mas_name AG2 --output_dir data/MAST
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "mcemri/MAD"
FULL_DATASET = "MAD_full_dataset.json"
HUMAN_LABELLED = "MAD_human_labelled_dataset.json"

BENCHMARKS = [
    "Olympiad", "SWE-Bench-Lite", "Test-C", "ProgramDev", "MMLU", "GSM", "GAIA",
]


def load_dataset(filename: str) -> list:
    """Download and parse one MAD dataset file from the Hub."""
    path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    print(f"loaded {len(data)} records from {filename}")
    return data


def split_by_annotation(data: list) -> tuple[list, list]:
    """Partition traces into (failing, clean) by their MAST annotation.

    A trace counts as clean only when every annotated failure mode is zero.
    """
    failing, clean = [], []
    for item in data:
        annotation = item.get("mast_annotation", {})
        (clean if annotation and all(v == 0 for v in annotation.values()) else failing).append(item)
    return failing, clean


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output_dir", type=Path, default=Path("data/MAST"))
    parser.add_argument(
        "--mas_name", help="Export only this MAS (e.g. AG2, OpenManus). Default: all."
    )
    parser.add_argument(
        "--benchmarks", nargs="*", default=BENCHMARKS,
        help=f"Benchmarks to export. Default: {' '.join(BENCHMARKS)}",
    )
    parser.add_argument(
        "--human_labelled", action="store_true",
        help="Use the human-labelled subset instead of the full dataset.",
    )
    args = parser.parse_args()

    data = load_dataset(HUMAN_LABELLED if args.human_labelled else FULL_DATASET)
    failing, clean = split_by_annotation(data)
    print(f"{len(failing)} traces with at least one failure mode, {len(clean)} clean")

    grouped = defaultdict(list)
    for item in failing:
        mas = item.get("mas_name", "unknown")
        benchmark = item.get("benchmark_name", "unknown")
        if args.mas_name and mas != args.mas_name:
            continue
        if benchmark not in args.benchmarks:
            continue
        grouped[(mas, benchmark)].append(item)

    if not grouped:
        raise SystemExit("nothing matched; check --mas_name and --benchmarks")

    for (mas, benchmark), samples in sorted(grouped.items()):
        out_dir = args.output_dir / mas / benchmark
        out_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            trace_id = sample.get("trace_id")
            trajectory = sample.get("trace", {}).get("trajectory", "")
            with open(out_dir / f"{trace_id}.txt", "w", encoding="utf-8") as handle:
                handle.write(trajectory)
        print(f"wrote {len(samples):>4} trajectories to {out_dir}")


if __name__ == "__main__":
    main()
