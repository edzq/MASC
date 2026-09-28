#!/usr/bin/env python
"""Measure how separable erroneous steps are in embedding space.

This is the preliminary analysis behind the paper's claim that step-level errors
in MAS have *weak semantic separability*: viewed in isolation, an erroneous step
embedding sits inside the cloud of normal ones, so a detector must reason over
the interaction history rather than over single steps.

For each subset it reports:

  * inter-cluster distance -- between the normal and abnormal centroids;
  * intra-cluster distance -- mean pairwise distance, and mean distance to the
    centroid, within each cluster;
  * a separability ratio (inter / intra): the lower, the harder the problem;

each computed twice -- on raw step embeddings, and after a one-neighbour context
aggregation that mixes every step with its nearest neighbour. Results are
written to JSON and, optionally, plotted.

Example:
    python analysis/embedding_separability.py --out_dir figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

SUBSETS = {"HC": "Hand-Crafted", "Alg": "Algorithm-Generated"}
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class StepEncoder:
    """Frozen sentence encoder producing unit-norm step embeddings."""

    def __init__(self, model_name: str = DEFAULT_ENCODER, device: str = "cpu") -> None:
        self.encoder = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def encode(self, sentences) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = self.encoder.encode(sentences, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)


def load_embeddings(
    subset_dir: Path, encoder: StepEncoder
) -> Tuple[np.ndarray, np.ndarray]:
    """Split every step of every trajectory into normal / abnormal embeddings."""
    normal, abnormal = [], []
    for path in tqdm(sorted(subset_dir.glob("*.json")), desc=subset_dir.name):
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        history = [entry.get("content", "") for entry in record.get("history", [])]
        if not history:
            continue
        mistake_step = int(record["mistake_step"])
        for index, embedding in enumerate(encoder.encode(history)):
            (abnormal if index == mistake_step else normal).append(embedding)
    return np.array(normal), np.array(abnormal)


def context_aggregate(embeddings: np.ndarray) -> np.ndarray:
    """Blend every step with its most similar other step, 50/50.

    A deliberately minimal stand-in for context: if even this much context
    improves separability, history-conditioned modelling is worth its cost.
    """
    tensor = torch.as_tensor(embeddings)
    similarity = F.cosine_similarity(tensor.unsqueeze(1), tensor.unsqueeze(0), dim=-1)
    similarity.fill_diagonal_(-1.0)  # never pick the step itself
    nearest = similarity.argmax(dim=-1)
    return (0.5 * tensor + 0.5 * tensor[nearest]).numpy()


def inter_cluster_distance(normal: np.ndarray, abnormal: np.ndarray) -> float:
    """Euclidean distance between the two cluster centroids."""
    return float(np.linalg.norm(normal.mean(axis=0) - abnormal.mean(axis=0)))


def intra_cluster_distance(embeddings: np.ndarray, mode: str = "pairwise") -> float:
    """Mean distance inside one cluster.

    ``pairwise`` averages all pairwise distances; ``center`` averages the
    distances to the centroid.
    """
    if mode == "pairwise":
        return float(pdist(embeddings, metric="euclidean").mean())
    if mode == "center":
        center = embeddings.mean(axis=0)
        return float(np.linalg.norm(embeddings - center, axis=1).mean())
    raise ValueError("mode must be 'pairwise' or 'center'")


def analyse(normal: np.ndarray, abnormal: np.ndarray) -> Dict[str, float]:
    """All distances for one representation of one subset."""
    inter = inter_cluster_distance(normal, abnormal)
    intra_normal = intra_cluster_distance(normal, "pairwise")
    return {
        "inter": inter,
        "intra_normal_pairwise": intra_normal,
        "intra_normal_center": intra_cluster_distance(normal, "center"),
        "intra_abnormal_pairwise": intra_cluster_distance(abnormal, "pairwise"),
        "separability_ratio": inter / intra_normal if intra_normal else float("nan"),
    }


def plot(results: Dict[str, Dict], out_dir: Path) -> None:
    """Bar charts of inter- and intra-cluster distance, raw vs context-aggregated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(results)
    representations = ["raw", "context_aggregated"]
    x = np.arange(len(representations))
    width = 0.8 / max(1, len(keys))
    colors = {"HC": "#45B7D1", "Alg": "#F9A602"}

    for metric, filename, ylabel in [
        ("inter", "inter_distance.pdf", "Inter distance"),
        ("intra_normal_pairwise", "intra_distance.pdf", "Intra distance"),
    ]:
        plt.figure(figsize=(8, 6))
        for i, key in enumerate(keys):
            values = [results[key][rep][metric] for rep in representations]
            offset = (i - (len(keys) - 1) / 2) * width
            plt.bar(x + offset, values, width, label=key, color=colors.get(key))
        plt.xticks(x, ["Raw", "Context-Agg"], fontsize=18)
        plt.yticks(fontsize=14)
        plt.ylabel(ylabel, fontsize=18)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"wrote {out_dir / filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_root", default="data/Who&When")
    parser.add_argument("--encoder_model", default=DEFAULT_ENCODER)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out_dir", type=Path, default=Path("figures"))
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    encoder = StepEncoder(args.encoder_model, device=args.device)
    results: Dict[str, Dict] = {}

    for key, subset in SUBSETS.items():
        subset_dir = Path(args.data_root) / subset
        if not subset_dir.is_dir():
            print(f"skipping missing subset: {subset_dir}")
            continue
        normal, abnormal = load_embeddings(subset_dir, encoder)
        print(f"\n{subset}: {len(normal)} normal steps, {len(abnormal)} abnormal steps")

        results[key] = {
            "raw": analyse(normal, abnormal),
            "context_aggregated": analyse(
                context_aggregate(normal), context_aggregate(abnormal)
            ),
        }
        for representation, metrics in results[key].items():
            rendered = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  {representation:<20} {rendered}")

    if not results:
        raise SystemExit("no subsets found; check --data_root")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "embedding_separability.json"
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwrote {out_json}")
    if not args.no_plot:
        plot(results, args.out_dir)


if __name__ == "__main__":
    main()
