#!/usr/bin/env python
"""Supervised BERT-classifier baseline for step-level error detection.

A frozen-or-finetuned ``all-MiniLM-L6-v2`` sentence encoder with a trainable MLP
head, trained on *individual* steps: every step up to and including the annotated
error becomes one labelled example, shuffled into mini-batches. Unlike MASC this
baseline sees the error labels and no interaction history, which is exactly the
comparison the paper draws (Sec. 4.1, "strong supervised models").

Default hyperparameters follow Table 6 of the paper; pass ``--subset`` to pick
the per-subset values.

Example:
    python baselines/supervised/train_bert.py --subset Hand-Crafted
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

#: Per-subset settings from Table 6 of the paper.
PRESETS = {
    "Hand-Crafted": {"epochs": 50, "lr": 1e-5, "weight_decay": 0.01, "batch_size": 32},
    "Algorithm-Generated": {"epochs": 50, "lr": 2e-5, "weight_decay": 0.01, "batch_size": 64},
}


class SentenceTransformerMLPClassifier(nn.Module):
    """Sentence embedding -> MLP -> single logit for 'is this step the error'."""

    def __init__(self, model_name: str = DEFAULT_ENCODER, hidden_dim: int = 256,
                 freeze_encoder: bool = True) -> None:
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sentences: Sequence[str]) -> torch.Tensor:
        device = next(self.classifier.parameters()).device
        embeddings = self.encoder.encode(list(sentences), normalize_embeddings=True)
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
        return self.classifier(embeddings).squeeze(-1)


class StepClassificationDataset(Dataset):
    """Flat list of (step text, label) pairs."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], self.labels[idx]


def build_dataset(folder_path: str | os.PathLike, file_list: Sequence[str]):
    """Flatten trajectories into per-step examples.

    Keeps every step up to *and including* the annotated error, so each
    trajectory contributes exactly one positive example.
    """
    texts: List[str] = []
    labels: List[int] = []
    used: List[str] = []
    for filename in file_list:
        with open(Path(folder_path) / filename, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        history = record.get("history", [])
        mistake_step = int(record.get("mistake_step", -1))
        for i in range(min(mistake_step + 1, len(history))):
            texts.append(history[i].get("content", ""))
            labels.append(1 if i == mistake_step else 0)
        used.append(filename)
    return texts, labels, used


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for texts, labels in dataloader:
        labels = labels.float().to(device)
        loss = criterion(model(texts), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(model, dataloader):
    """Step-level metrics; AUCs use the raw logits, the rest the 0.5 threshold."""
    model.eval()
    y_true, y_pred, y_score = [], [], []
    for texts, labels in dataloader:
        logits = model(texts)
        y_pred.extend((torch.sigmoid(logits).cpu() > 0.5).int().tolist())
        y_true.extend(labels.tolist())
        y_score.extend(logits.cpu().tolist())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_root", default="data/Who&When")
    parser.add_argument("--subset", default="Hand-Crafted", choices=list(PRESETS))
    parser.add_argument("--split_dir", default="data/splits")
    parser.add_argument("--encoder_model", default=DEFAULT_ENCODER)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--finetune_encoder", action="store_true",
                        help="Also train the sentence encoder (off by default).")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/baselines/bert")
    args = parser.parse_args()

    preset = PRESETS[args.subset]
    epochs = args.epochs or preset["epochs"]
    lr = args.lr or preset["lr"]
    weight_decay = preset["weight_decay"] if args.weight_decay is None else args.weight_decay
    batch_size = args.batch_size or preset["batch_size"]

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset_dir = Path(args.data_root) / args.subset
    split_dir = Path(args.split_dir) / args.subset

    with open(split_dir / "train_files.json", encoding="utf-8") as handle:
        train_files = json.load(handle)
    with open(split_dir / "test_files.json", encoding="utf-8") as handle:
        test_files = json.load(handle)

    train_texts, train_labels, _ = build_dataset(subset_dir, train_files)
    test_texts, test_labels, _ = build_dataset(subset_dir, test_files)
    print(
        f"[data] {args.subset}: {len(train_texts)} train steps "
        f"({sum(train_labels)} errors) / {len(test_texts)} test steps "
        f"({sum(test_labels)} errors)"
    )

    train_loader = DataLoader(
        StepClassificationDataset(train_texts, train_labels),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        StepClassificationDataset(test_texts, test_labels), batch_size=batch_size
    )

    model = SentenceTransformerMLPClassifier(
        args.encoder_model, args.hidden_dim, freeze_encoder=not args.finetune_encoder
    ).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    print(f"[train] epochs={epochs} lr={lr} weight_decay={weight_decay} batch={batch_size}")

    best = None
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, test_loader)
        rendered = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"[epoch {epoch:>3}] loss {loss:.4f} | {rendered}")
        if best is None or metrics["auc_roc"] > best["auc_roc"]:
            best = {**metrics, "epoch": epoch}

    print(f"\n[best] epoch {best['epoch']}: " +
          " | ".join(f"{k}: {v:.4f}" for k, v in best.items() if k != "epoch"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{args.subset}.json", "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "best": best}, handle, indent=2)


if __name__ == "__main__":
    main()
