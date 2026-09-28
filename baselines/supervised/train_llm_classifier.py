#!/usr/bin/env python
"""Supervised LLM-encoder classifier baseline for step-level error detection.

Represents each step by the hidden state of an open-weight LLM and trains a
linear/MLP head on top while the backbone stays frozen -- the
"LLM Classifier" row of Table 1. Like the BERT baseline it is trained on
*individual* steps with error labels and no interaction history, which is the
comparison MASC is measured against (paper Sec. 4.1).

Default hyperparameters follow Table 6 of the paper. Note it needs enough VRAM
to hold the backbone in fp16 (~16 GB for an 8B model).

Example:
    python baselines/supervised/train_llm_classifier.py --subset Hand-Crafted \\
        --model_path meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#: Per-subset settings from Table 6 of the paper. ``epochs`` differs between the
#: w/ GT and w/o GT conditions; the w/o GT value is used as the default.
PRESETS = {
    "Hand-Crafted": {"epochs": 10, "lr": 5e-5, "weight_decay": 0.05, "batch_size": 50},
    "Algorithm-Generated": {"epochs": 8, "lr": 1e-4, "weight_decay": 0.05, "batch_size": 50},
}


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
    """Flatten trajectories into per-step examples up to and including the error."""
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


def load_model(model_path: str, device: str, max_length: int):
    """Load the backbone with a single-logit classification head, backbone frozen."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        # Causal-LM tokenizers ship without a pad token, but batching needs one.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        torch_dtype=torch.float16,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    trainable = []
    for name, param in model.named_parameters():
        # ``score`` is the classification head; everything else stays frozen.
        param.requires_grad = "score" in name
        if param.requires_grad:
            trainable.append(name)
    print(f"[model] trainable layers: {trainable}")
    model.config.max_length = max_length
    return model, tokenizer


def encode(tokenizer, texts, device, max_length: int):
    return tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)


def train_one_epoch(
    model, tokenizer, loader, optimizer, criterion, device, max_length,
    accumulate_epoch_gradient: bool,
) -> float:
    """One training pass.

    With ``accumulate_epoch_gradient`` the whole epoch's losses are averaged into
    a single optimizer step -- full-batch gradient descent, which is how the
    original research script ran. Otherwise each mini-batch takes its own step,
    the standard behaviour and the default here.
    """
    model.train()
    total_loss, losses = 0.0, []
    optimizer.zero_grad()

    for texts, labels in tqdm(loader, leave=False):
        inputs = encode(tokenizer, texts, device, max_length)
        labels = labels.float().to(device)
        logits = model(**inputs).logits.squeeze(-1)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        if accumulate_epoch_gradient:
            losses.append(loss)
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    if accumulate_epoch_gradient and losses:
        torch.stack(losses).mean().backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, tokenizer, loader, device, max_length):
    """Step-level metrics; AUCs use the raw logits, the rest the 0.5 threshold."""
    model.eval()
    y_true, y_pred, y_score = [], [], []
    for texts, labels in tqdm(loader, leave=False):
        inputs = encode(tokenizer, texts, device, max_length)
        logits = model(**inputs).logits.squeeze(-1).float()
        y_score.extend(logits.cpu().tolist())
        y_pred.extend((torch.sigmoid(logits).cpu() > 0.5).int().tolist())
        y_true.extend(labels.tolist())
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
    parser.add_argument("--model_path", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Backbone id or local path (e.g. Qwen/Qwen2.5-7B-Instruct).")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument(
        "--accumulate_epoch_gradient", action="store_true",
        help="One optimizer step per epoch instead of per mini-batch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="outputs/baselines/llm_classifier")
    args = parser.parse_args()

    preset = PRESETS[args.subset]
    epochs = args.epochs or preset["epochs"]
    lr = args.lr or preset["lr"]
    weight_decay = preset["weight_decay"] if args.weight_decay is None else args.weight_decay
    batch_size = args.batch_size or preset["batch_size"]

    torch.manual_seed(args.seed)
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

    model, tokenizer = load_model(args.model_path, args.device, args.max_length)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    print(f"[train] epochs={epochs} lr={lr} weight_decay={weight_decay} batch={batch_size}")

    best = None
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, tokenizer, train_loader, optimizer, criterion, args.device,
            args.max_length, args.accumulate_epoch_gradient,
        )
        metrics = evaluate(model, tokenizer, test_loader, args.device, args.max_length)
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
