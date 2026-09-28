#!/usr/bin/env python
"""EXPLORATORY: learnable context selection for step-level error detection.

Not part of the paper's reported results. Where MASC conditions on the whole
history via next-execution reconstruction, this variant learns *which* past
steps to attend to: a scorer ranks the buffered history against the current step
and the top-K are pooled into a context vector that a supervised head then
classifies. It is kept here because it motivated the context-aware framing, but
it is supervised, needs step labels, and is not the method the paper evaluates.

For the paper's method see ``masc/`` and ``scripts/train_detector.py``.

Example:
    python experimental/learnable_step_selector.py --subset Algorithm-Generated
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
class AdaptiveStepSelectorWeighted(nn.Module):
    def __init__(self, encoder_name=DEFAULT_ENCODER, embedding_dim=384, hidden_dim=256, k=5):
        super().__init__()
        self.k = k
        self.embedding_dim = embedding_dim
        self.encoder = SentenceTransformer(encoder_name)
        for param in self.encoder.parameters():
            param.requires_grad = False  # freeze the encoder

        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # relevance score
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # binary: is this step anomalous
        )

    def encode(self, texts):
        with torch.no_grad():
            emb = self.encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        return emb  # shape: (N, D)

    def forward(self, current_step, buffer_steps, prev_context_texts=None):
        """
        current_step: str
        buffer_steps: List[str] -- every past step (the buffer)
        prev_context_texts: List[str] or None -- steps retained from the previous context
        """
        current_emb = self.encode([current_step])  # (1, D)
        buffer_embs = self.encode(buffer_steps)  # (N, D)

        if prev_context_texts:
            prev_context_embs = self.encode(prev_context_texts)  # (M, D)
            all_candidates = torch.cat([buffer_embs, prev_context_embs], dim=0)
        else:
            all_candidates = buffer_embs  # (N, D)

        N = all_candidates.shape[0]
        current_repeat = current_emb.repeat(N, 1)  # (N, D)
        pairwise = torch.cat([current_repeat, all_candidates], dim=1)  # (N, 2D)

        # score the candidates and softmax-normalise
        scores = self.scorer(pairwise).squeeze(-1)  # (N,)
        weights = torch.softmax(scores, dim=0)  # (N,)

        # take the top-K
        topk_idx = torch.topk(weights, k=min(self.k, N)).indices
        selected_embs = all_candidates[topk_idx]  # (k, D)
        selected_weights = weights[topk_idx].unsqueeze(-1)  # (k, 1)

        # weighted aggregation of the selected context
        context = (selected_embs * selected_weights).sum(dim=0)  # (D,)

        # classify: anomalous or not
        logit = self.classifier(context)  # (1,)
        return logit.squeeze(), [buffer_steps[i] for i in topk_idx.tolist()]  # also return which steps were selected


class AttentionScorer(nn.Module):
    def __init__(self, embedding_dim, scorer_type='dot', hidden_dim=128):
        super().__init__()
        self.scorer_type = scorer_type
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

        if scorer_type == 'mlp':
            self.score_mlp = nn.Sequential(
                nn.Linear(2 * embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, query, context):
        """
        query: (D,)
        context: (k, D)
        return: context_vec (D,), attn_weights (k,)
        """
        if self.scorer_type == 'dot':
            q = self.query_proj(query)  # (D,)
            k = self.key_proj(context)  # (k, D)
            scores = torch.matmul(k, q) / (q.shape[0] ** 0.5)  # (k,)

        elif self.scorer_type == 'mlp':
            q_repeat = query.unsqueeze(0).expand_as(context)  # (k, D)
            score_input = torch.cat([q_repeat, context], dim=1)  # (k, 2D)
            scores = self.score_mlp(score_input).squeeze(-1)  # (k,)

        else:
            raise ValueError(f"Unknown scorer_type: {self.scorer_type}")

        attn_weights = F.softmax(scores, dim=0)  # (k,)
        v = self.value_proj(context)  # (k, D)
        context_vec = torch.sum(attn_weights.unsqueeze(1) * v, dim=0)  # (D,)
        return context_vec, attn_weights


class StepSelector(nn.Module):
    def __init__(self, embedding_model, embedding_dim, k=5, scorer_type='dot'):
        super().__init__()
        self.embedding_model = embedding_model  # e.g. SentenceTransformer, BERT encoder, etc.
        self.embedding_dim = embedding_dim
        self.k = k
        self.buffer = []  # all history embeddings
        self.context_window = []  # indices of top-k relevant steps

        self.attn_scorer = AttentionScorer(embedding_dim, scorer_type=scorer_type)
        self.classifier = nn.Linear(embedding_dim, 2)

    def embed_step(self, step_text):
        with torch.no_grad():  # embedding model is frozen
            emb = self.embedding_model.encode(step_text, convert_to_tensor=True)
        return emb  # (D,)

    def update_buffer(self, embedding):
        self.buffer.append(embedding)

    def select_context_window(self, current_emb):
        if not self.buffer:
            return torch.zeros((0, self.hidden_dim), device=current_emb.device)

        buffer_tensor = torch.stack(self.buffer)  # (n, D)

        # Scoring
        attn_weights = self.scorer(current_emb, buffer_tensor)  # (n,)

        # context-window update policy
        if len(self.buffer) <= self.k:
            context_tensor = buffer_tensor  # (n, D)
            weights = attn_weights
        else:
            topk_indices = torch.topk(attn_weights, self.k).indices
            context_tensor = buffer_tensor[topk_indices]  # (k, D)
            weights = attn_weights[topk_indices]  # (k,)

        values = self.value_proj(context_tensor)  # (k or n, D)
        context_vec = torch.sum(weights.unsqueeze(-1) * values, dim=0)  # (D,)

        return context_vec

    def forward(self, step_text):
        # 1. encode the current step
        current_emb = self.embed_step(step_text)  # (D,)

        # 2. refresh the context window from the buffer
        context_vec = self.select_context_window(current_emb)  # (D,) aggregated vector

        # 3. concatenate and classify
        combined = torch.cat([current_emb, context_vec], dim=-1)  # (2D,)
        pred = self.classifier(combined).sigmoid()  # (1,)

        # 4. update the buffer
        self.buffer.append(current_emb.detach())

        return pred.squeeze()


class StepSelectorWithQuery(nn.Module):
    def __init__(self, encoder, hidden_dim=128, k=3, use_attention=True):
        super().__init__()
        self.encoder = encoder  # frozen SentenceTransformer
        self.k = k
        self.buffer = []  # embeddings of every past step
        self.context_window = []  # indices into the buffer forming the current context

        # optional: attention- or MLP-based similarity
        self.use_attention = use_attention
        if use_attention:
            self.q_proj = nn.Linear(encoder.get_sentence_embedding_dimension(), hidden_dim)
            self.k_proj = nn.Linear(encoder.get_sentence_embedding_dimension(), hidden_dim)
            self.v_proj = nn.Linear(encoder.get_sentence_embedding_dimension(), hidden_dim)
            self.foward_mlp = nn.Linear(hidden_dim, encoder.get_sentence_embedding_dimension())
        else:
            self.sim_scorer = nn.Sequential(
                nn.Linear(2 * encoder.get_sentence_embedding_dimension(), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * encoder.get_sentence_embedding_dimension(), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # binary output
        )

    def embed_step(self, text):
        embeddings = self.encoder.encode(text, normalize_embeddings=True)
        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(next(self.classifier.parameters()).device)
        return embeddings

    def select_context_window(self, current_emb):
        if not self.buffer:
            return torch.empty(0, current_emb.size(0), device=current_emb.device), None

        buffer_tensor = torch.stack([t.view(-1) for t in self.buffer])  # every entry must be [384]
        n = buffer_tensor.size(0)
        if self.use_attention:
            q = self.q_proj(current_emb)  # (H,)
            k = self.k_proj(buffer_tensor)  # (n, H)
            v = self.v_proj(buffer_tensor)  # (n, H)
            scores = torch.softmax((q @ k.T) / (k.size(-1) ** 0.5), dim=-1)  # (n,)
            context_vec = scores @ v  # (H,)
            if n > self.k:
                topk_indices = torch.topk(scores, self.k).indices
                # shapes: scores [1, T], v [T, D]

                topk_scores = torch.gather(scores, dim=1, index=topk_indices)  # [1, k]
                v_topk = v[topk_indices.squeeze(0)]  # [k, 128]
                context_vec = topk_scores @ v_topk  # [128]

                # context_vec = context_tensor.mean(dim=0)  # simpler alternative

        else:

            current_expand = current_emb.unsqueeze(0).expand(n, -1)
            pair_input = torch.cat([current_expand, buffer_tensor], dim=1)  # (n, 2D)
            scores = self.sim_scorer(pair_input).squeeze()  # (n,)
            context_vec = scores @ buffer_tensor
            if n > self.k:
                topk_indices = torch.topk(scores, self.k).indices
                context_vec = context_vec[topk_indices]  # (k, D)
                # context_vec = context_tensor.mean(dim=0)  # simpler alternative

        return context_vec

    def forward(self, step_text, query_text):
        # 1. embed the current step and the query
        current_emb = self.embed_step(step_text)
        query_emb = self.embed_step(query_text)

        # 2. on the first step the buffer is empty, so seed it with the query
        if len(self.buffer) == 0:
            self.buffer.append(query_emb)

        # 3. select the context
        context_vec = self.select_context_window(current_emb)
        if context_vec is None or context_vec.nelement() == 0:
            context_vec = torch.zeros_like(current_emb)

        # 4. concatenate the current step with the context and classify
        context_vec = self.foward_mlp(context_vec)
        x = torch.cat([current_emb, context_vec], dim=-1)
        logit = self.classifier(x)

        # 5. append the current step to the buffer
        self.buffer.append(current_emb)  # add .detach() here to stop gradients flowing through history

        return logit

class StepwiseEpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, step_texts, step_labels):
        """
        step_texts: List of episodes, each episode is List[str]
        step_labels: List of episodes, each episode is List[int]
        """
        self.episodes = list(zip(step_texts, step_labels))

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]  # one full trajectory: (steps, labels)
def split_json_files(folder_path, train_ratio=0.2, seed=42):
    random.seed(seed)
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    random.shuffle(all_files)

    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    return train_files, test_files
def build_dataset_from_files(folder_path, file_list):
    all_sentences = []
    all_labels = []
    file_used = []

    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            history = data.get("history", [])
            mistake_step = int(data.get("mistake_step", -1))
                # build the step labels
            for i in range(mistake_step + 1):  # includes the error step
                all_sentences.append(history[i]["content"])
                all_labels.append(1 if i == mistake_step else 0)
            file_used.append(filename)

    return all_sentences, all_labels, file_used
def build_dataset_from_single_files(folder_path, file_list):
    all_sentences = []
    all_labels = []
    file_used = []

    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        st = []
        s_label = []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "Hand-Crafted" not in folder_path:
                query = data.get("question", [])
                st.append(query)
                s_label.append(0)
            history = data.get("history", [])
            mistake_step = int(data.get("mistake_step", -1))

                # build the step labels
            for i in range(mistake_step + 1):  # includes the error step
                st.append(history[i]["content"])
                s_label.append(1 if i == mistake_step else 0)
            file_used.append(filename)
        all_sentences.append(st)
        all_labels.append(s_label)
    return all_sentences, all_labels, file_used


def evaluate_stepwise_model(model, dataset, device='cuda'):
    model.to(device)
    model.eval()

    all_preds, all_trues = [], []
    all_score = []
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        for steps, labels in dataloader:
            query_text = steps[0]
            steps = steps[1:]
            labels = labels[1:]
            model.buffer = []

            preds_in_sample, trues_in_sample = [], []
            pred_in_score = []
            for step_text, label in zip(steps, labels):
                logit = model(step_text, query_text)
                pred_in_score.append(logit.cpu().tolist()[0])
                pred = (torch.sigmoid(logit) > 0.5).long().item()
                preds_in_sample.append(pred)
                trues_in_sample.append(label.item())
            all_score.extend(pred_in_score)
            all_preds.extend(preds_in_sample)
            all_trues.extend(trues_in_sample)

    # test metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_trues, all_preds, average='binary', zero_division=0)
    acc = accuracy_score(all_trues, all_preds)
    AUROC = roc_auc_score(all_trues, all_score)
    AUPRC = average_precision_score(all_trues, all_score)
    print(f"[Test Set] Acc: {acc:.8f} | P: {precision:.8f} | R: {recall:.8f} | F1: {f1:.8f} | AUROC: {AUROC:.4f} | AUPRC: {AUPRC:.4f}")

def train_stepwise_model(model, dataset, test_dataset, epochs=5, lr=5e-3, batch_size=1, device='cuda'):
    model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(list(model.k_proj.parameters())+list(model.q_proj.parameters())+list(model.v_proj.parameters())+list(model.classifier.parameters())+list(model.foward_mlp.parameters()), lr=lr)

    # Positive/negative ratio of the labels. Errors are rare, so an unweighted
    # BCE is dominated by the negatives; pass this as BCEWithLogitsLoss(
    # pos_weight=...) to counter that. Left unweighted here to match the runs
    # this code was used for.
    all_labels = [label for episode in dataset for label in episode[1]]
    positives, negatives = sum(l == 1 for l in all_labels), sum(l == 0 for l in all_labels)
    print(f"[data] {positives} positive / {negatives} negative steps")
    criterion = nn.BCEWithLogitsLoss()

    # for epoch in range(epochs):
    #     model.train()
    #     total_loss, total_steps = 0, 0
    #     all_preds, all_trues = [], []
    #
    #     for steps, labels in dataloader:
    #         steps = steps[0]  # batch_size=1
    #         labels = labels[0]
    #         query_text = steps[0]  # first step acts as the query
    #         model.buffer = []  # clear the buffer before each trajectory
    #
    #         optimizer.zero_grad()
    #         losses = []
    #
    #         for step_text, label in zip(steps, labels):
    #             logit = model(step_text, query_text)
    #             loss = criterion(logit, torch.tensor(label, dtype=torch.float, device=device))
    #             losses.append(loss)
    #
    #             pred = (torch.sigmoid(logit) > 0.5).long().item()
    #             all_preds.append(pred)
    #             all_trues.append(label)
    #
    #         total_loss += torch.stack(losses).mean().item()
    #         torch.stack(losses).mean().backward()
    #         optimizer.step()
    #         total_steps += 1
    #
    #     precision, recall, f1, _ = precision_recall_fscore_support(all_trues, all_preds, average='binary')
    #     acc = accuracy_score(all_trues,all_preds)
    #     print(f"[Epoch {epoch+1}] Loss: {total_loss/total_steps:.4f} | Acc:{acc:.8f}| P: {precision:.8f} | R: {recall:.8f} | F1: {f1:.8f}")

    for epoch in range(epochs):
        model.train()
        total_loss, total_steps = 0, 0
        all_preds, all_trues = [], []
        all_score = []
        for steps, labels in dataloader:
            query_text = steps[0]
            steps = steps[1:]  # batch_size=1
            labels = labels[1:]
             # the first step acts as the query
            model.buffer = []  # clear the buffer before each trajectory

            optimizer.zero_grad()
            losses = []
            preds_in_sample, trues_in_sample = [], []
            pred_in_score = []
            for step_text, label in zip(steps, labels):
                logit = model(step_text, query_text)
                logit = logit.view(1)  # force shape [1]
                label_tensor = torch.tensor([label], dtype=torch.float, device=device)
                loss = criterion(logit, label_tensor)
                #loss = criterion(torch.tensor([logit.item()]).to(device), torch.tensor(label, dtype=torch.float, device=device))
                losses.append(loss)
                pred_in_score.append(logit.cpu().tolist()[0])
                pred = (torch.sigmoid(logit) > 0.5).long().item()
                preds_in_sample.append(pred)
                trues_in_sample.append(label.item())

            total_loss += torch.stack(losses).mean().item()
            torch.stack(losses).mean().backward()
            optimizer.step()
            total_steps += 1

            all_preds.extend(preds_in_sample)
            all_trues.extend(trues_in_sample)
            all_score.extend(pred_in_score)
        # epoch-level accuracy / F1 / AUC
        precision, recall, f1, _ = precision_recall_fscore_support(all_trues, all_preds, average='binary',
                                                                   zero_division=0)
        acc = accuracy_score(all_trues, all_preds)
        AUROC = roc_auc_score(all_trues, all_score)
        AUPRC = average_precision_score(all_trues, all_score)
        print(
            f"[Epoch {epoch + 1}] Loss: {total_loss / total_steps:.4f} | Acc:{acc:.8f}| P: {precision:.8f} | R: {recall:.8f} | F1: {f1:.8f} | AUROC: {AUROC:.4f} | AUPRC: {AUPRC:.4f}")
        evaluate_stepwise_model(model, test_dataset, device)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_root", default="data/Who&When")
    parser.add_argument(
        "--subset", default="Algorithm-Generated",
        choices=["Hand-Crafted", "Algorithm-Generated"],
    )
    parser.add_argument("--split_dir", default="data/splits")
    parser.add_argument("--encoder_model", default=DEFAULT_ENCODER)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--flat_steps", action="store_true",
        help="Train on flattened individual steps instead of whole trajectories.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_path = os.path.join(args.data_root, args.subset)
    split_dir = os.path.join(args.split_dir, args.subset)

    with open(os.path.join(split_dir, "train_files.json"), encoding="utf-8") as f:
        train_files = json.load(f)
    with open(os.path.join(split_dir, "test_files.json"), encoding="utf-8") as f:
        test_files = json.load(f)

    builder = build_dataset_from_files if args.flat_steps else build_dataset_from_single_files
    train_texts, train_labels, _ = builder(folder_path, train_files)
    test_texts, test_labels, _ = builder(folder_path, test_files)
    print(f"[data] {args.subset}: {len(train_texts)} train / {len(test_texts)} test")

    dataset = StepwiseEpisodeDataset(train_texts, train_labels)
    test_dataset = StepwiseEpisodeDataset(test_texts, test_labels)

    encoder_model = SentenceTransformer(args.encoder_model)
    model = StepSelectorWithQuery(encoder=encoder_model)
    train_stepwise_model(
        model, dataset, test_dataset, epochs=args.epochs, lr=args.lr, device=device
    )


if __name__ == "__main__":
    main()
