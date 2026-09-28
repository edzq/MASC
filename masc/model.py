"""Next-execution reconstruction with a prototype prior (paper Sec. 3.2-3.3)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class PrototypeReconstructor(nn.Module):
    """Predicts the embedding of the next execution step from the history.

    A frozen decoder-only LLM consumes projected step embeddings via
    ``inputs_embeds`` and its hidden states are projected back to the embedding
    space, giving ``x_hat_t = f_theta(LLM(q_tilde, h_tilde_1..t-1))`` (Eq. 7).
    A learnable prototype ``p`` anchors the reconstructions to the centroid of
    normal steps (Eq. 8) and is refined by single-/multi-head attention with
    ``p`` as query and the reconstructions as keys/values.

    Only ``input_proj``, ``output_proj``, ``prototype`` and ``attn`` are
    trainable when ``freeze_backbone=True`` (the default, and the setting used
    for all reported numbers).

    Args:
        emb_dim: Dimensionality of the sentence embeddings (``D``).
        base_model: HuggingFace id or local path of the frozen backbone.
        freeze_backbone: Freeze every backbone parameter.
        device_map: Passed to ``from_pretrained``; ``"auto"`` shards large
            backbones across the visible GPUs. Use ``None`` to load on one
            device and move the module yourself.
        torch_dtype: Backbone dtype. Projections and scores stay in fp32.
        num_heads: Heads of the prototype-update attention.
    """

    def __init__(
        self,
        emb_dim: int,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        freeze_backbone: bool = True,
        device_map: Optional[str] = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        load_kwargs = {"trust_remote_code": True}
        if device_map is not None:
            load_kwargs.update({"device_map": device_map, "torch_dtype": torch_dtype})
        self.backbone = AutoModel.from_pretrained(base_model, **load_kwargs)

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone

        self.emb_dim = emb_dim
        self.backbone_hidden = self.backbone.config.hidden_size
        self.input_proj = nn.Linear(emb_dim, self.backbone_hidden)
        self.output_proj = nn.Linear(self.backbone_hidden, emb_dim)
        # Reconstructions and anomaly scores are computed in fp32 for stability,
        # even when the backbone runs in bf16.
        self.out_dtype = torch.float32

        # Prototype of normality: Gaussian init, updated by gradient descent.
        self.prototype = nn.Parameter(torch.randn(1, emb_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, batch_first=True
        )

    def trainable_parameters(self):
        """Yields the parameters an optimizer should own."""
        return (p for p in self.parameters() if p.requires_grad)

    def _align_heads_to(self, device: torch.device) -> None:
        """Keep the trainable heads on the batch's device.

        Necessary because ``device_map="auto"`` may place the backbone across
        several devices while the heads are created on CPU.
        """
        self.input_proj = self.input_proj.to(device)
        self.output_proj = self.output_proj.to(device)
        self.attn = self.attn.to(device)

    def forward(
        self, embeddings: torch.Tensor, train_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct the next-step embedding for every position.

        Args:
            embeddings: ``(B, L, D)`` context embeddings.
            train_mode: If ``True`` the prototype is refined by attending over
                the reconstructions and the refined vector is returned (it
                carries the gradient for ``L_proto``). If ``False`` the learned
                prototype is returned unchanged.

        Returns:
            ``(pred, proto)`` where ``pred`` is ``(B, L, D)`` and ``proto`` is
            ``(B, D)``.
        """
        device = embeddings.device
        self._align_heads_to(device)

        hidden_in = self.input_proj(embeddings)
        hidden_in = hidden_in.to(dtype=getattr(self.backbone, "dtype", hidden_in.dtype))

        batch, length, _ = hidden_in.size()
        # Trajectories are processed one at a time, so there is no padding.
        attn_mask = torch.ones((batch, length), dtype=torch.long, device=device)

        outputs = self.backbone(inputs_embeds=hidden_in, attention_mask=attn_mask)
        pred = self.output_proj(outputs.last_hidden_state.to(self.out_dtype))

        prototype = self.prototype.to(device)
        if train_mode:
            query = prototype.unsqueeze(0).expand(batch, -1, -1)  # (B, 1, D)
            proto_out, _ = self.attn(query=query, key=pred, value=pred)
            return pred, proto_out.squeeze(1)
        return pred, prototype.expand(batch, -1)


def reconstruction_and_prototype_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    proto_out: torch.Tensor,
    lambda_proto: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Total training objective ``L = L_recon + lambda * L_proto`` (Eq. 11).

    ``L_recon`` is the MSE between predicted and realised step embeddings
    (Eq. 9). ``L_proto`` pulls the attention-refined prototype towards the mean
    of the realised normal steps, which is the empirical centroid of normality
    that Eq. 10 regularises towards.

    Args:
        pred: ``(B, L, D)`` reconstructions.
        target: ``(B, L, D)`` realised embeddings.
        proto_out: ``(B, D)`` attention-refined prototype.
        lambda_proto: Weight of the prototype term.

    Returns:
        ``(total, recon, proto)`` losses.
    """
    recon_loss = F.mse_loss(pred, target)
    proto_loss = F.mse_loss(proto_out, target.mean(dim=1))
    return recon_loss + lambda_proto * proto_loss, recon_loss, proto_loss


def train_epoch(
    model: PrototypeReconstructor,
    trajectories,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
    lambda_proto: float = 0.1,
) -> float:
    """One pass over the normal trajectories; returns the mean total loss.

    Training is fully unsupervised: each trajectory contributes the
    teacher-forced next-step prediction task, with no error labels involved.
    Trajectories shorter than two steps carry no next-step target and are
    skipped.
    """
    model.train()
    if model.freeze_backbone:
        # ``model.train()`` would otherwise re-enable dropout in the backbone.
        model.backbone.eval()

    total_loss, count = 0.0, 0
    for seq in trajectories:
        if seq.size(0) < 2:
            continue

        seq = seq.to(device).unsqueeze(0)  # (1, L, D)
        target = seq[:, 1:, :]
        pred, proto_out = model(seq[:, :-1, :], train_mode=True)
        if pred.numel() == 0:
            continue

        loss, _, _ = reconstruction_and_prototype_loss(
            pred, target, proto_out, lambda_proto=lambda_proto
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(1, count)


def save_heads(model: PrototypeReconstructor, path) -> None:
    """Save only the trainable parameters.

    The frozen backbone and encoder are reloaded from their HuggingFace ids, so
    a checkpoint is a few MB rather than tens of GB.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "emb_dim": model.emb_dim,
            "input_proj": model.input_proj.state_dict(),
            "output_proj": model.output_proj.state_dict(),
            "attn": model.attn.state_dict(),
            "prototype": model.prototype.detach().cpu(),
        },
        path,
    )


def load_heads(model: PrototypeReconstructor, path, strict: bool = True) -> None:
    """Load parameters written by :func:`save_heads` into ``model``."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if strict and state["emb_dim"] != model.emb_dim:
        raise ValueError(
            f"checkpoint emb_dim {state['emb_dim']} != model emb_dim {model.emb_dim}; "
            "the checkpoint was trained with a different sentence encoder"
        )
    model.input_proj.load_state_dict(state["input_proj"])
    model.output_proj.load_state_dict(state["output_proj"])
    model.attn.load_state_dict(state["attn"])
    with torch.no_grad():
        model.prototype.copy_(state["prototype"].to(model.prototype.device))
