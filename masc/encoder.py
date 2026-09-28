"""Frozen sentence encoder used for contextual encoding (paper Sec. 3.1)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEncoder(nn.Module):
    """Encodes queries / agent roles / agent outputs into unit-norm vectors.

    The encoder is frozen: only the projection layers and the prototype of
    :class:`~masc.model.PrototypeReconstructor` are trained.

    Args:
        model_name: HuggingFace id or local path of the encoder. The paper uses
            ``all-MiniLM-L6-v2`` for every reported number.
        device: Device the encoder is placed on.
        max_length: Token truncation length. Agent outputs in Who&When are long,
            so truncation is expected and matches the reported setup.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ENCODER,
        device: str = "cuda",
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(device)
        self.encoder.eval()
        self.device = device
        self.max_length = max_length
        self.hidden_size = self.encoder.config.hidden_size

    @torch.no_grad()
    def encode(self, sentences: Sequence[str]) -> torch.Tensor:
        """Encode a list of strings into an ``(n, hidden_size)`` tensor.

        Uses the ``[CLS]`` position of the last hidden state followed by L2
        normalisation, so that cosine similarity reduces to a dot product.
        """
        tokens = self.tokenizer(
            list(sentences),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.encoder(**tokens)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=-1)
