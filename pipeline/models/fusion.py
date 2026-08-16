"""
models/fusion.py

Two independent, swappable pieces:

    1. ViewPooling  -- aggregates (B, V, D) per-view embeddings into (B, D).
       Starts as simple mean/max pooling (MVCNN-style baseline per spec 7.3),
       with an attention-based alternative that can be swapped in via
       model.yaml's `pooling_method` WITHOUT touching any other file.

    2. GeometryFusion -- concatenates a normalized geometry feature vector
       onto the pooled image embedding. Fully disableable via
       model.yaml's `use_geometry_fusion: false`, in which case it's a no-op
       passthrough (spec 7.4 + section 14 ablation framework).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ViewPooling(nn.Module):
    """Base class for all view-pooling strategies. Subclasses implement forward()."""

    def forward(self, view_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            view_embeddings: (B, V, D)
        Returns:
            (B, D) pooled embedding.
        """
        raise NotImplementedError

    @property
    def output_dim_multiplier(self) -> int:
        """
        Some pooling strategies (e.g. concatenating mean+max) could change the
        output dimensionality relative to D. Default 1 (output dim == D).
        Exposed so fusion/heads can size their input layer correctly if a
        future pooling method changes this.
        """
        return 1


class MeanPooling(ViewPooling):
    def forward(self, view_embeddings: torch.Tensor) -> torch.Tensor:
        return view_embeddings.mean(dim=1)


class MaxPooling(ViewPooling):
    def forward(self, view_embeddings: torch.Tensor) -> torch.Tensor:
        return view_embeddings.max(dim=1).values


class AttentionPooling(ViewPooling):
    """
    Learned attention-based pooling over the V view embeddings: a small MLP
    scores each view, scores are softmax-normalized across V, and the pooled
    output is the weighted sum. This is the drop-in replacement for mean/max
    pooling referenced in spec 7.3 (exp05_attention_pooling.yaml uses this).
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, view_embeddings: torch.Tensor) -> torch.Tensor:
        # view_embeddings: (B, V, D)
        scores = self.score_mlp(view_embeddings).squeeze(-1)         # (B, V)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)          # (B, V, 1)
        pooled = (view_embeddings * weights).sum(dim=1)                # (B, D)
        return pooled


POOLING_REGISTRY = {
    "mean": MeanPooling,
    "max": MaxPooling,
    "attention": AttentionPooling,
}


def get_pooling(pooling_method: str, embedding_dim: int) -> ViewPooling:
    """Factory used by models/factory.py. Attention pooling needs embedding_dim to build its MLP."""
    if pooling_method not in POOLING_REGISTRY:
        raise ValueError(
            f"Unknown pooling_method '{pooling_method}'. Options: {list(POOLING_REGISTRY.keys())}"
        )
    cls = POOLING_REGISTRY[pooling_method]
    if cls is AttentionPooling:
        return AttentionPooling(embedding_dim=embedding_dim)
    return cls()


class GeometryFusion(nn.Module):
    """
    Concatenates a (already-normalized) geometry feature vector onto the pooled
    image embedding. When `enabled=False`, forward() is a pure passthrough --
    this is what lets exp01_baseline.yaml train on images only with zero code
    changes elsewhere (just a config flag).
    """

    def __init__(self, enabled: bool, geometry_dim: int = 0):
        super().__init__()
        self.enabled = enabled
        self.geometry_dim = geometry_dim if enabled else 0

    @property
    def output_dim_delta(self) -> int:
        """How many extra dims this fusion adds on top of the pooled embedding dim."""
        return self.geometry_dim

    def forward(self, pooled_embedding: torch.Tensor, geometry_vector: torch.Tensor | None) -> torch.Tensor:
        if not self.enabled or geometry_vector is None:
            return pooled_embedding
        return torch.cat([pooled_embedding, geometry_vector], dim=-1)


class GeometryOnlyProjection(nn.Module):
    """
    Used only for the "geometry only" ablation (spec 14.4 case 3): when the
    visual backbone is entirely frozen/bypassed, this projects the raw geometry
    vector up to a workable hidden dimension so the heads still receive a
    reasonably-sized input, instead of the tiny raw feature count.
    """

    def __init__(self, geometry_dim: int, output_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(geometry_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, geometry_vector: torch.Tensor) -> torch.Tensor:
        return self.proj(geometry_vector)
