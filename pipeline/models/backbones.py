"""
models/backbones.py

Unified interface for swappable image backbones. Every backbone implements:

    forward(images: Tensor[B, V, 3, H, W]) -> Tensor[B, V, D]

where D (embedding_dim) is fixed per-backbone and exposed via the `embedding_dim`
property. Downstream code (fusion.py) only depends on this interface, never on
which concrete backbone is in use -- so adding a new backbone later requires:

    1. Subclass Backbone, implement forward() and embedding_dim.
    2. Register it in BACKBONE_REGISTRY below with a string key.
    3. Reference that key in model.yaml's `backbone_name` field.

No other file needs to change. This satisfies the spec's requirement that no
single backbone (e.g. DINOv2) is hardcoded as the only option -- the choice is
purely config-driven, since there is no guarantee upfront which backbone will
perform best on this particular small, imbalanced, narrow-domain dataset.

Available backbones:
    - "dinov2_vits14" / "dinov2_vitb14" (default: dinov2_vits14) -- strong visual
      representations without needing huge fine-tuning data (best fit for a
      low-data regime like this one), via torch.hub.
    - "convnext_tiny" / "convnext_small" -- modern CNN alternative, typically
      faster to train than a ViT, useful comparison point on limited data.
    - "efficientnet_b0" -- classic, well-understood CNN baseline for comparison.
    - "clip_vit_b32" -- OpenAI CLIP visual encoder; opens the door to future
      text-based search on the website if ever wanted (not required now).

All backbones download pretrained ImageNet/self-supervised weights on first use
(requires internet -- see network notes in the design doc). Partial fine-tuning
(freeze all but the last N transformer blocks / conv stages) is controlled via
`freeze_backbone_except_last_n_layers` in model.yaml, applied uniformly here via
`_freeze_all_except_last_n`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Backbone(nn.Module, ABC):
    """Abstract base class every concrete backbone must implement."""

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, V, 3, H, W) float tensor, V = views per sample.
        Returns:
            (B, V, D) float tensor of per-view embeddings.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        raise NotImplementedError

    def _freeze_all_except_last_n(self, blocks: list[nn.Module], n: int) -> None:
        """
        Freeze every parameter except the last `n` entries of `blocks` (and
        anything not in `blocks` at all, e.g. patch-embed/stem layers stay frozen).
        Shared helper so every backbone applies partial fine-tuning identically.
        """
        for p in self.parameters():
            p.requires_grad = False

        n = max(0, min(n, len(blocks)))
        for block in blocks[len(blocks) - n:]:
            for p in block.parameters():
                p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"{self.__class__.__name__}: unfroze last {n}/{len(blocks)} blocks "
            f"({trainable:,}/{total:,} params trainable)."
        )

    def _reshape_for_2d_backbone(self, images: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Flatten (B, V, 3, H, W) -> (B*V, 3, H, W) for backbones that only accept 4D input."""
        b, v = images.shape[0], images.shape[1]
        flat = images.reshape(b * v, *images.shape[2:])
        return flat, b, v


class DINOv2Backbone(Backbone):
    """
    DINOv2 ViT-S/14 or ViT-B/14 via torch.hub (facebookresearch/dinov2).
    Default backbone choice per spec section 13: strong self-supervised visual
    representations that need comparatively little fine-tuning data.
    """

    _HUB_NAMES = {
        "dinov2_vits14": ("facebookresearch/dinov2", "dinov2_vits14", 384),
        "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14", 768),
    }

    def __init__(self, variant: str = "dinov2_vits14", freeze_except_last_n: int = 2):
        super().__init__()
        if variant not in self._HUB_NAMES:
            raise ValueError(f"Unknown DINOv2 variant '{variant}'. Options: {list(self._HUB_NAMES)}")

        repo, hub_entry, dim = self._HUB_NAMES[variant]
        self._embedding_dim = dim
        self.model = torch.hub.load(repo, hub_entry)  # requires internet on first call

        # DINOv2 exposes its transformer blocks at self.model.blocks
        blocks = list(self.model.blocks)
        self._freeze_all_except_last_n(blocks, freeze_except_last_n)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat, b, v = self._reshape_for_2d_backbone(images)
        features = self.model(flat)  # (B*V, D) -- CLS token / pooled output
        return features.reshape(b, v, self._embedding_dim)


class ConvNeXtBackbone(Backbone):
    """ConvNeXt (tiny/small) via torchvision, a fast modern CNN alternative to ViT."""

    _VARIANTS = {"convnext_tiny": 768, "convnext_small": 768}

    def __init__(self, variant: str = "convnext_tiny", freeze_except_last_n: int = 2):
        super().__init__()
        import torchvision.models as tvm

        if variant not in self._VARIANTS:
            raise ValueError(f"Unknown ConvNeXt variant '{variant}'. Options: {list(self._VARIANTS)}")

        self._embedding_dim = self._VARIANTS[variant]
        ctor = tvm.convnext_tiny if variant == "convnext_tiny" else tvm.convnext_small
        full_model = ctor(weights="DEFAULT")
        self.features = full_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        stages = list(self.features)  # sequential of ConvNeXt stages
        self._freeze_all_except_last_n(stages, freeze_except_last_n)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat, b, v = self._reshape_for_2d_backbone(images)
        feat_map = self.features(flat)
        pooled = self.pool(feat_map).flatten(1)  # (B*V, D)
        return pooled.reshape(b, v, self._embedding_dim)


class EfficientNetBackbone(Backbone):
    """EfficientNet-B0 via torchvision, classic CNN baseline for comparison."""

    def __init__(self, variant: str = "efficientnet_b0", freeze_except_last_n: int = 2):
        super().__init__()
        import torchvision.models as tvm

        if variant != "efficientnet_b0":
            raise ValueError(f"Unknown EfficientNet variant '{variant}'. Only 'efficientnet_b0' supported.")

        self._embedding_dim = 1280
        full_model = tvm.efficientnet_b0(weights="DEFAULT")
        self.features = full_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        stages = list(self.features)
        self._freeze_all_except_last_n(stages, freeze_except_last_n)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat, b, v = self._reshape_for_2d_backbone(images)
        feat_map = self.features(flat)
        pooled = self.pool(feat_map).flatten(1)
        return pooled.reshape(b, v, self._embedding_dim)


class CLIPBackbone(Backbone):
    """
    CLIP ViT-B/32 visual encoder (openai/clip via the `clip` or `open_clip` package,
    imported lazily). Kept as a documented future option per spec section 13
    (opens the door to text-based search later), not required for the initial ablations.
    """

    def __init__(self, variant: str = "clip_vit_b32", freeze_except_last_n: int = 2):
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "CLIPBackbone requires the 'open_clip_torch' package "
                "(pip install open_clip_torch --break-system-packages)."
            ) from e

        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.visual = model.visual
        self._embedding_dim = self.visual.output_dim

        blocks = list(self.visual.transformer.resblocks)
        self._freeze_all_except_last_n(blocks, freeze_except_last_n)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat, b, v = self._reshape_for_2d_backbone(images)
        features = self.visual(flat)
        return features.reshape(b, v, self._embedding_dim)


# ---------------------------------------------------------------------------
# Registry: maps model.yaml's `backbone_name` string -> (constructor, variant kwarg)
# To add a new backbone: implement a Backbone subclass above, then add one line here.
# ---------------------------------------------------------------------------

BACKBONE_REGISTRY = {
    "dinov2_vits14": (DINOv2Backbone, "dinov2_vits14"),
    "dinov2_vitb14": (DINOv2Backbone, "dinov2_vitb14"),
    "convnext_tiny": (ConvNeXtBackbone, "convnext_tiny"),
    "convnext_small": (ConvNeXtBackbone, "convnext_small"),
    "efficientnet_b0": (EfficientNetBackbone, "efficientnet_b0"),
    "clip_vit_b32": (CLIPBackbone, "clip_vit_b32"),
}


def get_backbone(backbone_name: str, freeze_except_last_n: int = 2) -> Backbone:
    """Factory function used by models/factory.py to build the configured backbone."""
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone_name '{backbone_name}'. "
            f"Available options: {list(BACKBONE_REGISTRY.keys())}"
        )
    cls, variant = BACKBONE_REGISTRY[backbone_name]
    return cls(variant=variant, freeze_except_last_n=freeze_except_last_n)
