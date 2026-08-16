"""
models/heads.py

Multi-task output heads built on top of the fused embedding (pooled image
embedding, optionally concatenated with geometry features via fusion.py).

Per explicit user decision, object_category uses OPTION (B) from spec 7.2:
a SEPARATE linear head per super_category, not one shared head with masking.
Concretely: MultiTaskHeads.object_category_heads is an nn.ModuleDict keyed by
super_category name, each mapping input_dim -> num_classes_for_that_super_category.

Head sizes are DYNAMIC (per spec section 6 + explicit user confirmation):
built from `active_taxonomy` (dataset.ActiveTaxonomy), which only includes
super_categories/object_categories with >= 1 sample in the CURRENT dataset.
A super_category with zero samples gets no head at all. A super_category that
exists but has zero valid object_category samples (e.g. very early in data
collection) gets a super_category head but NO object_category sub-head --
handled gracefully (skipped) rather than crashing on a 0-sized nn.Linear.

Carpet (or any super_category with has_object_category()==False in taxonomy.py)
never gets an object_category_heads entry at all -- this is a structural
property of the taxonomy, not a data-availability question.

Two other heads are always present as long as the dataset has any samples:
    - style_class:          nn.Linear(input_dim, len(STYLE_CLASSES))       multi-label
    - materials_primary:     nn.Linear(input_dim, len(MATERIALS))           single-label
    - materials_secondary:   nn.Linear(input_dim, len(MATERIALS))           multi-label
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

import taxonomy as tx

logger = logging.getLogger(__name__)


class MultiTaskHeads(nn.Module):
    def __init__(self, input_dim: int, active_super_categories: list[str],
                 active_object_categories: dict[str, list[str]]):
        """
        Args:
            input_dim: dimensionality of the fused embedding fed into every head.
            active_super_categories: ordered list from dataset.ActiveTaxonomy
                (defines the super_category head's class order/index mapping).
            active_object_categories: dict from dataset.ActiveTaxonomy mapping
                super_category -> ordered list of its currently-active
                object_category values (defines each sub-head's class order).
                A super_category absent from this dict, or present with an
                empty list, gets NO object_category_heads entry.
        """
        super().__init__()
        self.active_super_categories = list(active_super_categories)
        self.active_object_categories = {k: list(v) for k, v in active_object_categories.items()}

        self.super_category_head = nn.Linear(input_dim, len(self.active_super_categories))

        # Option (B): one independent nn.Linear per super_category that HAS
        # object_category classification AND has >=1 active sub-category sample.
        object_category_heads = {}
        for sc in self.active_super_categories:
            if not tx.has_object_category(sc):
                continue  # e.g. Carpet -- structurally no sub-head, by design
            num_classes = len(self.active_object_categories.get(sc, []))
            if num_classes == 0:
                logger.warning(
                    f"super_category '{sc}' has samples but zero active object_category "
                    f"values -- skipping its object_category sub-head until data exists."
                )
                continue
            object_category_heads[sc] = nn.Linear(input_dim, num_classes)
        self.object_category_heads = nn.ModuleDict(object_category_heads)

        self.style_head = nn.Linear(input_dim, len(tx.STYLE_CLASSES))
        self.materials_primary_head = nn.Linear(input_dim, len(tx.MATERIALS))
        self.materials_secondary_head = nn.Linear(input_dim, len(tx.MATERIALS))

    def forward(
        self,
        fused_embedding: torch.Tensor,
        super_category_gt: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            fused_embedding: (B, input_dim)
            super_category_gt: (B,) long tensor of ground-truth super_category
                indices, used ONLY during training to select which
                object_category sub-head each sample routes through (teacher
                forcing -- per design doc section 5.3). If None (inference),
                the model's own super_category prediction (argmax) is used
                instead to select the routing.

        Returns:
            dict with:
                "super_category_logits": (B, num_super_categories)
                "object_category_logits": list of length B, each element either
                    a (num_classes_for_that_sample's_super_category,) tensor,
                    or None if that sample's super_category has no active
                    object_category sub-head (e.g. Carpet, or a super_category
                    with zero sub-category samples so far). Kept as a per-sample
                    list (not a single padded tensor) because sub-head output
                    sizes differ across super_categories -- see losses.py for
                    how this is consumed.
                "style_logits": (B, len(STYLE_CLASSES))
                "materials_primary_logits": (B, len(MATERIALS))
                "materials_secondary_logits": (B, len(MATERIALS))
        """
        super_category_logits = self.super_category_head(fused_embedding)

        if super_category_gt is not None:
            routing_indices = super_category_gt
        else:
            routing_indices = super_category_logits.argmax(dim=-1)

        object_category_logits: list[torch.Tensor | None] = []
        for i in range(fused_embedding.shape[0]):
            sc_idx = int(routing_indices[i].item())
            sc_name = self.active_super_categories[sc_idx]
            if sc_name not in self.object_category_heads:
                object_category_logits.append(None)
            else:
                head = self.object_category_heads[sc_name]
                object_category_logits.append(head(fused_embedding[i:i + 1]).squeeze(0))

        return {
            "super_category_logits": super_category_logits,
            "object_category_logits": object_category_logits,
            "style_logits": self.style_head(fused_embedding),
            "materials_primary_logits": self.materials_primary_head(fused_embedding),
            "materials_secondary_logits": self.materials_secondary_head(fused_embedding),
        }

    def predict_object_category(
        self, fused_embedding: torch.Tensor, super_category_name: str
    ) -> torch.Tensor | None:
        """
        Convenience method for inference.py: get object_category logits for a
        single known/predicted super_category, without needing the batched
        routing machinery in forward(). Returns None if this super_category
        has no object_category sub-head (Carpet, or no data yet).
        """
        if super_category_name not in self.object_category_heads:
            return None
        head = self.object_category_heads[super_category_name]
        return head(fused_embedding)
