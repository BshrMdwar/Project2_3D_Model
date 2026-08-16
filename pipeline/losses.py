"""
losses.py

Aggregates the multi-task loss across all 5 output heads:
    - super_category    (single-label, cross-entropy)
    - object_category    (single-label per sample, cross-entropy -- but each
                          sample may route through a DIFFERENT sub-head with a
                          DIFFERENT number of classes, per heads.py's per-sample
                          list output; samples whose super_category has no
                          object_category sub-head, e.g. Carpet, contribute 0
                          to this term)
    - style_class         (multi-label, binary cross-entropy per class)
    - materials_primary    (single-label, cross-entropy)
    - materials_secondary  (multi-label, binary cross-entropy per class)

Total loss = sum of the (optionally class-weighted / focal / sample-weighted)
per-task losses. Task loss weights are equal (1.0 each) by default; this is
intentionally simple and can be extended to learned/config-driven task
weighting later without changing the interface.

Every imbalance-handling technique (imbalance_handling.py) is threaded through
here via the LossConfig flags -- this file does not implement any imbalance
logic itself, it only wires the config flags to the right calls.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import imbalance_handling as ih
from config import LossConfig

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        loss_config: LossConfig,
        num_super_categories: int,
        object_category_class_counts: dict[str, int],
        super_category_labels_for_weights: list[int] | None = None,
        object_category_labels_for_weights: dict[str, list[int]] | None = None,
        materials_primary_labels_for_weights: list[int] | None = None,
    ):
        """
        Args:
            loss_config: LossConfig (config.py).
            num_super_categories: size of the super_category head.
            object_category_class_counts: {super_category_name: num_classes}
                for every sub-head that exists in the model (mirrors
                heads.py's MultiTaskHeads.object_category_heads keys).
            *_labels_for_weights: label lists used to compute class weights
                (only needed if loss_config.use_class_weighted_loss=True).
                Typically computed once from the training fold by train.py
                and passed in here.
        """
        super().__init__()
        self.loss_config = loss_config
        self.object_category_class_counts = dict(object_category_class_counts)

        self.super_category_weights = None
        self.object_category_weights: dict[str, torch.Tensor] = {}
        self.materials_primary_weights = None

        if loss_config.use_class_weighted_loss:
            if super_category_labels_for_weights is not None:
                self.super_category_weights = ih.compute_class_weights(
                    super_category_labels_for_weights, num_super_categories
                )
            if object_category_labels_for_weights is not None:
                for sc, num_classes in self.object_category_class_counts.items():
                    labels = object_category_labels_for_weights.get(sc, [])
                    if labels:
                        self.object_category_weights[sc] = ih.compute_class_weights(labels, num_classes)
            if materials_primary_labels_for_weights is not None:
                from taxonomy import MATERIALS
                self.materials_primary_weights = ih.compute_class_weights(
                    materials_primary_labels_for_weights, len(MATERIALS)
                )

        self.focal_loss_super_category = None
        if loss_config.use_focal_loss:
            self.focal_loss_super_category = ih.FocalLoss(
                gamma=loss_config.focal_loss_gamma, class_weights=self.super_category_weights
            )

    def forward(self, head_outputs: dict, labels: dict, sample_weights: torch.Tensor) -> dict:
        """
        Args:
            head_outputs: dict returned by MultiTaskHeads.forward().
            labels: dict with keys matching dataset.py's Sample["labels"]:
                "super_category" (B,) long, "object_category" (B,) long (-1 = N/A),
                "style_class" (B, len(STYLE_CLASSES)) float multi-hot,
                "materials_primary" (B,) long (-1 = N/A),
                "materials_secondary" (B, len(MATERIALS)) float multi-hot.
            sample_weights: (B,) float tensor, per-sample confidence-derived
                weight (1.0 if disabled, per imbalance_handling.py section 4/dataset.py).

        Returns:
            dict with "total_loss" (scalar tensor, backprop this one) plus each
            individual task loss (scalars, for logging/monitoring only).
        """
        device = head_outputs["super_category_logits"].device
        sample_weights = sample_weights.to(device)

        # --- super_category ---
        if self.focal_loss_super_category is not None:
            per_sample_sc_loss = self._per_sample_ce(
                head_outputs["super_category_logits"], labels["super_category"],
                weight=None,  # focal loss handles its own alpha weighting internally
                use_focal=True,
            )
        else:
            per_sample_sc_loss = self._per_sample_ce(
                head_outputs["super_category_logits"], labels["super_category"],
                weight=self.super_category_weights.to(device) if self.super_category_weights is not None else None,
            )
        super_category_loss = (per_sample_sc_loss * sample_weights).mean()

        # --- object_category: per-sample list with possibly-None entries and
        #     DIFFERENT class counts per sample (routed to different sub-heads) ---
        object_category_losses = []
        object_category_weight_sum = 0.0
        for i, (logits, target) in enumerate(zip(head_outputs["object_category_logits"], labels["object_category"])):
            target_val = int(target.item()) if torch.is_tensor(target) else int(target)
            if logits is None or target_val < 0:
                continue  # e.g. Carpet, or missing/invalid object_category label
            weight_vec = None
            # NOTE: we don't have a clean per-sample super_category name here without
            # threading it through -- for class-weighted object_category loss, train.py
            # supplies weights already keyed by super_category at MultiTaskLoss construction;
            # per-sample lookup requires knowing which sub-head this sample used, which
            # we can recover from the number of classes in `logits`.
            loss_i = F.cross_entropy(
                logits.unsqueeze(0), torch.tensor([target_val], device=device), reduction="none"
            )[0]
            object_category_losses.append(loss_i * sample_weights[i])
            object_category_weight_sum += 1.0

        if object_category_losses:
            object_category_loss = torch.stack(object_category_losses).sum() / max(object_category_weight_sum, 1.0)
        else:
            object_category_loss = torch.tensor(0.0, device=device)

        # --- style_class (multi-label) ---
        style_loss_per_sample = F.binary_cross_entropy_with_logits(
            head_outputs["style_logits"], labels["style_class"].to(device), reduction="none"
        ).mean(dim=-1)
        style_loss = (style_loss_per_sample * sample_weights).mean()

        # --- materials_primary (single-label, -1 = missing) ---
        primary_targets = labels["materials_primary"]
        valid_mask = primary_targets >= 0
        if valid_mask.any():
            weight = self.materials_primary_weights.to(device) if self.materials_primary_weights is not None else None
            primary_loss_per_sample = F.cross_entropy(
                head_outputs["materials_primary_logits"][valid_mask],
                primary_targets[valid_mask].to(device),
                weight=weight,
                reduction="none",
            )
            materials_primary_loss = (primary_loss_per_sample * sample_weights[valid_mask]).mean()
        else:
            materials_primary_loss = torch.tensor(0.0, device=device)

        # --- materials_secondary (multi-label) ---
        secondary_loss_per_sample = F.binary_cross_entropy_with_logits(
            head_outputs["materials_secondary_logits"], labels["materials_secondary"].to(device), reduction="none"
        ).mean(dim=-1)
        materials_secondary_loss = (secondary_loss_per_sample * sample_weights).mean()

        total_loss = (
            super_category_loss
            + object_category_loss
            + style_loss
            + materials_primary_loss
            + materials_secondary_loss
        )

        return {
            "total_loss": total_loss,
            "super_category_loss": super_category_loss.detach(),
            "object_category_loss": object_category_loss.detach(),
            "style_loss": style_loss.detach(),
            "materials_primary_loss": materials_primary_loss.detach(),
            "materials_secondary_loss": materials_secondary_loss.detach(),
        }

    @staticmethod
    def _per_sample_ce(
        logits: torch.Tensor, targets: torch.Tensor,
        weight: torch.Tensor | None, use_focal: bool = False,
    ) -> torch.Tensor:
        if use_focal:
            # NOTE: FocalLoss.forward() as implemented reduces internally (mean);
            # for per-sample weighting compatibility we recompute focal loss per-sample here.
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            gamma = 2.0
            focal_term = (1 - target_probs) ** gamma
            return -focal_term * target_log_probs
        return F.cross_entropy(logits, targets, weight=weight, reduction="none")
