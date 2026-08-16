"""
imbalance_handling.py

=====================================================================
IMBALANCE HANDLING LAYER -- any part of this file can be independently
added, removed, or modified without affecting the rest of the pipeline.
=====================================================================

The dataset is small (~1,130 samples, growing) and severely, deliberately
UNBALANCED (per user decision: no undersampling, no dropped samples --
minority classes, including near-empty ones, are used as-is). This file
collects every imbalance-mitigation technique in one place, each behind its
own config flag, so techniques can be toggled independently to run ablations
(e.g. "what if we turn off focal loss but keep class weighting?").

Techniques implemented:
    1. Class-weighted loss       -- per-class weight = inverse frequency
    2. Focal loss                -- alternative/additional to weighted CE
    3. Balanced batch sampling   -- WeightedRandomSampler, independent of loss weighting
    4. Minimum-sample confidence threshold -- flags rare combos as
       "structurally low-confidence" in output metadata (no training-time
       rejection, just an inference-time signal -- see dataset.py's
       low_confidence_combo field, computed via this module's
       `compute_low_confidence_combos`).

Practical guidance on when to use which (documented in code, as required):
    - Class-weighted loss: use when class imbalance is moderate-to-severe but
      you still want the model to see every sample equally often per epoch
      (no resampling); the LOSS magnitude is what's rebalanced, not exposure.
    - Focal loss: use when the imbalance is compounded by "easy" majority-class
      samples dominating gradient signal even after weighting: focal loss
      down-weights already-confident (easy) predictions regardless of class,
      which pairs well with class weighting for severe imbalance rather than
      replacing it outright. Can be used INSTEAD of weighted CE (uncommon) or
      ADDED ON TOP of weighted CE (typical) via `use_class_weighted_loss` AND
      `use_focal_loss` both True.
    - Balanced batch sampling: use when you additionally want the model to
      physically SEE minority-class samples more often per epoch (changes
      exposure, not just gradient magnitude). This is a different lever from
      loss weighting -- can be used alone, together, or not at all.
    - Min-sample confidence threshold: NOT a training-time technique at all --
      purely a bookkeeping/output-metadata signal for downstream consumers
      (e.g. Django backend can display "low confidence: rare category" to users).
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Class-weighted loss
# ---------------------------------------------------------------------------

def compute_class_weights(labels: list[int], num_classes: int) -> "torch.Tensor":
    """
    Inverse-frequency class weights: weight[c] = total_samples / (num_classes * count[c]).
    Classes with zero samples get weight 0.0 (they can't appear as targets anyway,
    per the dynamic active-taxonomy exclusion in dataset.py -- this is just a safety
    fallback, not expected to trigger in normal use).
    """
    counts = Counter(labels)
    total = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        count = counts.get(c, 0)
        weights[c] = (total / (num_classes * count)) if count > 0 else 0.0

    if _TORCH_AVAILABLE:
        return torch.from_numpy(weights)
    return weights


# ---------------------------------------------------------------------------
# 2. Focal loss
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class FocalLoss(torch.nn.Module):
        """
        Multi-class focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Can be combined with class weights (alpha_t = class_weights[target]) or
        used with uniform alpha=1.0 if class weighting is handled elsewhere.
        """

        def __init__(self, gamma: float = 2.0, class_weights: "torch.Tensor | None" = None):
            super().__init__()
            self.gamma = gamma
            self.register_buffer("class_weights", class_weights, persistent=False)

        def forward(self, logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            focal_term = (1 - target_probs) ** self.gamma
            loss = -focal_term * target_log_probs

            if self.class_weights is not None:
                alpha_t = self.class_weights[targets]
                loss = loss * alpha_t

            return loss.mean()


# ---------------------------------------------------------------------------
# 3. Balanced batch sampling
# ---------------------------------------------------------------------------

def compute_sample_weights_for_balanced_sampling(labels: list[int]) -> np.ndarray:
    """
    Per-SAMPLE weight (not per-class) for use with torch.utils.data.WeightedRandomSampler,
    such that each class is sampled roughly equally often per epoch regardless of
    its actual frequency in the dataset. weight[i] = 1 / count[label[i]].
    """
    counts = Counter(labels)
    weights = np.array([1.0 / counts[label] for label in labels], dtype=np.float64)
    return weights


def build_weighted_random_sampler(labels: list[int]):
    """
    Returns a torch.utils.data.WeightedRandomSampler ready to pass to a DataLoader's
    `sampler=` argument (mutually exclusive with `shuffle=True`).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required to build a WeightedRandomSampler.")
    from torch.utils.data import WeightedRandomSampler
    weights = compute_sample_weights_for_balanced_sampling(labels)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(labels),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# 4. Minimum-sample confidence threshold (bookkeeping only, no training effect)
# ---------------------------------------------------------------------------

def compute_low_confidence_combos(
    super_categories: list[str],
    object_categories: list[str],
    min_samples: int,
) -> set[tuple[str, str]]:
    """
    Identify (super_category, object_category) combinations with fewer than
    `min_samples` training examples. Used purely as an output-metadata signal
    (dataset.py attaches this as `low_confidence_combo` per sample, and
    inference.py surfaces it in the `warnings` list) -- this does NOT exclude,
    reweight, or otherwise change how these samples are used in training.
    """
    combo_counts = Counter(zip(super_categories, object_categories))
    flagged = {combo for combo, count in combo_counts.items() if count < min_samples}
    if flagged:
        logger.info(
            f"{len(flagged)} (super_category, object_category) combos flagged as "
            f"structurally low-confidence (< {min_samples} samples): {sorted(flagged)}"
        )
    return flagged
