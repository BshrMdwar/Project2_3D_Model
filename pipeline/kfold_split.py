"""
kfold_split.py

Stratified K-Fold splitting over (super_category, object_category) combined
labels. Per the original spec (section 8 -- followed verbatim per explicit
user instruction to default to the spec whenever no explicit override was
given): a stratification class with FEWER samples than K cannot be split by
sklearn's StratifiedKFold (it would raise). Rather than crash, such
under-populated combos are included in every fold (with a clearly logged
warning) so training never hard-fails on a rare combo -- this only affects
evaluation validity for that specific rare combo, not overall pipeline
robustness.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def _combined_stratify_labels(super_categories: list[str], object_categories: list[str]) -> list[str]:
    """Combine (super_category, object_category) into one string label per sample for stratification."""
    return [f"{sc}::{oc}" for sc, oc in zip(super_categories, object_categories)]


def make_kfold_splits(
    sample_ids: list[str],
    super_categories: list[str],
    object_categories: list[str],
    k_folds: int,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    """
    Args:
        sample_ids: list of sample ids, same order/length as super_categories/object_categories.
        super_categories: super_category label per sample (same order as sample_ids).
        object_categories: object_category label per sample, or "" if not applicable (e.g. Carpet).
        k_folds: number of folds.
        seed: for reproducible fold assignment.

    Returns:
        list of (train_ids, val_ids) tuples, one per fold.

    Behavior for under-populated stratification classes (< k_folds samples):
        Per the original spec, these samples are included in EVERY fold's
        training set, and also included in every fold's validation set (since
        there's no valid way to hold them out for exactly one fold while
        keeping the rest balanced). This is logged clearly as a warning so
        it's visible in training logs, not silently absorbed.
    """
    assert len(sample_ids) == len(super_categories) == len(object_categories)

    combined_labels = _combined_stratify_labels(super_categories, object_categories)
    label_counts = Counter(combined_labels)

    under_populated_labels = {label for label, count in label_counts.items() if count < k_folds}
    under_populated_indices = [i for i, lbl in enumerate(combined_labels) if lbl in under_populated_labels]
    splittable_indices = [i for i, lbl in enumerate(combined_labels) if lbl not in under_populated_labels]

    if under_populated_indices:
        affected_labels = sorted(under_populated_labels)
        logger.warning(
            f"{len(under_populated_indices)} sample(s) belong to (super_category, "
            f"object_category) combos with fewer than k_folds={k_folds} samples: "
            f"{affected_labels}. Per spec, these are included in EVERY fold's train "
            f"AND validation sets (cannot be meaningfully held out for a single fold). "
            f"Evaluation metrics on these specific combos should be interpreted with "
            f"this caveat in mind."
        )

    folds: list[tuple[list[str], list[str]]] = []

    if not splittable_indices:
        # Degenerate case: every combo is under-populated. Every fold is identical
        # (all samples in train and val) -- log clearly and proceed rather than crash.
        logger.warning(
            "ALL samples belong to under-populated combos -- every fold will be "
            "identical (all data in both train and val). K-fold cross-validation "
            "is not meaningful in this state; consider this a placeholder until "
            "more data is collected."
        )
        all_ids = list(sample_ids)
        return [(all_ids, all_ids) for _ in range(k_folds)]

    splittable_ids = [sample_ids[i] for i in splittable_indices]
    splittable_labels = [combined_labels[i] for i in splittable_indices]
    under_populated_ids = [sample_ids[i] for i in under_populated_indices]

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    X_placeholder = np.zeros(len(splittable_ids))

    for train_idx, val_idx in skf.split(X_placeholder, splittable_labels):
        train_ids = [splittable_ids[i] for i in train_idx] + under_populated_ids
        val_ids = [splittable_ids[i] for i in val_idx] + under_populated_ids
        folds.append((train_ids, val_ids))

    logger.info(f"Built {len(folds)} stratified folds over {len(sample_ids)} total samples.")
    return folds
