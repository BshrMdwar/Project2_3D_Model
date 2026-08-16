"""
metrics.py

Computes classification metrics for each of the 5 output heads, plus
per-super_category breakdowns for object_category (since each super_category
routes to a different sub-head with a different class count -- there is no
single confusion matrix across all of them).

All metrics use sklearn under the hood for correctness/standard definitions.
Handles the "zero samples for this class in this batch/fold" case gracefully
(sklearn's zero_division=0 default avoids warnings/crashes on rare classes,
which matters a lot for this severely imbalanced dataset).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


def compute_single_label_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict:
    """Standard accuracy/precision/recall/F1 (macro) for a single-label classification head."""
    if len(y_true) == 0:
        return {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0, "n_samples": 0}

    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "n_samples": len(y_true),
    }


def compute_confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    labels = list(range(num_classes))
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_multi_label_metrics(y_true: np.ndarray, y_pred_binary: np.ndarray) -> dict:
    """
    For multi-label heads (style_class, materials_secondary): micro/macro F1
    over the binary multi-hot vectors. y_true/y_pred_binary shape: (N, num_classes).
    """
    if y_true.shape[0] == 0:
        return {"f1_micro": 0.0, "f1_macro": 0.0, "n_samples": 0}

    return {
        "f1_micro": float(f1_score(y_true, y_pred_binary, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred_binary, average="macro", zero_division=0)),
        "n_samples": int(y_true.shape[0]),
    }


class ObjectCategoryMetricsAccumulator:
    """
    Accumulates predictions per super_category (since each has its own sub-head
    with its own class vocabulary) across a full evaluation pass, then reports
    per-super_category metrics plus a single macro-averaged summary across all
    super_categories with at least one sample.
    """

    def __init__(self):
        self._per_super: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})

    def add_batch(self, super_category_names: list[str], y_true: list[int], y_pred: list[int]) -> None:
        """
        Args:
            super_category_names: (B,) the super_category name each sample's
                object_category prediction was routed through.
            y_true / y_pred: (B,) object_category class indices, -1 meaning
                "not applicable" (skipped, e.g. Carpet or missing label).
        """
        for sc, t, p in zip(super_category_names, y_true, y_pred):
            if t < 0:
                continue
            self._per_super[sc]["y_true"].append(t)
            self._per_super[sc]["y_pred"].append(p)

    def report(self, num_classes_per_super: dict[str, int]) -> dict:
        """
        Args:
            num_classes_per_super: {super_category_name: num_classes} for
                every sub-head that exists (from ActiveTaxonomy).
        Returns:
            {
                "per_super_category": {sc_name: {accuracy, precision_macro, ...}, ...},
                "macro_avg_f1": float  (mean of per-super_category f1_macro,
                                        across super_categories that had >=1 sample)
            }
        """
        per_super_report = {}
        f1_values = []
        for sc, num_classes in num_classes_per_super.items():
            y_true = self._per_super[sc]["y_true"]
            y_pred = self._per_super[sc]["y_pred"]
            metrics = compute_single_label_metrics(y_true, y_pred, num_classes)
            per_super_report[sc] = metrics
            if metrics["n_samples"] > 0:
                f1_values.append(metrics["f1_macro"])

        macro_avg_f1 = float(np.mean(f1_values)) if f1_values else 0.0
        return {"per_super_category": per_super_report, "macro_avg_f1": macro_avg_f1}
