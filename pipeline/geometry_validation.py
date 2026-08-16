"""
geometry_validation.py

Validates a geometry JSON record BEFORE it is used to extract training features.
Checks for logically-invalid values (non-positive dimensions), NaN/Infinity, and
flags statistical outliers relative to a reference distribution (optional, used
by explore_dataset.py / the Dataset Health Report -- not required for a single
isolated validation call).

This module never raises on a single bad sample -- it returns a structured
result so the caller (dataset.py, explore_dataset.py) can decide whether to log
a warning and exclude the sample, per the project's defensive-error-handling
convention (bad samples are skipped, not fatal).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GeometryValidationResult:
    model_id: str
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Fields that must be strictly positive if present (logically impossible otherwise).
_POSITIVE_REQUIRED_PATHS = [
    "dimensions.width",
    "dimensions.depth",
    "dimensions.height",
    "mesh_density.vertices",
    "mesh_density.faces",
    "geometry.surface_area",
]


def _get_dotted(obj: dict, dotted_path: str):
    current = obj
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _iter_all_numeric_leaves(obj: dict, prefix: str = ""):
    """Yield (dotted_path, value) for every numeric leaf in a nested dict."""
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _iter_all_numeric_leaves(value, path)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield path, value


def validate_geometry(
    geometry_json: dict,
    model_id: str | None = None,
    reference_stats: dict[str, tuple[float, float]] | None = None,
    outlier_z_threshold: float = 5.0,
) -> GeometryValidationResult:
    """
    Validate a single geometry JSON record.

    Args:
        geometry_json: parsed geometry/{id}.json content.
        model_id: identifier for logging/reporting; falls back to
            geometry_json.get("model_id", "<unknown>") if not given.
        reference_stats: optional dict of {dotted_path: (mean, std)} computed
            across the dataset (typically by explore_dataset.py). If provided,
            numeric fields more than `outlier_z_threshold` standard deviations
            from the mean are flagged as warnings (not errors -- outliers are
            not necessarily wrong, just worth a human glance).
        outlier_z_threshold: z-score threshold for the outlier warning above.

    Returns:
        GeometryValidationResult with is_valid=False only for hard errors
        (non-positive required dims, NaN/Infinity anywhere). Statistical
        outliers are warnings only and do not flip is_valid to False.
    """
    mid = model_id or geometry_json.get("model_id", "<unknown>")
    result = GeometryValidationResult(model_id=mid, is_valid=True)

    if not isinstance(geometry_json, dict):
        result.is_valid = False
        result.errors.append("geometry_json is not a dict (corrupt or malformed file)")
        return result

    # 1. Required-positive checks
    for path in _POSITIVE_REQUIRED_PATHS:
        value = _get_dotted(geometry_json, path)
        if value is None:
            result.warnings.append(f"Expected field '{path}' is missing.")
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            result.errors.append(f"Field '{path}' is not numeric (value={value!r}).")
            result.is_valid = False
            continue
        if fval <= 0:
            result.errors.append(f"Field '{path}' must be > 0, got {fval}.")
            result.is_valid = False

    # 2. NaN / Infinity check across every numeric leaf in the document
    for path, value in _iter_all_numeric_leaves(geometry_json):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            result.errors.append(f"Field '{path}' is NaN/Infinity (value={value}).")
            result.is_valid = False

    # 3. Optional statistical outlier detection (warning-level only)
    if reference_stats:
        for path, value in _iter_all_numeric_leaves(geometry_json):
            if path not in reference_stats:
                continue
            mean, std = reference_stats[path]
            if std <= 1e-9:
                continue
            z = abs((value - mean) / std)
            if z > outlier_z_threshold:
                result.warnings.append(
                    f"Field '{path}' is a statistical outlier (z={z:.1f}, value={value}, "
                    f"dataset mean={mean:.4f}, std={std:.4f}). May indicate a render or "
                    f"source-model error -- worth a manual look, not necessarily wrong."
                )

    if not result.is_valid:
        logger.warning(f"[{mid}] geometry validation FAILED: {result.errors}")
    elif result.warnings:
        logger.info(f"[{mid}] geometry validation passed with {len(result.warnings)} warning(s).")

    return result


def compute_reference_stats(all_geometry_jsons: list[dict]) -> dict[str, tuple[float, float]]:
    """
    Compute (mean, std) per dotted numeric path across a list of geometry JSON
    records. Used to build `reference_stats` for outlier detection above.
    Intended to be called once by explore_dataset.py / the Dataset Health
    Report over the whole (or training-fold) dataset, not per-sample.
    """
    import numpy as np

    collected: dict[str, list[float]] = {}
    for gjson in all_geometry_jsons:
        for path, value in _iter_all_numeric_leaves(gjson):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                continue  # don't let corrupt samples poison the reference stats
            collected.setdefault(path, []).append(float(value))

    stats: dict[str, tuple[float, float]] = {}
    for path, values in collected.items():
        arr = np.array(values, dtype=np.float64)
        stats[path] = (float(arr.mean()), float(arr.std()))
    return stats
