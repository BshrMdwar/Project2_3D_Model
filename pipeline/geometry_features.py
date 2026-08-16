"""
geometry_features.py

Defensively extracts a fixed-length numeric feature vector from a geometry JSON
record, using a config-driven list of dotted paths (e.g. "dimensions.aspect_hw").

The geometry JSON schema is explicitly documented as subject to change (fields
may be added/removed/renamed by the Blender-side extraction worker). This module
NEVER assumes a field exists -- every access goes through safe dotted-path lookup
with a default of 0.0, and missing fields are logged + returned so callers (e.g.
dataset.py) can record them as per-sample metadata rather than silently losing
information.

Variable-length fields (e.g. `materials_and_textures.materials_breakdown`, a list
of dicts) are NOT supported by this module. Per explicit user decision, this data
is not currently used as a training signal (future uploads are not guaranteed to
populate it consistently), so no aggregation logic is implemented for it. If such
a path appears in the configured feature list, `build_feature_list_or_raise` fails
fast at config-load time with a clear error, rather than failing deep into a
training run.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Dotted paths that are known to be variable-length / non-scalar in the schema.
# Referencing any of these (or any path under them) in a feature list is a
# configuration error, caught eagerly.
_KNOWN_UNSUPPORTED_PREFIXES = (
    "materials_and_textures.materials_breakdown",
    "render_links.views",
)


class UnsupportedGeometryFeatureError(ValueError):
    """Raised when a configured feature path points at a variable-length / non-scalar field."""


def _get_dotted(obj: dict, dotted_path: str) -> Any:
    """
    Safely resolve a dotted path like "dimensions.aspect_hw" against a nested dict.
    Returns None if any segment along the path is missing or not a dict.
    Never raises.
    """
    current: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def validate_feature_list(feature_list: list[str]) -> None:
    """
    Fail-fast validation of a configured feature list, called at config-load time
    (before any training starts). Raises UnsupportedGeometryFeatureError if any
    entry points at a known variable-length field.
    """
    for path in feature_list:
        for bad_prefix in _KNOWN_UNSUPPORTED_PREFIXES:
            if path == bad_prefix or path.startswith(bad_prefix + "."):
                raise UnsupportedGeometryFeatureError(
                    f"Feature path '{path}' refers to a variable-length / non-scalar "
                    f"geometry field ('{bad_prefix}'). This module only supports fixed "
                    f"scalar numeric features. Remove it from geometry_feature_list in "
                    f"the config, or aggregate it into a fixed-length derived field "
                    f"upstream (e.g. in the Blender worker) before referencing it here."
                )


def extract_geometry_vector(
    geometry_json: dict,
    feature_list: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Extract a fixed-length numeric feature vector from a geometry JSON record.

    Args:
        geometry_json: the parsed {id}.json content from dataset/geometry/.
        feature_list: ordered list of dotted paths defining which fields to use
            and in what order. This list is the single source of truth for
            "what geometry features exist" -- defined once in model.yaml config,
            never hardcoded elsewhere.

    Returns:
        vector: np.ndarray of shape (len(feature_list),), dtype float32.
            Missing or non-numeric fields are filled with 0.0.
        missing_fields: list of dotted paths that were missing or non-numeric
            for this specific sample. Empty list if everything was present.
            Callers should attach this to per-sample metadata (not swallow it).
    """
    values = np.zeros(len(feature_list), dtype=np.float32)
    missing_fields: list[str] = []

    for i, path in enumerate(feature_list):
        raw = _get_dotted(geometry_json, path)

        if raw is None:
            missing_fields.append(path)
            continue

        # Symmetry axis and similar categorical strings sometimes live alongside
        # numeric fields in this schema (e.g. shape_descriptors.symmetry_axis).
        # We defensively coerce; anything that can't become a float is treated
        # as missing rather than crashing the whole extraction.
        try:
            values[i] = float(raw)
        except (TypeError, ValueError):
            missing_fields.append(path)
            logger.debug(
                f"Geometry field '{path}' present but not numeric (value={raw!r}); "
                f"treated as missing (0.0)."
            )

    if missing_fields:
        logger.warning(
            f"Geometry extraction: {len(missing_fields)}/{len(feature_list)} "
            f"configured fields missing/non-numeric: {missing_fields}"
        )

    return values, missing_fields


def get_feature_names(feature_list: list[str]) -> list[str]:
    """
    Return human-readable feature names (identical to the dotted paths) in the
    exact order they appear in the output vector. Useful for logging, scaler
    persistence, and visualization/plot_embeddings.py axis labels.
    """
    return list(feature_list)
