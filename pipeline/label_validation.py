"""
label_validation.py

Validates a metadata (label) JSON record against taxonomy.py, the single source
of truth for valid category values. This is considered MORE important than
geometry_validation.py: label errors (especially VLM-sourced auto-labels) affect
training quality far more directly than a single wrong geometric statistic.

Like geometry_validation.py, this never raises for a single bad sample -- it
returns a structured result so the caller (dataset.py, explore_dataset.py,
the Dataset Health Report) can log a warning and exclude the sample rather than
crash the whole pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import taxonomy as tx

logger = logging.getLogger(__name__)


@dataclass
class LabelValidationResult:
    model_id: str
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_REQUIRED_TOP_LEVEL_FIELDS = ["super_category", "style_class"]


def validate_label(metadata_json: dict, model_id: str | None = None) -> LabelValidationResult:
    """
    Validate a single metadata/{id}.json record.

    Checks:
        - required top-level fields are present and non-empty
        - super_category exists in taxonomy.py
        - object_category (if present) is a legal sub-category of super_category
          per taxonomy.py's hierarchy (SUPER_CATEGORIES dict) -- including the
          special case where super_category has NO object_category at all
          (currently only "Carpet"), where object_category must be absent/null
        - style_class values (multi-label list) all exist in STYLE_CLASSES
        - materials.primary exists in MATERIALS
        - materials.secondary values (multi-label list) all exist in MATERIALS

    Returns:
        LabelValidationResult. is_valid=False for any hard taxonomy violation
        or missing required field. Optional/soft issues (e.g. missing
        confidence_scores on a Human-annotated record, which is expected per
        the schema) are not flagged at all -- they're normal, not warnings.
    """
    if not isinstance(metadata_json, dict):
        mid = model_id or "<unknown>"
        result = LabelValidationResult(model_id=mid, is_valid=False)
        result.errors.append("metadata_json is not a dict (corrupt or malformed file)")
        return result

    mid = model_id or metadata_json.get("id", "<unknown>")
    result = LabelValidationResult(model_id=mid, is_valid=True)

    # 1. Required top-level fields present and non-empty
    for field_name in _REQUIRED_TOP_LEVEL_FIELDS:
        value = metadata_json.get(field_name)
        if value is None or value == "" or value == []:
            result.errors.append(f"Required field '{field_name}' is missing or empty.")
            result.is_valid = False

    super_category = metadata_json.get("super_category")

    # 2. super_category must exist in taxonomy
    if super_category is not None and not tx.is_valid_super_category(super_category):
        result.errors.append(
            f"super_category '{super_category}' is not a recognized category in taxonomy.py."
        )
        result.is_valid = False

    # 3. object_category hierarchy check
    object_category = metadata_json.get("object_category")
    if super_category is not None and tx.is_valid_super_category(super_category):
        expects_object_category = tx.has_object_category(super_category)

        if expects_object_category:
            if not object_category:
                result.errors.append(
                    f"super_category '{super_category}' requires an object_category, "
                    f"but none was provided."
                )
                result.is_valid = False
            elif not tx.is_valid_object_category(super_category, object_category):
                result.errors.append(
                    f"object_category '{object_category}' is not valid under "
                    f"super_category '{super_category}'. "
                    f"Valid options: {tx.get_object_categories(super_category)}"
                )
                result.is_valid = False
        else:
            # e.g. Carpet -- object_category should not be meaningfully present
            if object_category:
                result.warnings.append(
                    f"super_category '{super_category}' has no object_category "
                    f"classification in this taxonomy, but object_category="
                    f"'{object_category}' was provided in the label. It will be "
                    f"ignored during training, not used as an error."
                )

    # 4. style_class multi-label validity
    style_classes = metadata_json.get("style_class", [])
    if isinstance(style_classes, list):
        for style in style_classes:
            if not tx.is_valid_style(style):
                result.errors.append(f"style_class value '{style}' not in STYLE_CLASSES.")
                result.is_valid = False
    elif style_classes:
        result.errors.append("style_class must be a list (multi-label field).")
        result.is_valid = False

    # 5. materials.primary / materials.secondary
    materials = metadata_json.get("materials", {})
    if isinstance(materials, dict):
        primary = materials.get("primary")
        if primary and not tx.is_valid_material(primary):
            result.errors.append(f"materials.primary '{primary}' not in MATERIALS.")
            result.is_valid = False

        secondary = materials.get("secondary", [])
        if isinstance(secondary, list):
            for mat in secondary:
                if not tx.is_valid_material(mat):
                    result.errors.append(f"materials.secondary value '{mat}' not in MATERIALS.")
                    result.is_valid = False
        elif secondary:
            result.errors.append("materials.secondary must be a list (multi-label field).")
            result.is_valid = False
    elif materials:
        result.errors.append("'materials' field must be a dict.")
        result.is_valid = False

    # 6. confidence_scores is optional by schema design (VLM-only) -- no check needed,
    #    consumers must already treat it as Optional[dict].

    if not result.is_valid:
        logger.warning(f"[{mid}] label validation FAILED: {result.errors}")
    elif result.warnings:
        logger.info(f"[{mid}] label validation passed with {len(result.warnings)} warning(s).")

    return result
