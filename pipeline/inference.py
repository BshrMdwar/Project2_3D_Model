"""
inference.py

Loads a self-contained checkpoint (produced by train.py's train_single_fold)
and runs prediction on new renders + (optionally) geometry JSON, with NO
dependency on any external config.yaml file or the original dataset --
everything needed is embedded in the checkpoint itself (design doc section 6).

This is the function meant to be called from the Django backend:
    predict(checkpoint_path, render_image_paths, geometry_json=None) -> dict

Output includes:
    - top prediction + confidence for every head
    - the low_confidence_combo-style warning if the predicted
      (super_category, object_category) combo was flagged as low-confidence at
      training time (only if that metadata was embedded in the checkpoint --
      currently not persisted per-checkpoint, documented as a possible future
      addition, see NOTE below)
    - a `warnings` list surfacing anything unusual encountered during inference
      (e.g. fewer than expected_views images provided) -- non-fatal, best-effort.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import taxonomy as tx
from config import PipelineConfig, _dict_to_pipeline_config
from models.backbones import get_backbone
from models.fusion import get_pooling, GeometryFusion, GeometryOnlyProjection
from models.heads import MultiTaskHeads
from models.factory import CompleteModel
from geometry_features import extract_geometry_vector

logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str | Path) -> dict:
    """Load the raw self-contained checkpoint dict (weights + config + taxonomy + scaler)."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def rebuild_model_from_checkpoint(checkpoint: dict) -> tuple[CompleteModel, PipelineConfig, dict, dict]:
    """
    Reconstruct the exact model architecture used at training time from the
    checkpoint's embedded config + active taxonomy, then load its weights.

    Returns:
        (model, config, active_super_categories, active_object_categories)
    """
    config = _dict_to_pipeline_config(checkpoint["config"])
    active_super_categories = checkpoint["active_super_categories"]
    active_object_categories = checkpoint["active_object_categories"]

    backbone = get_backbone(
        config.model.backbone_name,
        freeze_except_last_n=config.model.freeze_backbone_except_last_n_layers,
    )
    embedding_dim = backbone.embedding_dim
    pooling = get_pooling(config.model.pooling_method, embedding_dim=embedding_dim)

    geometry_dim = 0
    if config.model.use_geometry_fusion:
        geometry_dim = len(config.resolved_geometry_feature_list())
    fusion = GeometryFusion(enabled=config.model.use_geometry_fusion, geometry_dim=geometry_dim)

    geometry_only_projection = None
    heads_input_dim = embedding_dim + fusion.output_dim_delta
    if config.model.freeze_visual_backbone_entirely:
        geo_hidden_dim = 128
        geometry_only_projection = GeometryOnlyProjection(geometry_dim=geometry_dim, output_dim=geo_hidden_dim)
        heads_input_dim = embedding_dim + geo_hidden_dim

    heads = MultiTaskHeads(
        input_dim=heads_input_dim,
        active_super_categories=active_super_categories,
        active_object_categories=active_object_categories,
    )

    model = CompleteModel(
        backbone=backbone, pooling=pooling, fusion=fusion, heads=heads,
        geometry_only_projection=geometry_only_projection,
        freeze_visual_backbone_entirely=config.model.freeze_visual_backbone_entirely,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config, active_super_categories, active_object_categories


def _load_and_preprocess_views(image_paths: list[str], image_size: int) -> torch.Tensor:
    views = []
    for p in image_paths:
        img = Image.open(p).convert("RGB").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        views.append(arr)
    stacked = np.stack(views, axis=0)  # (V, H, W, 3)
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).float()  # (V, 3, H, W)
    return tensor.unsqueeze(0)  # (1, V, 3, H, W)


def predict(
    checkpoint_path: str | Path,
    render_image_paths: list[str],
    geometry_json: dict | None = None,
) -> dict:
    """
    Main inference entry point.

    Args:
        checkpoint_path: path to a self_contained.pt file produced by train.py.
        render_image_paths: list of file paths to rendered view images. Ideally
            matches the expected_views_per_sample used at training time; fewer
            views will still run (best-effort) with a warning, since a Django
            backend caller may only have access to a subset of angles.
        geometry_json: optional parsed geometry JSON dict. Required only if the
            checkpoint's config has use_geometry_fusion=True.

    Returns:
        {
            "predictions": {
                "super_category": {"label": str, "confidence": float},
                "object_category": {"label": str | None, "confidence": float | None},
                "style_class": [{"label": str, "confidence": float}, ...],  # multi-label, thresholded
                "materials_primary": {"label": str, "confidence": float},
                "materials_secondary": [{"label": str, "confidence": float}, ...],
            },
            "warnings": [str, ...],
        }
    """
    warnings: list[str] = []
    checkpoint = load_checkpoint(checkpoint_path)
    model, config, active_super_categories, active_object_categories = rebuild_model_from_checkpoint(checkpoint)

    expected_views = config.dataset.expected_views_per_sample
    if len(render_image_paths) != expected_views:
        warnings.append(
            f"Expected {expected_views} render views, received {len(render_image_paths)}. "
            f"Proceeding on a best-effort basis -- prediction confidence may be reduced."
        )
    if len(render_image_paths) == 0:
        raise ValueError("At least one render image path is required for prediction.")

    images = _load_and_preprocess_views(render_image_paths, config.dataset.image_size)

    geometry_vector = None
    if config.model.use_geometry_fusion:
        if geometry_json is None:
            warnings.append(
                "This model was trained with geometry fusion enabled, but no geometry_json "
                "was provided -- using an all-zero geometry vector as a fallback. Predictions "
                "may be less accurate than with real geometry data."
            )
            feature_list = config.resolved_geometry_feature_list()
            raw_vector = np.zeros(len(feature_list), dtype=np.float32)
        else:
            feature_list = config.resolved_geometry_feature_list()
            raw_vector, missing = extract_geometry_vector(geometry_json, feature_list)
            if missing:
                warnings.append(f"Geometry fields missing at inference time: {missing}")

        scaler_bytes = checkpoint.get("scaler")
        if scaler_bytes is not None:
            scaler = pickle.loads(scaler_bytes)
            raw_vector = scaler.transform(raw_vector.reshape(1, -1))[0]

        geometry_vector = torch.from_numpy(raw_vector.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        outputs = model(images, geometry_vector=geometry_vector, super_category_gt=None)

    super_probs = torch.softmax(outputs["super_category_logits"], dim=-1)[0]
    super_idx = int(super_probs.argmax().item())
    super_label = active_super_categories[super_idx]
    super_confidence = float(super_probs[super_idx].item())

    object_prediction = {"label": None, "confidence": None}
    object_logits = outputs["object_category_logits"][0]
    if object_logits is not None:
        object_probs = torch.softmax(object_logits, dim=-1)
        object_idx = int(object_probs.argmax().item())
        object_vocab = active_object_categories.get(super_label, [])
        if object_idx < len(object_vocab):
            object_prediction = {
                "label": object_vocab[object_idx],
                "confidence": float(object_probs[object_idx].item()),
            }
    elif tx.has_object_category(super_label):
        warnings.append(
            f"Predicted super_category '{super_label}' has no trained object_category "
            f"sub-head yet (insufficient training data at time of this model's training)."
        )

    style_probs = torch.sigmoid(outputs["style_logits"])[0]
    style_predictions = [
        {"label": tx.STYLE_CLASSES[i], "confidence": float(p.item())}
        for i, p in enumerate(style_probs) if p.item() > 0.5
    ]
    if not style_predictions:
        # always surface the single highest-confidence style even under threshold,
        # so the caller never gets an empty style list with no information at all
        top_i = int(style_probs.argmax().item())
        style_predictions = [{"label": tx.STYLE_CLASSES[top_i], "confidence": float(style_probs[top_i].item())}]

    materials_primary_probs = torch.softmax(outputs["materials_primary_logits"], dim=-1)[0]
    mp_idx = int(materials_primary_probs.argmax().item())
    materials_primary_prediction = {
        "label": tx.MATERIALS[mp_idx], "confidence": float(materials_primary_probs[mp_idx].item())
    }

    materials_secondary_probs = torch.sigmoid(outputs["materials_secondary_logits"])[0]
    materials_secondary_predictions = [
        {"label": tx.MATERIALS[i], "confidence": float(p.item())}
        for i, p in enumerate(materials_secondary_probs) if p.item() > 0.5
    ]

    return {
        "predictions": {
            "super_category": {"label": super_label, "confidence": super_confidence},
            "object_category": object_prediction,
            "style_class": style_predictions,
            "materials_primary": materials_primary_prediction,
            "materials_secondary": materials_secondary_predictions,
        },
        "warnings": warnings,
    }
