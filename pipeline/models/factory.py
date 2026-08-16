"""
models/factory.py

Assembles the complete model (backbone -> pooling -> geometry fusion -> heads)
from a PipelineConfig + dataset.ActiveTaxonomy. This is the ONE place that
wires all the swappable pieces together; every other models/ file only
implements one piece and knows nothing about the others.

Handles the three fusion "modes" from spec section 14.4:
    1. Images only                     (use_geometry_fusion=False)
    2. Images + geometry fusion        (use_geometry_fusion=True, normal backbone)
    3. Geometry only                   (freeze_visual_backbone_entirely=True --
       the visual backbone still runs forward() for shape consistency in the
       training loop, but ALL its parameters are frozen -- i.e. it contributes
       a fixed, non-learned embedding -- and predictions rely on
       GeometryOnlyProjection for the actual learned signal instead)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from config import PipelineConfig
from dataset import ActiveTaxonomy
from models.backbones import get_backbone
from models.fusion import get_pooling, GeometryFusion, GeometryOnlyProjection
from models.heads import MultiTaskHeads

logger = logging.getLogger(__name__)


class CompleteModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooling: nn.Module,
        fusion: GeometryFusion,
        heads: MultiTaskHeads,
        geometry_only_projection: GeometryOnlyProjection | None = None,
        freeze_visual_backbone_entirely: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.fusion = fusion
        self.heads = heads
        self.geometry_only_projection = geometry_only_projection
        self.freeze_visual_backbone_entirely = freeze_visual_backbone_entirely

        if freeze_visual_backbone_entirely:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info(
                "freeze_visual_backbone_entirely=True: ALL backbone parameters frozen. "
                "Visual features act as a fixed (non-learned) input; "
                "GeometryOnlyProjection carries the actual learned signal for this ablation."
            )

    def forward(
        self,
        images: torch.Tensor,
        geometry_vector: torch.Tensor | None = None,
        super_category_gt: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            images: (B, V, 3, H, W)
            geometry_vector: (B, F) or None
            super_category_gt: (B,) long tensor, or None (see heads.py forward()
                docstring for the teacher-forcing routing behavior).
        """
        view_embeddings = self.backbone(images)        # (B, V, D)
        pooled = self.pooling(view_embeddings)           # (B, D)

        if self.freeze_visual_backbone_entirely and self.geometry_only_projection is not None:
            if geometry_vector is None:
                raise ValueError(
                    "freeze_visual_backbone_entirely=True requires a geometry_vector "
                    "on every batch (geometry-only ablation) -- got None. Check that "
                    "use_geometry_fusion=True is also set for this experiment config."
                )
            geo_projected = self.geometry_only_projection(geometry_vector)  # (B, hidden)
            # In this mode, the frozen visual pooled embedding is concatenated
            # alongside the geometry projection (frozen backbone still supplies
            # SOME signal, just non-learned) -- heads then see [pooled(frozen); geo_projected].
            fused = torch.cat([pooled, geo_projected], dim=-1)
        else:
            fused = self.fusion(pooled, geometry_vector)   # (B, D) or (B, D+F)

        return self.heads(fused, super_category_gt=super_category_gt)


def build_model(config: PipelineConfig, active_taxonomy: ActiveTaxonomy) -> CompleteModel:
    """
    Main entry point used by train.py. Builds the full model from config +
    the dataset's active taxonomy (dynamic head sizes, per spec section 6).
    """
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
        if not config.model.use_geometry_fusion:
            raise ValueError(
                "model.freeze_visual_backbone_entirely=True requires "
                "model.use_geometry_fusion=True as well (this ablation mode needs "
                "a geometry vector to learn from, since the visual path is frozen)."
            )
        geo_hidden_dim = 128
        geometry_only_projection = GeometryOnlyProjection(
            geometry_dim=geometry_dim, output_dim=geo_hidden_dim
        )
        # heads see [frozen pooled visual embedding ; learned geometry projection]
        heads_input_dim = embedding_dim + geo_hidden_dim

    heads = MultiTaskHeads(
        input_dim=heads_input_dim,
        active_super_categories=active_taxonomy.active_super_categories,
        active_object_categories=active_taxonomy.active_object_categories,
    )

    model = CompleteModel(
        backbone=backbone,
        pooling=pooling,
        fusion=fusion,
        heads=heads,
        geometry_only_projection=geometry_only_projection,
        freeze_visual_backbone_entirely=config.model.freeze_visual_backbone_entirely,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model built for experiment '{config.experiment_name}': "
        f"backbone={config.model.backbone_name}, pooling={config.model.pooling_method}, "
        f"geometry_fusion={config.model.use_geometry_fusion}, "
        f"{total_params:,} total params, {trainable_params:,} trainable "
        f"({100 * trainable_params / max(total_params, 1):.1f}%)."
    )

    return model
