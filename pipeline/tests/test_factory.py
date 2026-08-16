import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

import config as cfg
from dataset import ActiveTaxonomy
import models.backbones as backbones_mod
import models.factory as factory_mod
from models.backbones import Backbone


class TinyFakeBackbone(Backbone):
    """No-download fake backbone for factory-level integration tests."""

    def __init__(self, variant: str = "fake", freeze_except_last_n: int = 1, embedding_dim: int = 16):
        super().__init__()
        self._embedding_dim = embedding_dim
        self.proj = nn.Conv2d(3, embedding_dim, kernel_size=8, stride=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.blocks = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim)])
        self._freeze_all_except_last_n(list(self.blocks), freeze_except_last_n)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat, b, v = self._reshape_for_2d_backbone(images)
        feat = self.pool(self.proj(flat)).flatten(1)
        feat = self.blocks[0](feat)
        return feat.reshape(b, v, self._embedding_dim)


def _patch_registry():
    """Monkeypatch the backbone registry so factory.build_model uses our fake backbone."""
    backbones_mod.BACKBONE_REGISTRY["fake_backbone"] = (TinyFakeBackbone, "fake")


def _make_active_taxonomy():
    metadata_jsons = [
        {"super_category": "Bed", "object_category": "King"},
        {"super_category": "Bed", "object_category": "Queen"},
        {"super_category": "Chair", "object_category": "Armchair"},
        {"super_category": "Carpet"},
    ]
    return ActiveTaxonomy(metadata_jsons)


def test_build_model_images_only_baseline():
    _patch_registry()
    pc = cfg.PipelineConfig()
    pc.model.backbone_name = "fake_backbone"
    pc.model.use_geometry_fusion = False
    active_tax = _make_active_taxonomy()

    model = factory_mod.build_model(pc, active_tax)
    images = torch.rand(2, 12, 3, 32, 32)
    out = model(images, geometry_vector=None, super_category_gt=torch.tensor([0, 1]))
    assert out["super_category_logits"].shape == (2, len(active_tax.active_super_categories))


def test_build_model_with_geometry_fusion():
    _patch_registry()
    pc = cfg.PipelineConfig()
    pc.model.backbone_name = "fake_backbone"
    pc.model.use_geometry_fusion = True
    pc.model.geometry_features = ["dimensions.aspect_hw", "geometry.occupancy_ratio"]
    active_tax = _make_active_taxonomy()

    model = factory_mod.build_model(pc, active_tax)
    images = torch.rand(2, 12, 3, 32, 32)
    geometry = torch.rand(2, 2)
    out = model(images, geometry_vector=geometry, super_category_gt=torch.tensor([0, 2]))
    assert out["super_category_logits"].shape == (2, len(active_tax.active_super_categories))
    # object_category_logits[1] corresponds to gt index 2 = "Carpet" -> must be None
    assert out["object_category_logits"][1] is None


def test_build_model_geometry_only_ablation_requires_geometry_fusion():
    _patch_registry()
    pc = cfg.PipelineConfig()
    pc.model.backbone_name = "fake_backbone"
    pc.model.use_geometry_fusion = False  # deliberately inconsistent
    pc.model.freeze_visual_backbone_entirely = True
    active_tax = _make_active_taxonomy()

    try:
        factory_mod.build_model(pc, active_tax)
        assert False, "expected ValueError for inconsistent config"
    except ValueError as e:
        assert "use_geometry_fusion" in str(e)


def test_build_model_geometry_only_ablation_works_and_freezes_backbone():
    _patch_registry()
    pc = cfg.PipelineConfig()
    pc.model.backbone_name = "fake_backbone"
    pc.model.use_geometry_fusion = True
    pc.model.geometry_features = ["dimensions.aspect_hw", "geometry.occupancy_ratio"]
    pc.model.freeze_visual_backbone_entirely = True
    active_tax = _make_active_taxonomy()

    model = factory_mod.build_model(pc, active_tax)
    for p in model.backbone.parameters():
        assert not p.requires_grad

    images = torch.rand(2, 12, 3, 32, 32)
    geometry = torch.rand(2, 2)
    out = model(images, geometry_vector=geometry, super_category_gt=torch.tensor([0, 1]))
    assert out["super_category_logits"].shape[0] == 2


def test_build_model_geometry_only_ablation_fails_without_geometry_vector_at_forward():
    _patch_registry()
    pc = cfg.PipelineConfig()
    pc.model.backbone_name = "fake_backbone"
    pc.model.use_geometry_fusion = True
    pc.model.geometry_features = ["dimensions.aspect_hw"]
    pc.model.freeze_visual_backbone_entirely = True
    active_tax = _make_active_taxonomy()
    model = factory_mod.build_model(pc, active_tax)

    images = torch.rand(1, 12, 3, 32, 32)
    try:
        model(images, geometry_vector=None, super_category_gt=torch.tensor([0]))
        assert False, "expected ValueError when geometry_vector is missing in geometry-only mode"
    except ValueError as e:
        assert "geometry_vector" in str(e)


def test_backbone_partial_freezing_applied_in_full_model():
    _patch_registry()
    pc = cfg.PipelineConfig()
    pc.model.backbone_name = "fake_backbone"
    pc.model.freeze_backbone_except_last_n_layers = 0
    active_tax = _make_active_taxonomy()
    model = factory_mod.build_model(pc, active_tax)
    assert not model.backbone.blocks[0].weight.requires_grad


def test_attention_pooling_end_to_end():
    _patch_registry()
    pc = cfg.PipelineConfig()
    pc.model.backbone_name = "fake_backbone"
    pc.model.pooling_method = "attention"
    active_tax = _make_active_taxonomy()
    model = factory_mod.build_model(pc, active_tax)
    images = torch.rand(2, 12, 3, 32, 32)
    out = model(images, super_category_gt=torch.tensor([0, 1]))
    assert out["super_category_logits"].shape == (2, len(active_tax.active_super_categories))


def _run_all():
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
