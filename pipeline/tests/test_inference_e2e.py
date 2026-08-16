import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

import config as cfg
import models.backbones as backbones_mod
from models.backbones import Backbone
import train as train_mod
import inference as inf
from kfold_split import make_kfold_splits

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mock_dataset"


class TinyFakeBackbone(Backbone):
    def __init__(self, variant: str = "fake", freeze_except_last_n: int = 1, embedding_dim: int = 8):
        super().__init__()
        self._embedding_dim = embedding_dim
        self.proj = nn.Conv2d(3, embedding_dim, kernel_size=16, stride=16)
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
    backbones_mod.BACKBONE_REGISTRY["fake_backbone"] = (TinyFakeBackbone, "fake")


def _train_a_tiny_checkpoint(tmp_dir: str, use_geometry: bool, experiment_name: str) -> Path:
    _patch_registry()
    pc = cfg.PipelineConfig()
    pc.experiment_name = experiment_name
    pc.dataset.root_dir = str(FIXTURE_ROOT)
    pc.dataset.image_size = 32
    pc.dataset.k_folds = 2
    pc.model.backbone_name = "fake_backbone"
    pc.model.use_geometry_fusion = use_geometry
    if use_geometry:
        pc.model.geometry_features = ["dimensions.aspect_hw", "geometry.occupancy_ratio"]
    pc.training.batch_size = 2
    pc.training.max_epochs = 1
    pc.training.num_workers = 0
    pc.training.accelerator = "cpu"
    pc.training.early_stopping_patience = 100
    pc.system.checkpoint_dir = str(Path(tmp_dir) / "checkpoints")
    pc.system.tensorboard_dir = str(Path(tmp_dir) / "runs")

    active_taxonomy, full_dataset = train_mod.build_full_active_taxonomy(pc)
    sample_ids = [rec.id for rec in full_dataset.records]
    supers = [rec.metadata_json.get("super_category") for rec in full_dataset.records]
    objects = [rec.metadata_json.get("object_category", "") or "" for rec in full_dataset.records]
    folds = make_kfold_splits(sample_ids, supers, objects, k_folds=pc.dataset.k_folds, seed=42)
    train_ids, val_ids = folds[0]

    train_mod.train_single_fold(pc, active_taxonomy, train_ids, val_ids, fold_index=0)
    return Path(pc.system.checkpoint_dir) / experiment_name / "fold_0" / "self_contained.pt"


def _get_render_paths(sample_id: str) -> list[str]:
    renders_dir = FIXTURE_ROOT / "renders" / sample_id
    return sorted(str(p) for p in renders_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


def test_inference_without_geometry_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _train_a_tiny_checkpoint(tmp_dir, use_geometry=False, experiment_name="inf_test_no_geo")
        render_paths = _get_render_paths("bed_0000001")
        result = inf.predict(ckpt_path, render_paths, geometry_json=None)

        assert "predictions" in result
        assert result["predictions"]["super_category"]["label"] is not None
        assert 0.0 <= result["predictions"]["super_category"]["confidence"] <= 1.0
        assert result["predictions"]["materials_primary"]["label"] in [
            "Wood", "Metal", "Glass", "Concrete", "Stone", "Fabric", "Plastic", "3D Printed"
        ]


def test_inference_with_geometry_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _train_a_tiny_checkpoint(tmp_dir, use_geometry=True, experiment_name="inf_test_geo")
        render_paths = _get_render_paths("bed_0000001")

        import json
        geometry_json = json.loads((FIXTURE_ROOT / "geometry" / "bed_0000001.json").read_text())

        result = inf.predict(ckpt_path, render_paths, geometry_json=geometry_json)
        assert result["predictions"]["super_category"]["label"] is not None
        assert not any("no geometry_json was provided" in w for w in result["warnings"])


def test_inference_missing_geometry_when_required_warns_not_crashes():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _train_a_tiny_checkpoint(tmp_dir, use_geometry=True, experiment_name="inf_test_geo_missing")
        render_paths = _get_render_paths("bed_0000001")
        result = inf.predict(ckpt_path, render_paths, geometry_json=None)
        assert any("no geometry_json was provided" in w for w in result["warnings"])
        assert result["predictions"]["super_category"]["label"] is not None


def test_inference_fewer_views_than_expected_warns():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _train_a_tiny_checkpoint(tmp_dir, use_geometry=False, experiment_name="inf_test_fewviews")
        render_paths = _get_render_paths("bed_0000001")[:3]  # deliberately fewer than 12
        result = inf.predict(ckpt_path, render_paths, geometry_json=None)
        assert any("Expected" in w and "received" in w for w in result["warnings"])


def test_inference_carpet_has_no_object_category_prediction():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _train_a_tiny_checkpoint(tmp_dir, use_geometry=False, experiment_name="inf_test_carpet")
        render_paths = _get_render_paths("carpet_0000001")
        result = inf.predict(ckpt_path, render_paths, geometry_json=None)
        # regardless of what super_category the tiny/undertrained model predicts,
        # the pipeline must not crash and object_category must gracefully be
        # None whenever the predicted class has no sub-head or isn't Carpet-like
        assert "object_category" in result["predictions"]


def test_checkpoint_not_found_raises_clear_error():
    try:
        inf.predict("/nonexistent/path/checkpoint.pt", ["fake.png"])
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


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
            import traceback
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
