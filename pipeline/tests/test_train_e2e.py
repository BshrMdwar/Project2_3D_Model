"""
End-to-end smoke test for train.py: runs a REAL (tiny) training loop against
the mock dataset fixture, with the backbone registry monkeypatched to a fake
no-download backbone (so this runs fully offline in seconds).

This is intentionally a smoke test (does it run without crashing, do losses
decrease/stay finite, is a checkpoint produced and loadable) rather than a
check of final model quality, since the mock dataset has only ~6 samples.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

import config as cfg
import models.backbones as backbones_mod
from models.backbones import Backbone
import train as train_mod

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


def test_full_training_loop_runs_one_epoch_no_crash():
    _patch_registry()

    with tempfile.TemporaryDirectory() as tmp_dir:
        pc = cfg.PipelineConfig()
        pc.experiment_name = "smoke_test_exp"
        pc.dataset.root_dir = str(FIXTURE_ROOT)
        pc.dataset.image_size = 32  # tiny for speed
        pc.dataset.k_folds = 2       # small dataset, keep folds small
        pc.model.backbone_name = "fake_backbone"
        pc.model.use_geometry_fusion = False
        pc.training.batch_size = 2
        pc.training.max_epochs = 1
        pc.training.num_workers = 0
        pc.training.accelerator = "cpu"
        pc.training.early_stopping_patience = 100  # don't early-stop in a 1-epoch smoke test
        pc.system.checkpoint_dir = str(Path(tmp_dir) / "checkpoints")
        pc.system.tensorboard_dir = str(Path(tmp_dir) / "runs")

        active_taxonomy, full_dataset = train_mod.build_full_active_taxonomy(pc)
        sample_ids = [rec.id for rec in full_dataset.records]
        supers = [rec.metadata_json.get("super_category") for rec in full_dataset.records]
        objects = [rec.metadata_json.get("object_category", "") or "" for rec in full_dataset.records]

        from kfold_split import make_kfold_splits
        folds = make_kfold_splits(sample_ids, supers, objects, k_folds=pc.dataset.k_folds, seed=42)

        train_ids, val_ids = folds[0]
        result = train_mod.train_single_fold(pc, active_taxonomy, train_ids, val_ids, fold_index=0)

        assert "val_total_loss" in result
        assert not (result["val_total_loss"] != result["val_total_loss"])  # not NaN

        checkpoint_path = Path(pc.system.checkpoint_dir) / "smoke_test_exp" / "fold_0" / "self_contained.pt"
        assert checkpoint_path.exists(), "self-contained checkpoint was not created"

        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in loaded
        assert "active_super_categories" in loaded
        assert loaded["experiment_name"] == "smoke_test_exp"


def test_full_training_loop_with_geometry_fusion():
    _patch_registry()

    with tempfile.TemporaryDirectory() as tmp_dir:
        pc = cfg.PipelineConfig()
        pc.experiment_name = "smoke_test_geo_exp"
        pc.dataset.root_dir = str(FIXTURE_ROOT)
        pc.dataset.image_size = 32
        pc.dataset.k_folds = 2
        pc.model.backbone_name = "fake_backbone"
        pc.model.use_geometry_fusion = True
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

        from kfold_split import make_kfold_splits
        folds = make_kfold_splits(sample_ids, supers, objects, k_folds=pc.dataset.k_folds, seed=42)
        train_ids, val_ids = folds[0]

        result = train_mod.train_single_fold(pc, active_taxonomy, train_ids, val_ids, fold_index=0)
        assert "val_total_loss" in result

        checkpoint_path = Path(pc.system.checkpoint_dir) / "smoke_test_geo_exp" / "fold_0" / "self_contained.pt"
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert loaded["scaler"] is not None  # geometry fusion enabled -> scaler must be saved


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
