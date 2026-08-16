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
from kfold_split import make_kfold_splits
from visualization.plot_embeddings import extract_fused_embeddings, plot_tsne

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


def _train_checkpoint(tmp_dir):
    _patch_registry()
    pc = cfg.PipelineConfig()
    pc.experiment_name = "viz_test_exp"
    pc.dataset.root_dir = str(FIXTURE_ROOT)
    pc.dataset.image_size = 32
    pc.dataset.k_folds = 2
    pc.model.backbone_name = "fake_backbone"
    pc.model.use_geometry_fusion = False
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
    return Path(pc.system.checkpoint_dir) / "viz_test_exp" / "fold_0" / "self_contained.pt"


def test_extract_fused_embeddings_shape():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _train_checkpoint(tmp_dir)
        data = extract_fused_embeddings(str(ckpt_path))
        assert data["embeddings"].shape[0] == 5  # 5 valid samples in mock dataset
        assert len(data["super_category"]) == 5


def test_plot_tsne_creates_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _train_checkpoint(tmp_dir)
        data = extract_fused_embeddings(str(ckpt_path))
        output_path = str(Path(tmp_dir) / "plot.png")
        plot_tsne(data, color_by="super_category", output_path=output_path)
        assert Path(output_path).exists()


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
