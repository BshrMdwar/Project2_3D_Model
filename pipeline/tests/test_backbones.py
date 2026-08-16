"""
Tests for models/backbones.py.

Real backbones (DINOv2/ConvNeXt/EfficientNet/CLIP) require downloading pretrained
weights over the network, which is not available in this sandbox (see network
notes in the design doc). So this test suite verifies:
    1. The Backbone ABC interface contract itself, using a lightweight FAKE
       backbone that doesn't need any download.
    2. The freeze-except-last-N helper logic in isolation.
    3. The registry structure (keys exist, point to valid classes) WITHOUT
       actually instantiating the network-downloading backbones.

Full instantiation of real backbones (DINOv2Backbone(), etc.) should be smoke-
tested once on the actual training machine (with internet), not here.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from models.backbones import Backbone, BACKBONE_REGISTRY


class FakeBackbone(Backbone):
    """Minimal concrete Backbone for interface-contract testing -- no pretrained weights."""

    def __init__(self, embedding_dim: int = 16, num_blocks: int = 4, freeze_except_last_n: int = 1):
        super().__init__()
        self._embedding_dim = embedding_dim
        self.blocks = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(num_blocks)])
        self.stem = nn.Conv2d(3, embedding_dim, kernel_size=16, stride=16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._freeze_all_except_last_n(list(self.blocks), freeze_except_last_n)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat, b, v = self._reshape_for_2d_backbone(images)
        feat = self.pool(self.stem(flat)).flatten(1)  # (B*V, D)
        for block in self.blocks:
            feat = block(feat)
        return feat.reshape(b, v, self._embedding_dim)


def test_fake_backbone_output_shape():
    backbone = FakeBackbone(embedding_dim=16)
    images = torch.rand(2, 12, 3, 32, 32)  # B=2, V=12
    out = backbone(images)
    assert out.shape == (2, 12, 16)


def test_embedding_dim_property():
    backbone = FakeBackbone(embedding_dim=32)
    assert backbone.embedding_dim == 32


def test_freeze_except_last_n_freezes_correctly():
    backbone = FakeBackbone(embedding_dim=8, num_blocks=4, freeze_except_last_n=1)
    # stem should be frozen (not in `blocks` passed to the freeze helper)
    assert not backbone.stem.weight.requires_grad
    # only the LAST block should be trainable
    assert not backbone.blocks[0].weight.requires_grad
    assert not backbone.blocks[1].weight.requires_grad
    assert not backbone.blocks[2].weight.requires_grad
    assert backbone.blocks[3].weight.requires_grad


def test_freeze_except_last_n_zero_freezes_everything():
    backbone = FakeBackbone(embedding_dim=8, num_blocks=3, freeze_except_last_n=0)
    for block in backbone.blocks:
        assert not block.weight.requires_grad


def test_freeze_except_last_n_all_unfreezes_everything_in_blocks():
    backbone = FakeBackbone(embedding_dim=8, num_blocks=3, freeze_except_last_n=3)
    for block in backbone.blocks:
        assert block.weight.requires_grad


def test_registry_keys_present():
    expected_keys = {
        "dinov2_vits14", "dinov2_vitb14",
        "convnext_tiny", "convnext_small",
        "efficientnet_b0", "clip_vit_b32",
    }
    assert expected_keys.issubset(set(BACKBONE_REGISTRY.keys()))


def test_registry_values_are_backbone_subclasses():
    for name, (cls, variant) in BACKBONE_REGISTRY.items():
        assert issubclass(cls, Backbone), f"{name} does not map to a Backbone subclass"


def test_gradients_flow_through_unfrozen_block_only():
    backbone = FakeBackbone(embedding_dim=8, num_blocks=2, freeze_except_last_n=1)
    images = torch.rand(1, 3, 3, 32, 32, requires_grad=False)
    out = backbone(images)
    loss = out.sum()
    loss.backward()
    assert backbone.blocks[1].weight.grad is not None
    assert backbone.blocks[0].weight.grad is None  # frozen -- requires_grad=False, no grad computed


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
