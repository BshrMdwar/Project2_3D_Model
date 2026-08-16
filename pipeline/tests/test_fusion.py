import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from models.fusion import (
    MeanPooling, MaxPooling, AttentionPooling, get_pooling,
    GeometryFusion, GeometryOnlyProjection,
)


def test_mean_pooling_shape_and_value():
    pooling = MeanPooling()
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (B=1, V=2, D=2)
    out = pooling(x)
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[2.0, 3.0]]))


def test_max_pooling_shape_and_value():
    pooling = MaxPooling()
    x = torch.tensor([[[1.0, 5.0], [3.0, 2.0]]])
    out = pooling(x)
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[3.0, 5.0]]))


def test_attention_pooling_output_shape():
    pooling = AttentionPooling(embedding_dim=16)
    x = torch.rand(4, 12, 16)  # B=4, V=12, D=16
    out = pooling(x)
    assert out.shape == (4, 16)


def test_attention_pooling_weights_sum_to_one_effectively():
    # verify pooled output lies within convex hull of view embeddings (weighted avg property)
    pooling = AttentionPooling(embedding_dim=4)
    x = torch.rand(2, 5, 4)
    out = pooling(x)
    mins = x.min(dim=1).values
    maxs = x.max(dim=1).values
    assert (out >= mins - 1e-4).all() and (out <= maxs + 1e-4).all()


def test_get_pooling_factory_all_methods():
    for method in ["mean", "max", "attention"]:
        pooling = get_pooling(method, embedding_dim=8)
        x = torch.rand(2, 6, 8)
        out = pooling(x)
        assert out.shape == (2, 8)


def test_get_pooling_invalid_method_raises():
    try:
        get_pooling("nonexistent_method", embedding_dim=8)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_geometry_fusion_disabled_is_passthrough():
    fusion = GeometryFusion(enabled=False)
    pooled = torch.rand(3, 16)
    geometry = torch.rand(3, 5)
    out = fusion(pooled, geometry)
    assert torch.allclose(out, pooled)
    assert out.shape == pooled.shape


def test_geometry_fusion_enabled_concatenates():
    fusion = GeometryFusion(enabled=True, geometry_dim=5)
    pooled = torch.rand(3, 16)
    geometry = torch.rand(3, 5)
    out = fusion(pooled, geometry)
    assert out.shape == (3, 21)
    assert torch.allclose(out[:, :16], pooled)
    assert torch.allclose(out[:, 16:], geometry)


def test_geometry_fusion_enabled_but_none_vector_is_passthrough():
    # e.g. a sample where geometry extraction failed entirely -- must not crash
    fusion = GeometryFusion(enabled=True, geometry_dim=5)
    pooled = torch.rand(2, 16)
    out = fusion(pooled, None)
    assert torch.allclose(out, pooled)


def test_geometry_only_projection_shape():
    proj = GeometryOnlyProjection(geometry_dim=10, output_dim=64)
    geometry = torch.rand(4, 10)
    out = proj(geometry)
    assert out.shape == (4, 64)


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
