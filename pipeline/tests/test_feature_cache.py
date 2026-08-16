import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from feature_cache import FeatureCache


def test_disabled_cache_always_returns_none():
    with tempfile.TemporaryDirectory() as d:
        cache = FeatureCache(d, backbone_name="dinov2_vits14", backbone_is_fully_frozen=False)
        cache.set("sample1", np.random.rand(384))
        assert cache.get("sample1") is None


def test_enabled_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        cache = FeatureCache(d, backbone_name="dinov2_vits14", backbone_is_fully_frozen=True)
        vec = np.random.rand(384).astype(np.float32)
        cache.set("sample1", vec)
        loaded = cache.get("sample1")
        assert loaded is not None
        assert np.allclose(loaded, vec)


def test_missing_key_returns_none():
    with tempfile.TemporaryDirectory() as d:
        cache = FeatureCache(d, backbone_name="dinov2_vits14", backbone_is_fully_frozen=True)
        assert cache.get("nonexistent_sample") is None


def test_different_backbones_do_not_collide():
    with tempfile.TemporaryDirectory() as d:
        cache_a = FeatureCache(d, backbone_name="dinov2_vits14", backbone_is_fully_frozen=True)
        cache_b = FeatureCache(d, backbone_name="convnext_tiny", backbone_is_fully_frozen=True)
        vec_a = np.ones(10, dtype=np.float32)
        vec_b = np.zeros(10, dtype=np.float32)
        cache_a.set("sample1", vec_a)
        cache_b.set("sample1", vec_b)
        assert np.allclose(cache_a.get("sample1"), vec_a)
        assert np.allclose(cache_b.get("sample1"), vec_b)


def test_clear_removes_all_cached_files():
    with tempfile.TemporaryDirectory() as d:
        cache = FeatureCache(d, backbone_name="dinov2_vits14", backbone_is_fully_frozen=True)
        cache.set("sample1", np.random.rand(10))
        cache.set("sample2", np.random.rand(10))
        assert len(list(Path(d).glob("*.npy"))) == 2
        cache.clear()
        assert len(list(Path(d).glob("*.npy"))) == 0


def test_corrupt_cache_file_handled_gracefully():
    with tempfile.TemporaryDirectory() as d:
        cache = FeatureCache(d, backbone_name="dinov2_vits14", backbone_is_fully_frozen=True)
        path = cache._cache_path("bad_sample")
        path.write_bytes(b"not a valid npy file")
        assert cache.get("bad_sample") is None  # should not raise


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
