import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import config as cfg
import augmentation as aug


def _dummy_views(v=12, h=32, w=32):
    return np.random.rand(v, h, w, 3).astype(np.float32)


def test_output_shape_preserved():
    config = cfg.AugmentationConfig()
    pipeline = aug.MultiViewAugmentation(config, rng=np.random.default_rng(0))
    views = _dummy_views()
    out = pipeline(views)
    assert out.shape == views.shape
    assert out.dtype == np.float32


def test_output_in_valid_range():
    config = cfg.AugmentationConfig()
    pipeline = aug.MultiViewAugmentation(config, rng=np.random.default_rng(1))
    views = _dummy_views()
    out = pipeline(views)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_all_views_get_same_flip_decision():
    config = cfg.AugmentationConfig(
        use_background_synthesis=False, use_color_jitter=False,
        use_random_scale=False, use_rotation=False,
        use_horizontal_flip=True, horizontal_flip_prob=1.0,  # force flip always
    )
    pipeline = aug.MultiViewAugmentation(config, rng=np.random.default_rng(2))
    views = _dummy_views(v=4)
    out = pipeline(views)
    # every view should be exactly horizontally flipped (since it's the only op enabled)
    for v in range(4):
        assert np.allclose(out[v], views[v][:, ::-1, :])


def test_all_views_get_same_color_jitter_not_independent_random():
    config = cfg.AugmentationConfig(
        use_background_synthesis=False, use_random_scale=False,
        use_rotation=False, use_horizontal_flip=False,
        use_color_jitter=True, color_jitter_strength=0.5,
    )
    # use a constant-value image so we can directly check the SAME multiplicative
    # factor was applied to every view (multi-view consistency requirement)
    views = np.full((6, 8, 8, 3), 0.5, dtype=np.float32)
    pipeline = aug.MultiViewAugmentation(config, rng=np.random.default_rng(3))
    out = pipeline(views)
    # all views identical input -> all views should get identical output
    # (since params are sampled once and img is uniform, contrast/brightness collapse identically)
    first = out[0]
    for v in range(1, 6):
        assert np.allclose(out[v], first, atol=1e-5), "views received inconsistent augmentation params"


def test_disabling_all_augmentation_is_near_identity():
    config = cfg.AugmentationConfig(
        use_background_synthesis=False, use_color_jitter=False,
        use_random_scale=False, use_horizontal_flip=False, use_rotation=False,
    )
    pipeline = aug.MultiViewAugmentation(config, rng=np.random.default_rng(4))
    views = _dummy_views()
    out = pipeline(views)
    assert np.allclose(out, views, atol=1e-6)


def test_scale_preserves_shape():
    config = cfg.AugmentationConfig(
        use_background_synthesis=False, use_color_jitter=False,
        use_horizontal_flip=False, use_rotation=False,
        use_random_scale=True, scale_range=(0.7, 1.3),
    )
    pipeline = aug.MultiViewAugmentation(config, rng=np.random.default_rng(5))
    views = _dummy_views(h=40, w=40)
    out = pipeline(views)
    assert out.shape == views.shape


def test_reproducible_with_same_rng_seed():
    config = cfg.AugmentationConfig()
    views = _dummy_views()
    p1 = aug.MultiViewAugmentation(config, rng=np.random.default_rng(99))
    p2 = aug.MultiViewAugmentation(config, rng=np.random.default_rng(99))
    out1 = p1(views)
    out2 = p2(views)
    assert np.allclose(out1, out2)


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
