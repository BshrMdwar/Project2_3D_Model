import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import seed as seed_mod


def test_python_random_reproducible():
    seed_mod.set_seed(42)
    a = [random.random() for _ in range(5)]
    seed_mod.set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b, "random module not reproducible after set_seed"


def test_numpy_reproducible():
    seed_mod.set_seed(123)
    a = np.random.rand(5)
    seed_mod.set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b), "numpy not reproducible after set_seed"


def test_different_seeds_differ():
    seed_mod.set_seed(1)
    a = np.random.rand(5)
    seed_mod.set_seed(2)
    b = np.random.rand(5)
    assert not np.allclose(a, b), "different seeds produced identical output (suspicious)"


def test_torch_reproducible_if_available():
    if not seed_mod._TORCH_AVAILABLE:
        print("  SKIP  torch not installed in this environment, skipping torch check")
        return
    import torch
    seed_mod.set_seed(7)
    a = torch.rand(5)
    seed_mod.set_seed(7)
    b = torch.rand(5)
    assert torch.allclose(a, b), "torch not reproducible after set_seed"


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
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
