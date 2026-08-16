import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import kfold_split as kf


def test_basic_kfold_splits_are_correct_size():
    ids = [f"s{i}" for i in range(20)]
    supers = ["Bed"] * 10 + ["Chair"] * 10
    objects = ["King"] * 10 + ["Armchair"] * 10
    folds = kf.make_kfold_splits(ids, supers, objects, k_folds=5, seed=42)
    assert len(folds) == 5
    for train_ids, val_ids in folds:
        assert len(train_ids) + len(val_ids) == 20 or set(train_ids) | set(val_ids) == set(ids)


def test_every_sample_appears_in_validation_exactly_once_when_splittable():
    ids = [f"s{i}" for i in range(20)]
    supers = ["Bed"] * 10 + ["Chair"] * 10
    objects = ["King"] * 10 + ["Armchair"] * 10
    folds = kf.make_kfold_splits(ids, supers, objects, k_folds=5, seed=42)
    val_membership = {sid: 0 for sid in ids}
    for _, val_ids in folds:
        for sid in val_ids:
            val_membership[sid] += 1
    assert all(count == 1 for count in val_membership.values())


def test_under_populated_combo_appears_in_every_fold():
    # "Carpet::" has only 2 samples, less than k_folds=5 -> must appear in every fold
    ids = [f"bed{i}" for i in range(10)] + ["carpet0", "carpet1"]
    supers = ["Bed"] * 10 + ["Carpet", "Carpet"]
    objects = ["King"] * 10 + ["", ""]
    folds = kf.make_kfold_splits(ids, supers, objects, k_folds=5, seed=42)
    for train_ids, val_ids in folds:
        assert "carpet0" in train_ids and "carpet1" in train_ids
        assert "carpet0" in val_ids and "carpet1" in val_ids


def test_degenerate_all_under_populated_does_not_crash():
    ids = ["a", "b", "c"]
    supers = ["Carpet", "Carpet", "Window"]
    objects = ["", "", "Casement"]
    folds = kf.make_kfold_splits(ids, supers, objects, k_folds=5, seed=1)
    assert len(folds) == 5
    for train_ids, val_ids in folds:
        assert set(train_ids) == set(ids)
        assert set(val_ids) == set(ids)


def test_reproducible_with_same_seed():
    ids = [f"s{i}" for i in range(20)]
    supers = ["Bed"] * 10 + ["Chair"] * 10
    objects = ["King"] * 10 + ["Armchair"] * 10
    folds1 = kf.make_kfold_splits(ids, supers, objects, k_folds=5, seed=7)
    folds2 = kf.make_kfold_splits(ids, supers, objects, k_folds=5, seed=7)
    assert folds1 == folds2


def test_different_seed_gives_different_split():
    ids = [f"s{i}" for i in range(20)]
    supers = ["Bed"] * 10 + ["Chair"] * 10
    objects = ["King"] * 10 + ["Armchair"] * 10
    folds1 = kf.make_kfold_splits(ids, supers, objects, k_folds=5, seed=1)
    folds2 = kf.make_kfold_splits(ids, supers, objects, k_folds=5, seed=2)
    assert folds1 != folds2


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
