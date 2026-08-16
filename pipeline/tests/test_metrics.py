import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import metrics as m


def test_single_label_metrics_perfect_prediction():
    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 1, 2, 0, 1]
    result = m.compute_single_label_metrics(y_true, y_pred, num_classes=3)
    assert result["accuracy"] == 1.0
    assert result["f1_macro"] == 1.0


def test_single_label_metrics_empty_input():
    result = m.compute_single_label_metrics([], [], num_classes=3)
    assert result["n_samples"] == 0
    assert result["accuracy"] == 0.0


def test_single_label_metrics_no_crash_on_rare_class_absent_from_batch():
    # class 2 never appears in this batch at all -- zero_division should not crash/warn-fail
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    result = m.compute_single_label_metrics(y_true, y_pred, num_classes=3)
    assert 0.0 <= result["f1_macro"] <= 1.0


def test_confusion_matrix_shape():
    y_true = [0, 1, 2, 1]
    y_pred = [0, 1, 1, 1]
    cm = m.compute_confusion_matrix(y_true, y_pred, num_classes=3)
    assert cm.shape == (3, 3)


def test_multi_label_metrics_perfect_prediction():
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[1, 0, 1], [0, 1, 0]])
    result = m.compute_multi_label_metrics(y_true, y_pred)
    assert result["f1_micro"] == 1.0
    assert result["f1_macro"] == 1.0


def test_multi_label_metrics_empty_input():
    result = m.compute_multi_label_metrics(np.zeros((0, 3)), np.zeros((0, 3)))
    assert result["n_samples"] == 0


def test_object_category_accumulator_routes_by_super_category():
    acc = m.ObjectCategoryMetricsAccumulator()
    acc.add_batch(["Bed", "Chair", "Bed"], [0, 0, 1], [0, 0, 1])
    acc.add_batch(["Carpet"], [-1], [-1])  # should be skipped entirely (no object_category)
    report = acc.report({"Bed": 2, "Chair": 1})
    assert report["per_super_category"]["Bed"]["n_samples"] == 2
    assert report["per_super_category"]["Chair"]["n_samples"] == 1
    assert "Carpet" not in report["per_super_category"]
    assert 0.0 <= report["macro_avg_f1"] <= 1.0


def test_object_category_accumulator_empty_super_category_reports_zero():
    acc = m.ObjectCategoryMetricsAccumulator()
    acc.add_batch(["Bed"], [0], [0])
    report = acc.report({"Bed": 2, "Window": 3})  # Window never appears in any batch
    assert report["per_super_category"]["Window"]["n_samples"] == 0


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
