import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

import config as cfg
from losses import MultiTaskLoss
import taxonomy as tx


def _fake_head_outputs(batch_size=3, num_super=3, num_style=len(tx.STYLE_CLASSES), num_mat=len(tx.MATERIALS)):
    return {
        "super_category_logits": torch.randn(batch_size, num_super, requires_grad=True),
        "object_category_logits": [
            torch.randn(2, requires_grad=True),  # sample 0 -> 2-class subhead
            None,                                  # sample 1 -> no subhead (e.g. Carpet)
            torch.randn(3, requires_grad=True),   # sample 2 -> 3-class subhead
        ],
        "style_logits": torch.randn(batch_size, num_style, requires_grad=True),
        "materials_primary_logits": torch.randn(batch_size, num_mat, requires_grad=True),
        "materials_secondary_logits": torch.randn(batch_size, num_mat, requires_grad=True),
    }


def _fake_labels(batch_size=3, num_style=len(tx.STYLE_CLASSES), num_mat=len(tx.MATERIALS)):
    return {
        "super_category": torch.tensor([0, 1, 2]),
        "object_category": torch.tensor([1, -1, 2]),  # sample 1 has no valid object_category (Carpet)
        "style_class": torch.zeros(batch_size, num_style),
        "materials_primary": torch.tensor([0, -1, 1]),  # sample 1 missing materials_primary
        "materials_secondary": torch.zeros(batch_size, num_mat),
    }


def test_total_loss_is_finite_and_scalar():
    loss_config = cfg.LossConfig(use_class_weighted_loss=False)
    loss_fn = MultiTaskLoss(loss_config, num_super_categories=3, object_category_class_counts={"A": 2, "B": 3})
    outputs = _fake_head_outputs()
    labels = _fake_labels()
    weights = torch.ones(3)
    result = loss_fn(outputs, labels, weights)
    assert result["total_loss"].dim() == 0
    assert torch.isfinite(result["total_loss"])


def test_object_category_loss_skips_none_entries_gracefully():
    loss_config = cfg.LossConfig(use_class_weighted_loss=False)
    loss_fn = MultiTaskLoss(loss_config, num_super_categories=3, object_category_class_counts={"A": 2, "B": 3})
    outputs = _fake_head_outputs()
    labels = _fake_labels()
    weights = torch.ones(3)
    result = loss_fn(outputs, labels, weights)
    assert torch.isfinite(result["object_category_loss"])


def test_all_object_category_none_gives_zero_loss_not_crash():
    loss_config = cfg.LossConfig(use_class_weighted_loss=False)
    loss_fn = MultiTaskLoss(loss_config, num_super_categories=3, object_category_class_counts={})
    outputs = _fake_head_outputs()
    outputs["object_category_logits"] = [None, None, None]
    labels = _fake_labels()
    labels["object_category"] = torch.tensor([-1, -1, -1])
    weights = torch.ones(3)
    result = loss_fn(outputs, labels, weights)
    assert result["object_category_loss"].item() == 0.0


def test_materials_primary_missing_all_gives_zero_not_crash():
    loss_config = cfg.LossConfig(use_class_weighted_loss=False)
    loss_fn = MultiTaskLoss(loss_config, num_super_categories=3, object_category_class_counts={"A": 2, "B": 3})
    outputs = _fake_head_outputs()
    labels = _fake_labels()
    labels["materials_primary"] = torch.tensor([-1, -1, -1])
    weights = torch.ones(3)
    result = loss_fn(outputs, labels, weights)
    assert result["materials_primary_loss"].item() == 0.0


def test_class_weighted_loss_runs():
    loss_config = cfg.LossConfig(use_class_weighted_loss=True)
    loss_fn = MultiTaskLoss(
        loss_config, num_super_categories=3, object_category_class_counts={"A": 2, "B": 3},
        super_category_labels_for_weights=[0, 0, 1, 2],
        materials_primary_labels_for_weights=[0, 1, 1, 1],
    )
    outputs = _fake_head_outputs()
    labels = _fake_labels()
    weights = torch.ones(3)
    result = loss_fn(outputs, labels, weights)
    assert torch.isfinite(result["total_loss"])


def test_focal_loss_runs():
    loss_config = cfg.LossConfig(use_class_weighted_loss=False, use_focal_loss=True)
    loss_fn = MultiTaskLoss(loss_config, num_super_categories=3, object_category_class_counts={"A": 2, "B": 3})
    outputs = _fake_head_outputs()
    labels = _fake_labels()
    weights = torch.ones(3)
    result = loss_fn(outputs, labels, weights)
    assert torch.isfinite(result["total_loss"])


def test_sample_weighting_changes_loss_value():
    loss_config = cfg.LossConfig(use_class_weighted_loss=False)
    loss_fn = MultiTaskLoss(loss_config, num_super_categories=3, object_category_class_counts={"A": 2, "B": 3})
    outputs = _fake_head_outputs()
    labels = _fake_labels()

    uniform_weights = torch.ones(3)
    skewed_weights = torch.tensor([0.1, 0.1, 5.0])

    result_uniform = loss_fn(outputs, labels, uniform_weights)
    result_skewed = loss_fn(outputs, labels, skewed_weights)
    assert not torch.isclose(result_uniform["total_loss"], result_skewed["total_loss"])


def test_backward_pass_produces_gradients():
    loss_config = cfg.LossConfig(use_class_weighted_loss=False)
    loss_fn = MultiTaskLoss(loss_config, num_super_categories=3, object_category_class_counts={"A": 2, "B": 3})
    outputs = _fake_head_outputs()
    labels = _fake_labels()
    weights = torch.ones(3)
    result = loss_fn(outputs, labels, weights)
    result["total_loss"].backward()
    assert outputs["super_category_logits"].grad is not None
    assert outputs["style_logits"].grad is not None


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
