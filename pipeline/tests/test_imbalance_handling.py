import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

import imbalance_handling as ih


def test_class_weights_inverse_frequency():
    labels = [0, 0, 0, 0, 1]  # class 0: 4 samples, class 1: 1 sample
    weights = ih.compute_class_weights(labels, num_classes=2)
    assert weights[0] < weights[1], "minority class should get a higher weight"


def test_class_weights_zero_for_absent_class():
    labels = [0, 0, 1, 1]
    weights = ih.compute_class_weights(labels, num_classes=3)  # class 2 has 0 samples
    assert weights[2] == 0.0


def test_focal_loss_runs_and_reduces_easy_example_loss():
    loss_fn = ih.FocalLoss(gamma=2.0)
    # confident-correct prediction (easy example)
    easy_logits = torch.tensor([[10.0, 0.0]])
    easy_target = torch.tensor([0])
    # unconfident/incorrect prediction (hard example)
    hard_logits = torch.tensor([[0.0, 0.1]])
    hard_target = torch.tensor([0])

    easy_loss = loss_fn(easy_logits, easy_target)
    hard_loss = loss_fn(hard_logits, hard_target)
    assert easy_loss.item() < hard_loss.item(), "focal loss should down-weight easy examples"


def test_focal_loss_with_class_weights():
    weights = torch.tensor([2.0, 0.5])
    loss_fn = ih.FocalLoss(gamma=2.0, class_weights=weights)
    logits = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    targets = torch.tensor([0, 1])
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_balanced_sampling_weights_favor_minority():
    labels = [0, 0, 0, 0, 1]
    weights = ih.compute_sample_weights_for_balanced_sampling(labels)
    assert weights[-1] > weights[0], "minority sample should get higher per-sample weight"


def test_weighted_random_sampler_builds_and_samples():
    labels = [0, 0, 0, 1]
    sampler = ih.build_weighted_random_sampler(labels)
    indices = list(sampler)
    assert len(indices) == len(labels)


def test_low_confidence_combos_flags_rare_combinations():
    supers = ["Bed", "Bed", "Bed", "Chair"]
    objects = ["King", "King", "Queen", "Armchair"]
    flagged = ih.compute_low_confidence_combos(supers, objects, min_samples=2)
    assert ("Bed", "Queen") in flagged  # only 1 sample
    assert ("Chair", "Armchair") in flagged  # only 1 sample
    assert ("Bed", "King") not in flagged  # 2 samples, meets threshold


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
