import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from models.heads import MultiTaskHeads
import taxonomy as tx

ACTIVE_SUPER = ["Bed", "Chair", "Carpet", "Storage"]
ACTIVE_OBJECT = {
    "Bed": ["King", "Queen"],
    "Chair": ["Armchair"],
    "Storage": ["Wardrobe"],
    # Carpet intentionally absent -- has_object_category("Carpet") == False
}


def _build_heads(input_dim=16):
    return MultiTaskHeads(
        input_dim=input_dim,
        active_super_categories=ACTIVE_SUPER,
        active_object_categories=ACTIVE_OBJECT,
    )


def test_super_category_head_output_shape():
    heads = _build_heads()
    x = torch.rand(3, 16)
    out = heads(x, super_category_gt=torch.tensor([0, 1, 2]))
    assert out["super_category_logits"].shape == (3, len(ACTIVE_SUPER))


def test_carpet_has_no_object_category_head_registered():
    heads = _build_heads()
    assert "Carpet" not in heads.object_category_heads
    assert "Bed" in heads.object_category_heads
    assert "Chair" in heads.object_category_heads
    assert "Storage" in heads.object_category_heads


def test_object_category_head_sizes_match_active_taxonomy():
    heads = _build_heads()
    assert heads.object_category_heads["Bed"].out_features == 2   # King, Queen
    assert heads.object_category_heads["Chair"].out_features == 1  # Armchair only
    assert heads.object_category_heads["Storage"].out_features == 1  # Wardrobe only


def test_forward_routes_to_correct_subhead_via_teacher_forcing():
    heads = _build_heads()
    x = torch.rand(4, 16)
    # gt indices: 0=Bed, 1=Chair, 2=Carpet, 3=Storage
    gt = torch.tensor([0, 1, 2, 3])
    out = heads(x, super_category_gt=gt)
    logits = out["object_category_logits"]
    assert logits[0].shape == (2,)   # Bed -> 2 classes
    assert logits[1].shape == (1,)   # Chair -> 1 class
    assert logits[2] is None          # Carpet -> no sub-head at all
    assert logits[3].shape == (1,)   # Storage -> 1 class


def test_forward_without_gt_uses_predicted_argmax_routing():
    heads = _build_heads()
    x = torch.rand(2, 16)
    out = heads(x, super_category_gt=None)  # inference mode
    assert len(out["object_category_logits"]) == 2
    # just verify it doesn't crash and produces either a tensor or None per sample
    for logit in out["object_category_logits"]:
        assert logit is None or isinstance(logit, torch.Tensor)


def test_style_and_materials_heads_shapes():
    heads = _build_heads()
    x = torch.rand(5, 16)
    out = heads(x, super_category_gt=torch.zeros(5, dtype=torch.long))
    assert out["style_logits"].shape == (5, len(tx.STYLE_CLASSES))
    assert out["materials_primary_logits"].shape == (5, len(tx.MATERIALS))
    assert out["materials_secondary_logits"].shape == (5, len(tx.MATERIALS))


def test_super_category_with_samples_but_no_object_samples_skips_gracefully():
    # e.g. "Window" is active (has samples) but zero valid object_category samples yet
    active_super = ["Bed", "Window"]
    active_object = {"Bed": ["King"]}  # Window deliberately missing -> 0 classes
    heads = MultiTaskHeads(
        input_dim=8, active_super_categories=active_super, active_object_categories=active_object
    )
    assert "Window" not in heads.object_category_heads
    assert "Bed" in heads.object_category_heads


def test_predict_object_category_convenience_method():
    heads = _build_heads()
    x = torch.rand(2, 16)
    out = heads.predict_object_category(x, "Bed")
    assert out.shape == (2, 2)
    out_carpet = heads.predict_object_category(x, "Carpet")
    assert out_carpet is None


def test_gradients_flow_through_correct_subhead_only():
    heads = _build_heads()
    x = torch.rand(2, 16, requires_grad=True)
    gt = torch.tensor([0, 0])  # both Bed
    out = heads(x, super_category_gt=gt)
    loss = sum(l.sum() for l in out["object_category_logits"] if l is not None)
    loss.backward()
    assert heads.object_category_heads["Bed"].weight.grad is not None
    # Chair head should have no grad since no sample routed through it
    assert heads.object_category_heads["Chair"].weight.grad is None


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
