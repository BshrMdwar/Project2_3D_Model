import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import label_validation as lv


GOOD_BED_LABEL = {
    "id": "bed_0000002",
    "super_category": "Bed",
    "object_category": "King",
    "style_class": ["Minimalism"],
    "materials": {"primary": "Fabric", "secondary": ["Fabric"]},
}

GOOD_CARPET_LABEL = {
    "id": "carpet_0000001",
    "super_category": "Carpet",
    "style_class": ["Industrial"],
    "materials": {"primary": "Wood", "secondary": []},
}


def test_valid_bed_label_passes():
    result = lv.validate_label(GOOD_BED_LABEL)
    assert result.is_valid
    assert result.errors == []


def test_valid_carpet_label_with_no_object_category_passes():
    result = lv.validate_label(GOOD_CARPET_LABEL)
    assert result.is_valid
    assert result.errors == []


def test_carpet_with_object_category_is_warning_not_error():
    bad = {**GOOD_CARPET_LABEL, "object_category": "Cut Pile"}
    result = lv.validate_label(bad)
    assert result.is_valid  # still valid, just a warning
    assert len(result.warnings) >= 1


def test_bed_missing_required_object_category_fails():
    bad = {**GOOD_BED_LABEL}
    del bad["object_category"]
    result = lv.validate_label(bad)
    assert not result.is_valid


def test_invalid_object_category_for_super_category_fails():
    bad = {**GOOD_BED_LABEL, "object_category": "Armchair"}  # Armchair is a Chair subtype, not Bed
    result = lv.validate_label(bad)
    assert not result.is_valid


def test_unknown_super_category_fails():
    bad = {**GOOD_BED_LABEL, "super_category": "Spaceship"}
    result = lv.validate_label(bad)
    assert not result.is_valid


def test_shelf_and_closet_as_standalone_super_category_now_invalid():
    # Because Shelf/Closet were merged into "Storage" per user decision
    bad = {**GOOD_BED_LABEL, "super_category": "Shelf", "object_category": "Standalone Shelves"}
    result = lv.validate_label(bad)
    assert not result.is_valid


def test_storage_merged_category_valid():
    ok = {
        "id": "storage_001",
        "super_category": "Storage",
        "object_category": "Walk In Closet",
        "style_class": ["Bauhaus"],
        "materials": {"primary": "Wood", "secondary": []},
    }
    result = lv.validate_label(ok)
    assert result.is_valid


def test_invalid_style_class_fails():
    bad = {**GOOD_BED_LABEL, "style_class": ["Cyberpunk"]}
    result = lv.validate_label(bad)
    assert not result.is_valid


def test_invalid_material_fails():
    bad = {**GOOD_BED_LABEL, "materials": {"primary": "Unobtainium", "secondary": []}}
    result = lv.validate_label(bad)
    assert not result.is_valid


def test_missing_confidence_scores_is_fine_no_warning():
    # confidence_scores is Optional by schema (only present for VLM records)
    result = lv.validate_label(GOOD_BED_LABEL)
    assert result.is_valid
    assert not any("confidence" in w for w in result.warnings)


def test_corrupt_non_dict_input_handled_safely():
    result = lv.validate_label("not a dict")
    assert not result.is_valid
    assert len(result.errors) >= 1


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
