"""
Unit tests for taxonomy.py
Run with: python -m pytest tests/test_taxonomy.py -v
(or just: python tests/test_taxonomy.py  -- it also runs standalone without pytest)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import taxonomy as tx


def test_num_super_categories_is_10():
    # 11 original - Shelf - Closet + Storage(merged) = 10
    assert tx.get_num_super_categories() == 10, (
        f"Expected 10 super_categories after Shelf+Closet merge, got {tx.get_num_super_categories()}"
    )


def test_shelf_and_closet_do_not_exist_standalone():
    assert not tx.is_valid_super_category("Shelf")
    assert not tx.is_valid_super_category("Closet")


def test_storage_merged_category_exists_with_all_subcategories():
    assert tx.is_valid_super_category("Storage")
    subs = tx.get_object_categories("Storage")
    expected = {
        "Standalone Shelves", "Wall Mounted Shelves", "Built In Shelves",
        "Walk In Closet", "Reach In Closet", "Wardrobe",
    }
    assert set(subs) == expected, f"Storage subcategories mismatch: {subs}"


def test_carpet_has_no_object_category():
    assert tx.is_valid_super_category("Carpet")
    assert tx.has_object_category("Carpet") is False
    assert tx.get_object_categories("Carpet") == []
    assert tx.get_num_object_categories("Carpet") == 0


def test_normal_category_has_object_category():
    assert tx.has_object_category("Bed") is True
    assert tx.get_num_object_categories("Bed") == 8


def test_is_valid_object_category_respects_hierarchy():
    # "King" is a valid Bed sub-category but not a valid Chair sub-category
    assert tx.is_valid_object_category("Bed", "King") is True
    assert tx.is_valid_object_category("Chair", "King") is False


def test_is_valid_object_category_false_for_carpet_regardless_of_value():
    assert tx.is_valid_object_category("Carpet", "anything") is False


def test_unknown_super_category_is_safe():
    assert tx.is_valid_super_category("NotARealCategory") is False
    assert tx.get_object_categories("NotARealCategory") == []
    assert tx.get_num_object_categories("NotARealCategory") == 0
    assert tx.has_object_category("NotARealCategory") is False
    assert tx.is_valid_object_category("NotARealCategory", "King") is False


def test_style_and_material_lists_match_spec():
    assert tx.STYLE_CLASSES == [
        "Minimalism", "Bauhaus", "Post Modern",
        "French Classic", "Industrial", "Tropical",
    ]
    assert tx.MATERIALS == [
        "Wood", "Metal", "Glass", "Concrete", "Stone", "Fabric", "Plastic", "3D Printed",
    ]
    assert tx.is_valid_style("Bauhaus")
    assert not tx.is_valid_style("Cyberpunk")
    assert tx.is_valid_material("Wood")
    assert not tx.is_valid_material("Unobtainium")


def test_all_super_categories_list():
    names = tx.get_all_super_categories()
    assert len(names) == 10
    assert "Storage" in names
    assert "Shelf" not in names
    assert "Closet" not in names
    assert "Carpet" in names


def _run_all():
    """Standalone runner (no pytest dependency needed)."""
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
