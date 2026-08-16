import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

import config as cfg
import dataset as ds
import taxonomy as tx

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mock_dataset"


def _make_config(use_geometry=False):
    pc = cfg.PipelineConfig()
    pc.dataset.root_dir = str(FIXTURE_ROOT)
    pc.dataset.expected_views_per_sample = 12
    pc.model.use_geometry_fusion = use_geometry
    pc.model.geometry_features = "all"
    return pc


def test_discover_samples_excludes_broken_ones():
    pc = _make_config(use_geometry=False)
    records, warnings = ds.discover_samples(
        root_dir=pc.dataset.root_dir,
        expected_views=pc.dataset.expected_views_per_sample,
        require_geometry=False,
    )
    ids = {r.id for r in records}
    # good ones present
    assert "bed_0000001" in ids
    assert "chair_0000001" in ids
    assert "carpet_0000001" in ids
    assert "storage_0000001" in ids
    # broken ones excluded
    assert "chair_0000002_broken_renders" not in ids  # incomplete renders
    assert "chair_0000003_no_metadata" not in ids       # missing metadata.json
    assert any("chair_0000002_broken_renders" in w for w in warnings)
    assert any("chair_0000003_no_metadata" in w for w in warnings)


def test_full_dataset_excludes_bad_label_via_validation():
    pc = _make_config(use_geometry=False)
    dset = ds.ArchDataset(pc)
    ids = {r.id for r in dset.records}
    assert "bed_0000003_bad_label" not in ids
    assert any("bed_0000003_bad_label" in w for w in dset.validation_warnings)
    # good samples remain
    assert "bed_0000001" in ids
    assert "carpet_0000001" in ids


def test_active_taxonomy_excludes_zero_sample_categories():
    pc = _make_config(use_geometry=False)
    dset = ds.ArchDataset(pc)
    active = dset.active_taxonomy.active_super_categories
    # Only categories present in the mock dataset should appear
    assert set(active) == {"Bed", "Chair", "Carpet", "Storage"}
    # e.g. Window/Door/Table/Sofa/Light/Cabinet have 0 samples in this mock set
    assert "Window" not in active
    assert "Sofa" not in active


def test_carpet_has_no_object_category_head_entries():
    pc = _make_config(use_geometry=False)
    dset = ds.ArchDataset(pc)
    assert "Carpet" not in dset.active_taxonomy.active_object_categories


def test_getitem_returns_correct_shapes_no_geometry():
    pc = _make_config(use_geometry=False)
    dset = ds.ArchDataset(pc)
    sample = dset[0]
    assert sample["images"].shape == (12, 3, 224, 224)
    assert sample["geometry_vector"] is None
    assert isinstance(sample["labels"]["super_category"], int)
    assert sample["labels"]["style_class"].shape == (len(tx.STYLE_CLASSES),)
    assert sample["labels"]["materials_secondary"].shape == (len(tx.MATERIALS),)


def test_getitem_with_geometry_fusion_enabled():
    pc = _make_config(use_geometry=True)
    dset = ds.ArchDataset(pc)
    sample = dset[0]
    assert sample["geometry_vector"] is not None
    assert sample["geometry_vector"].shape == (len(cfg.DEFAULT_GEOMETRY_FEATURES),)


def test_carpet_object_category_label_is_minus_one():
    pc = _make_config(use_geometry=False)
    dset = ds.ArchDataset(pc)
    carpet_records = [i for i, r in enumerate(dset.records) if r.id == "carpet_0000001"]
    assert len(carpet_records) == 1
    sample = dset[carpet_records[0]]
    assert sample["labels"]["object_category"] == -1


def test_bed_object_category_label_is_valid_index():
    pc = _make_config(use_geometry=False)
    dset = ds.ArchDataset(pc)
    bed_idx = [i for i, r in enumerate(dset.records) if r.id == "bed_0000001"][0]
    sample = dset[bed_idx]
    assert sample["labels"]["object_category"] >= 0
    assert sample["labels"]["object_category"] < dset.active_taxonomy.num_object_categories("Bed")


def test_low_confidence_combo_flagging():
    pc = _make_config(use_geometry=False)
    pc.loss.min_sample_confidence_threshold = 2  # low threshold so our tiny mock triggers it
    dset = ds.ArchDataset(pc)
    # every combo in our mock dataset has only 1 sample except none -- with threshold=2,
    # ALL combos should be flagged low-confidence
    sample = dset[0]
    assert sample["low_confidence_combo"] is True


def test_confidence_score_sample_weighting_toggle():
    pc = _make_config(use_geometry=False)
    pc.loss.use_confidence_score_sample_weighting = True
    dset = ds.ArchDataset(pc)
    bed2_idx = [i for i, r in enumerate(dset.records) if r.id == "bed_0000002"][0]
    sample = dset[bed2_idx]
    # bed_0000002 has confidence_scores {"super_category": 0.9, "object_category": 0.7} -> avg 0.8
    assert abs(sample["sample_weight"] - 0.8) < 1e-6

    bed1_idx = [i for i, r in enumerate(dset.records) if r.id == "bed_0000001"][0]
    sample1 = dset[bed1_idx]
    # bed_0000001 is Human-annotated, no confidence_scores -> default weight 1.0
    assert sample1["sample_weight"] == 1.0


def test_missing_geometry_fields_reported_when_partial_schema():
    pc = _make_config(use_geometry=True)
    pc.model.geometry_features = ["dimensions.aspect_hw", "nonexistent.field.path"]
    dset = ds.ArchDataset(pc)
    sample = dset[0]
    assert "nonexistent.field.path" in sample["missing_geometry_fields"]


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
