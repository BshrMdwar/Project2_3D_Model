import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import geometry_features as gf
import geometry_validation as gv


SAMPLE_GEOMETRY = {
    "model_id": "bed_0000001",
    "dimensions": {
        "width": 2.9508, "depth": 2.0895, "height": 1.1437,
        "aspect_hw": 0.3876, "aspect_hd": 0.5474, "aspect_wd": 1.4122,
    },
    "mesh_density": {
        "vertices": 378576, "faces": 371111,
        "log_vertices": 12.844175, "log_faces": 12.824259,
    },
    "geometry": {
        "surface_area": 25.611, "volume_estimate": 7.051681,
        "occupancy_ratio": 1.0,
    },
    "shape_descriptors": {
        "compactness": 2.98705, "elongation": 2.5801, "symmetry_axis": "Z",
    },
    "materials_and_textures": {
        "materials_breakdown": [
            {"material_name": "Fabric_Main", "surface_area_share": 0.62},
        ],
    },
}


def test_extract_basic_features_in_order():
    feature_list = ["dimensions.aspect_hw", "geometry.occupancy_ratio", "shape_descriptors.compactness"]
    vector, missing = gf.extract_geometry_vector(SAMPLE_GEOMETRY, feature_list)
    assert missing == []
    assert vector.shape == (3,)
    assert abs(vector[0] - 0.3876) < 1e-6
    assert abs(vector[1] - 1.0) < 1e-6
    assert abs(vector[2] - 2.98705) < 1e-4


def test_missing_field_defaults_to_zero_and_is_reported():
    feature_list = ["dimensions.aspect_hw", "nonexistent.field", "another.missing.one"]
    vector, missing = gf.extract_geometry_vector(SAMPLE_GEOMETRY, feature_list)
    assert vector[1] == 0.0
    assert vector[2] == 0.0
    assert set(missing) == {"nonexistent.field", "another.missing.one"}


def test_non_numeric_field_treated_as_missing_not_crash():
    feature_list = ["shape_descriptors.symmetry_axis"]  # this is a string "Z", not numeric
    vector, missing = gf.extract_geometry_vector(SAMPLE_GEOMETRY, feature_list)
    assert vector[0] == 0.0
    assert missing == ["shape_descriptors.symmetry_axis"]


def test_empty_dict_geometry_returns_all_missing_no_crash():
    feature_list = ["dimensions.aspect_hw", "geometry.occupancy_ratio"]
    vector, missing = gf.extract_geometry_vector({}, feature_list)
    assert np.all(vector == 0.0)
    assert len(missing) == 2


def test_materials_breakdown_raises_at_config_validation_time():
    bad_list = ["dimensions.aspect_hw", "materials_and_textures.materials_breakdown"]
    try:
        gf.validate_feature_list(bad_list)
        assert False, "expected UnsupportedGeometryFeatureError"
    except gf.UnsupportedGeometryFeatureError:
        pass


def test_valid_feature_list_passes_validation():
    good_list = ["dimensions.aspect_hw", "geometry.occupancy_ratio", "shape_descriptors.compactness"]
    gf.validate_feature_list(good_list)  # should not raise


def test_geometry_validation_passes_for_good_sample():
    result = gv.validate_geometry(SAMPLE_GEOMETRY)
    assert result.is_valid
    assert result.errors == []


def test_geometry_validation_fails_on_non_positive_dimension():
    bad = {**SAMPLE_GEOMETRY, "dimensions": {**SAMPLE_GEOMETRY["dimensions"], "width": -1.0}}
    result = gv.validate_geometry(bad, model_id="bad_sample")
    assert not result.is_valid
    assert any("width" in e for e in result.errors)


def test_geometry_validation_fails_on_nan():
    bad = {**SAMPLE_GEOMETRY, "geometry": {**SAMPLE_GEOMETRY["geometry"], "surface_area": float("nan")}}
    result = gv.validate_geometry(bad, model_id="nan_sample")
    assert not result.is_valid
    assert any("NaN" in e or "Infinity" in e for e in result.errors)


def test_geometry_validation_missing_required_field_is_warning_not_error():
    minimal = {"dimensions": {"width": 1.0, "depth": 1.0, "height": 1.0},
               "mesh_density": {"vertices": 10, "faces": 10},
               "geometry": {"surface_area": 1.0}}
    result = gv.validate_geometry(minimal)
    assert result.is_valid  # nothing invalid, just sparse


def test_reference_stats_and_outlier_detection():
    stats = gv.compute_reference_stats([SAMPLE_GEOMETRY, SAMPLE_GEOMETRY, SAMPLE_GEOMETRY])
    outlier = {**SAMPLE_GEOMETRY, "geometry": {**SAMPLE_GEOMETRY["geometry"], "surface_area": 999999.0}}
    result = gv.validate_geometry(outlier, reference_stats=stats, outlier_z_threshold=3.0)
    # std is 0 across identical samples, so outlier logic should skip that field (std<=eps) safely, no crash
    assert isinstance(result.is_valid, bool)


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
