import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import config as cfg

REPO_ROOT = Path(__file__).parent.parent
EXP_DIR = REPO_ROOT / "configs" / "experiments"
BASE_DIR = REPO_ROOT / "configs" / "base"


def test_load_baseline_experiment():
    pc = cfg.load_experiment_config(EXP_DIR / "exp01_baseline.yaml")
    assert pc.experiment_name == "exp01_baseline"
    assert pc.model.use_geometry_fusion is False
    # inherited from base, untouched
    assert pc.training.batch_size == 16
    assert pc.dataset.k_folds == 5


def test_load_full_geometry_experiment():
    pc = cfg.load_experiment_config(EXP_DIR / "exp02_full_geometry.yaml")
    assert pc.model.use_geometry_fusion is True
    assert pc.model.geometry_features == "all"
    feature_list = pc.resolved_geometry_feature_list()
    assert len(feature_list) == len(cfg.DEFAULT_GEOMETRY_FEATURES)
    assert "dimensions.aspect_hw" in feature_list


def test_load_shape_only_subset_experiment():
    pc = cfg.load_experiment_config(EXP_DIR / "exp03_geometry_shape_only.yaml")
    feature_list = pc.resolved_geometry_feature_list()
    assert feature_list == [
        "dimensions.aspect_hw", "dimensions.aspect_hd", "dimensions.aspect_wd",
        "shape_descriptors.compactness", "shape_descriptors.elongation",
        "geometry.occupancy_ratio",
    ]
    # unrelated base sections still inherited
    assert pc.training.learning_rate == 1e-4


def test_load_geometry_only_ablation_freezes_backbone():
    pc = cfg.load_experiment_config(EXP_DIR / "exp06_geometry_only.yaml")
    assert pc.model.freeze_visual_backbone_entirely is True
    assert pc.model.use_geometry_fusion is True


def test_load_attention_pooling_experiment_overrides_only_pooling():
    pc = cfg.load_experiment_config(EXP_DIR / "exp05_attention_pooling.yaml")
    assert pc.model.pooling_method == "attention"
    assert pc.model.use_geometry_fusion is True  # inherited from its own override, not base


def test_unknown_config_field_raises_clear_error(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        bad_yaml = Path(d) / "bad_exp.yaml"
        bad_yaml.write_text("extends: base\nmodel:\n  use_geoemtry_fusion: true\n")  # typo!
        try:
            cfg.load_experiment_config(bad_yaml, base_dir=BASE_DIR)
            assert False, "expected ValueError for unknown field (typo)"
        except ValueError as e:
            assert "Unknown field" in str(e)


def test_experiment_name_defaults_to_filename_stem(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        yaml_path = Path(d) / "my_custom_experiment.yaml"
        yaml_path.write_text("extends: base\n")
        pc = cfg.load_experiment_config(yaml_path, base_dir=BASE_DIR)
        assert pc.experiment_name == "my_custom_experiment"


def test_materials_breakdown_in_config_fails_fast():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        yaml_path = Path(d) / "bad_geo.yaml"
        yaml_path.write_text(
            "extends: base\n"
            "model:\n"
            "  use_geometry_fusion: true\n"
            "  geometry_features:\n"
            "    - \"materials_and_textures.materials_breakdown\"\n"
        )
        pc = cfg.load_experiment_config(yaml_path, base_dir=BASE_DIR)
        try:
            pc.resolved_geometry_feature_list()
            assert False, "expected UnsupportedGeometryFeatureError"
        except Exception as e:
            assert "variable-length" in str(e)


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
