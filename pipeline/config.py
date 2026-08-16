"""
config.py

Loads and merges YAML configs (configs/base/*.yaml + configs/experiments/expXX.yaml)
into a single typed PipelineConfig dataclass. This is the single object passed
through the entire pipeline (dataset.py, models/factory.py, train.py, etc.).

Design rationale: a typed dataclass instead of a raw dict catches config typos
(e.g. `use_geoemtry_fusion`) at load time via explicit field access, rather than
silently producing a None/KeyError deep into a training run.

Experiment YAML files use `extends: base` to inherit everything from
configs/base/*.yaml and override only the fields that differ. See
configs/experiments/*.yaml for real examples.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml

import geometry_features as gf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-configs (mirror the configs/base/*.yaml file split)
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    root_dir: str = "dataset"
    models_subdir: str = "models"
    metadata_subdir: str = "metadata"
    geometry_subdir: str = "geometry"
    renders_subdir: str = "renders"
    expected_views_per_sample: int = 12
    k_folds: int = 5
    image_size: int = 224


@dataclass
class TrainingConfig:
    batch_size: int = 16
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    precision: str = "32"  # "32", "16-mixed", "bf16-mixed" (passed to Lightning Trainer)
    early_stopping_patience: int = 15
    accelerator: str = "auto"  # "auto", "cpu", "gpu" -- Lightning resolves this per-machine
    monitor_metric: str = "val_super_category_f1"
    monitor_mode: str = "max"


@dataclass
class ModelConfig:
    backbone_name: str = "dinov2_vits14"  # see models/backbones.py BACKBONE_REGISTRY
    freeze_backbone_except_last_n_layers: int = 2
    pooling_method: str = "mean"  # "mean", "max", "attention"
    use_geometry_fusion: bool = False
    # "all" (use every entry in DEFAULT_GEOMETRY_FEATURES) or an explicit list of
    # dotted paths (see geometry_features.py). This is the single place that
    # defines which geometry fields become model inputs.
    geometry_features: Any = field(default_factory=lambda: "all")
    freeze_visual_backbone_entirely: bool = False  # geometry-only ablation (14.4 case 3)


@dataclass
class AugmentationConfig:
    use_background_synthesis: bool = True
    use_color_jitter: bool = True
    color_jitter_strength: float = 0.3
    use_random_scale: bool = True
    scale_range: tuple = (0.85, 1.15)
    use_horizontal_flip: bool = True
    horizontal_flip_prob: float = 0.5
    use_rotation: bool = True
    max_rotation_degrees: float = 15.0


@dataclass
class LossConfig:
    use_class_weighted_loss: bool = True
    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    use_balanced_batch_sampling: bool = False
    min_sample_confidence_threshold: int = 10
    use_confidence_score_sample_weighting: bool = False  # optional, per agreed suggestion #3


@dataclass
class SystemConfig:
    seed: int = 42
    deterministic_cudnn: bool = True
    use_feature_cache: bool = True
    feature_cache_dir: str = "cache/embeddings"
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"


@dataclass
class PipelineConfig:
    experiment_name: str = "unnamed_experiment"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def resolved_geometry_feature_list(self) -> list[str]:
        """
        Resolve model.geometry_features ("all" or explicit list) into a concrete
        list of dotted paths, validating it eagerly against known-unsupported
        variable-length fields (fail-fast, per geometry_features.py).
        """
        if self.model.geometry_features == "all":
            feature_list = list(DEFAULT_GEOMETRY_FEATURES)
        elif isinstance(self.model.geometry_features, list):
            feature_list = list(self.model.geometry_features)
        else:
            raise ValueError(
                f"model.geometry_features must be 'all' or a list of dotted paths, "
                f"got {self.model.geometry_features!r}"
            )
        gf.validate_feature_list(feature_list)
        return feature_list

    def to_dict(self) -> dict:
        return asdict(self)


# Default full geometry feature set used when model.geometry_features == "all".
# This is the ONE place to add/remove a geometry feature globally -- editing
# this list (or an experiment's explicit override list) is the only code change
# needed to add/drop a geometry feature from training, per the spec's requirement.
DEFAULT_GEOMETRY_FEATURES: list[str] = [
    "dimensions.aspect_hw",
    "dimensions.aspect_hd",
    "dimensions.aspect_wd",
    "mesh_density.log_vertices",
    "mesh_density.log_faces",
    "mesh_density.faces_per_volume",
    "mesh_density.faces_per_area",
    "geometry.occupancy_ratio",
    "shape_descriptors.compactness",
    "shape_descriptors.elongation",
    "structure.connected_components",
    "structure.objects_count",
    "materials_and_textures.avg_roughness",
    "materials_and_textures.avg_metallic",
    "physics_proxy.stability_score",
]


# ---------------------------------------------------------------------------
# YAML loading + base/experiment merge
# ---------------------------------------------------------------------------

_SUBCONFIG_MAP = {
    "dataset": DatasetConfig,
    "training": TrainingConfig,
    "model": ModelConfig,
    "augmentation": AugmentationConfig,
    "loss": LossConfig,
    "system": SystemConfig,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_base_config(base_dir: Path) -> dict:
    """Load and merge every configs/base/*.yaml file into one dict, keyed by section name."""
    merged: dict = {}
    for section_name in _SUBCONFIG_MAP:
        section_path = base_dir / f"{section_name}.yaml"
        section_data = _load_yaml(section_path)
        if section_data:
            merged[section_name] = section_data
    return merged


def load_experiment_config(
    experiment_yaml_path: str | Path,
    base_dir: Optional[str | Path] = None,
) -> PipelineConfig:
    """
    Load an experiment YAML file, merge it on top of configs/base/*.yaml (if
    `extends: base` is set, which is the expected convention), and return a
    fully-typed PipelineConfig.

    Args:
        experiment_yaml_path: path to e.g. configs/experiments/exp02_full_geometry.yaml
        base_dir: path to the configs/base/ directory. If not given, inferred as
            "<experiment file's grandparent>/base" (i.e. assumes the standard
            configs/base/ + configs/experiments/ layout).
    """
    experiment_yaml_path = Path(experiment_yaml_path)
    experiment_data = _load_yaml(experiment_yaml_path)

    if base_dir is None:
        base_dir = experiment_yaml_path.parent.parent / "base"
    base_dir = Path(base_dir)

    merged_dict: dict = {}
    if experiment_data.get("extends") == "base":
        merged_dict = load_base_config(base_dir)
    elif "extends" in experiment_data:
        logger.warning(
            f"Unrecognized 'extends' value {experiment_data['extends']!r} in "
            f"{experiment_yaml_path} -- expected 'base'. Proceeding without a base merge."
        )

    experiment_overrides = {k: v for k, v in experiment_data.items() if k != "extends"}
    merged_dict = _deep_merge(merged_dict, experiment_overrides)

    # experiment_name defaults to the yaml filename stem if not explicitly set
    merged_dict.setdefault("experiment_name", experiment_yaml_path.stem)

    return _dict_to_pipeline_config(merged_dict)


def _dict_to_pipeline_config(merged_dict: dict) -> PipelineConfig:
    kwargs: dict = {}
    for section_name, dataclass_type in _SUBCONFIG_MAP.items():
        section_data = merged_dict.get(section_name, {})
        # Only pass known fields -- unknown keys raise clearly rather than
        # silently vanishing, catching config typos immediately.
        known_fields = {f.name for f in dataclass_type.__dataclass_fields__.values()}
        unknown = set(section_data.keys()) - known_fields
        if unknown:
            raise ValueError(
                f"Unknown field(s) {unknown} in config section '{section_name}'. "
                f"Valid fields are: {sorted(known_fields)}. Check for a typo."
            )
        kwargs[section_name] = dataclass_type(**section_data)

    return PipelineConfig(
        experiment_name=merged_dict.get("experiment_name", "unnamed_experiment"),
        **kwargs,
    )
