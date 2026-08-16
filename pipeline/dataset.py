"""
dataset.py

PyTorch Dataset that discovers samples across the four dataset/ subfolders
(models/, metadata/, geometry/, renders/), matches them by `id`, validates them
(via label_validation.py + geometry_validation.py), and yields a unified Sample
dict per __getitem__ call.

Robustness contract (per spec section 2 + 16.5):
    - A sample missing renders, missing metadata, or missing geometry (when
      geometry fusion is enabled) is EXCLUDED from training, with a logged
      warning -- never a crash.
    - A sample that fails label_validation or geometry_validation is likewise
      excluded with a logged warning.
    - Corrupt/unopenable images are excluded at the sample level (all-or-nothing
      per sample, since MVCNN-style pooling needs a consistent view count).

Sample dict shape (see design doc section 4):
    {
        "id": str,
        "images": FloatTensor (V, 3, H, W)            V = expected_views_per_sample
        "geometry_vector": FloatTensor (F,) or None     F = len(feature_list)
        "missing_geometry_fields": list[str],
        "labels": {
            "super_category": int,               # index into active_super_categories
            "object_category": int or -1,          # index into that super_category's
                                                     # own sub-head vocabulary; -1 if this
                                                     # super_category has no object_category
                                                     # (e.g. Carpet) or the field is absent
            "style_class": FloatTensor (len(STYLE_CLASSES),)      multi-hot
            "materials_primary": int,
            "materials_secondary": FloatTensor (len(MATERIALS),)  multi-hot
        },
        "sample_weight": float,
        "low_confidence_combo": bool,
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Dataset = object  # fallback base class so this module still imports

from PIL import Image

import taxonomy as tx
import geometry_features as gf
import geometry_validation as gv
import label_validation as lv
from config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class SampleRecord:
    """Lightweight record produced during dataset discovery/indexing (before image loading)."""
    id: str
    model_path: Path
    metadata_path: Path
    geometry_path: Path
    renders_dir: Path
    metadata_json: dict
    geometry_json: dict


def discover_samples(
    root_dir: str | Path,
    models_subdir: str = "models",
    metadata_subdir: str = "metadata",
    geometry_subdir: str = "geometry",
    renders_subdir: str = "renders",
    expected_views: int = 12,
    require_geometry: bool = True,
) -> tuple[list[SampleRecord], list[str]]:
    """
    Scan the four dataset subfolders, match by `id`, and return valid
    SampleRecords plus a list of human-readable warning strings for excluded
    IDs. This function does NOT load images or validate label/geometry content
    -- it only checks structural presence/matching. Content validation happens
    in ArchDataset.__init__ via label_validation.py / geometry_validation.py.

    An id is included only if it has:
        - a model file in models/ (any extension)
        - a metadata/{id}.json file
        - a renders/{id}/ directory with >= expected_views images
        - a geometry/{id}.json file (only required if require_geometry=True,
          i.e. when the experiment config has use_geometry_fusion=True)
    """
    root_dir = Path(root_dir)
    models_dir = root_dir / models_subdir
    metadata_dir = root_dir / metadata_subdir
    geometry_dir = root_dir / geometry_subdir
    renders_dir = root_dir / renders_subdir

    warnings: list[str] = []

    if not models_dir.exists():
        warnings.append(f"models_dir does not exist: {models_dir}")
        return [], warnings

    # id -> model file path (first match wins; warn on duplicate ids across extensions)
    model_ids: dict[str, Path] = {}
    for model_file in models_dir.iterdir():
        if model_file.is_file():
            sample_id = model_file.stem
            if sample_id in model_ids:
                warnings.append(
                    f"Duplicate model id '{sample_id}' "
                    f"({model_ids[sample_id].name} vs {model_file.name}) -- keeping first."
                )
                continue
            model_ids[sample_id] = model_file

    records: list[SampleRecord] = []

    for sample_id, model_path in sorted(model_ids.items()):
        metadata_path = metadata_dir / f"{sample_id}.json"
        geometry_path = geometry_dir / f"{sample_id}.json"
        sample_renders_dir = renders_dir / sample_id

        if not metadata_path.exists():
            warnings.append(f"[{sample_id}] excluded: missing metadata file {metadata_path}")
            continue

        if require_geometry and not geometry_path.exists():
            warnings.append(f"[{sample_id}] excluded: missing geometry file {geometry_path}")
            continue

        if not sample_renders_dir.is_dir():
            warnings.append(f"[{sample_id}] excluded: missing renders directory {sample_renders_dir}")
            continue

        view_files = sorted(
            p for p in sample_renders_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if len(view_files) < expected_views:
            warnings.append(
                f"[{sample_id}] excluded: only {len(view_files)}/{expected_views} "
                f"render views found in {sample_renders_dir}"
            )
            continue

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_json = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            warnings.append(f"[{sample_id}] excluded: corrupt metadata JSON ({e})")
            continue

        geometry_json: dict = {}
        if geometry_path.exists():
            try:
                with open(geometry_path, "r", encoding="utf-8") as f:
                    geometry_json = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                if require_geometry:
                    warnings.append(f"[{sample_id}] excluded: corrupt geometry JSON ({e})")
                    continue
                else:
                    warnings.append(f"[{sample_id}] geometry JSON corrupt, ignored ({e})")
                    geometry_json = {}

        records.append(SampleRecord(
            id=sample_id,
            model_path=model_path,
            metadata_path=metadata_path,
            geometry_path=geometry_path,
            renders_dir=sample_renders_dir,
            metadata_json=metadata_json,
            geometry_json=geometry_json,
        ))

    logger.info(
        f"discover_samples: {len(records)} valid samples found, "
        f"{len(warnings)} excluded/warned."
    )
    return records, warnings


class ActiveTaxonomy:
    """
    Computes which super_categories/object_categories actually have >= 1 sample
    in the CURRENT dataset (per spec section 6: zero-sample categories are
    dynamically excluded from the output heads, not hardcoded/forced).

    This is built once from the full discovered+validated sample set before
    model construction, then passed to models/factory.py.
    """

    def __init__(self, metadata_jsons: list[dict]):
        seen_super: set[str] = set()
        seen_object: dict[str, set[str]] = {}

        for m in metadata_jsons:
            sc = m.get("super_category")
            if not sc or not tx.is_valid_super_category(sc):
                continue
            seen_super.add(sc)
            if tx.has_object_category(sc):
                oc = m.get("object_category")
                if oc and tx.is_valid_object_category(sc, oc):
                    seen_object.setdefault(sc, set()).add(oc)

        # Preserve taxonomy.py's declared order for stability/determinism
        self.active_super_categories: list[str] = [
            sc for sc in tx.get_all_super_categories() if sc in seen_super
        ]
        self.active_object_categories: dict[str, list[str]] = {
            sc: [oc for oc in tx.get_object_categories(sc) if oc in seen_object.get(sc, set())]
            for sc in self.active_super_categories
            if tx.has_object_category(sc)
        }

        if not self.active_super_categories:
            raise ValueError(
                "No valid super_category found across the dataset after validation -- "
                "cannot build a model with zero output classes. Check your dataset/ "
                "folder and metadata files."
            )

        for sc in self.active_super_categories:
            if tx.has_object_category(sc) and not self.active_object_categories.get(sc):
                logger.warning(
                    f"super_category '{sc}' has samples but NO valid object_category "
                    f"samples yet -- its object_category sub-head will have 0 classes "
                    f"and will be skipped at model-build time until data exists."
                )

    def super_category_index(self, super_category: str) -> int:
        return self.active_super_categories.index(super_category)

    def object_category_index(self, super_category: str, object_category: str) -> int:
        return self.active_object_categories[super_category].index(object_category)

    def num_object_categories(self, super_category: str) -> int:
        return len(self.active_object_categories.get(super_category, []))


def _multi_hot(values: list[str], vocabulary: list[str]) -> np.ndarray:
    vec = np.zeros(len(vocabulary), dtype=np.float32)
    for v in values:
        if v in vocabulary:
            vec[vocabulary.index(v)] = 1.0
    return vec


class ArchDataset(Dataset):
    """
    PyTorch Dataset over the 3D architecture/furniture model dataset.

    Construction performs full discovery + validation eagerly (so failures are
    surfaced before training starts, per the Dataset Health Report philosophy),
    then __getitem__ only does image loading + augmentation + feature extraction
    per sample (cheap, repeatable per epoch).
    """

    def __init__(
        self,
        config: PipelineConfig,
        active_taxonomy: Optional[ActiveTaxonomy] = None,
        record_ids: Optional[list[str]] = None,
        transform=None,
        is_training: bool = True,
    ):
        """
        Args:
            config: the full PipelineConfig.
            active_taxonomy: precomputed ActiveTaxonomy. If None, computed fresh
                from this dataset's own discovered+validated samples (typical
                for a first full-dataset pass; for K-fold, pass the taxonomy
                computed ONCE across the full dataset so all folds share the
                same label space).
            record_ids: optional explicit subset of sample ids to include (used
                by K-fold splitting in train.py). If None, uses every valid
                discovered sample.
            transform: an augmentation.MultiViewAugmentation instance (or None
                for no augmentation, e.g. at eval time).
            is_training: whether this split is used for scaler fitting eligibility
                (the geometry StandardScaler must only ever be fit on training data).
        """
        self.config = config
        self.transform = transform
        self.is_training = is_training

        require_geometry = config.model.use_geometry_fusion
        all_records, discovery_warnings = discover_samples(
            root_dir=config.dataset.root_dir,
            models_subdir=config.dataset.models_subdir,
            metadata_subdir=config.dataset.metadata_subdir,
            geometry_subdir=config.dataset.geometry_subdir,
            renders_subdir=config.dataset.renders_subdir,
            expected_views=config.dataset.expected_views_per_sample,
            require_geometry=require_geometry,
        )
        self.discovery_warnings = discovery_warnings

        # Content validation pass (label + geometry)
        valid_records: list[SampleRecord] = []
        self.validation_warnings: list[str] = []
        for rec in all_records:
            label_result = lv.validate_label(rec.metadata_json, model_id=rec.id)
            if not label_result.is_valid:
                self.validation_warnings.append(
                    f"[{rec.id}] excluded: label validation failed: {label_result.errors}"
                )
                continue

            if require_geometry:
                geo_result = gv.validate_geometry(rec.geometry_json, model_id=rec.id)
                if not geo_result.is_valid:
                    self.validation_warnings.append(
                        f"[{rec.id}] excluded: geometry validation failed: {geo_result.errors}"
                    )
                    continue

            valid_records.append(rec)

        for w in self.validation_warnings:
            logger.warning(w)

        if record_ids is not None:
            wanted = set(record_ids)
            valid_records = [r for r in valid_records if r.id in wanted]

        self.records: list[SampleRecord] = valid_records

        if active_taxonomy is None:
            active_taxonomy = ActiveTaxonomy([r.metadata_json for r in self.records])
        self.active_taxonomy = active_taxonomy

        self.geometry_feature_list = (
            config.resolved_geometry_feature_list() if require_geometry else []
        )

        # Geometry scaler: fit externally (by train.py, on the training fold only)
        # and injected via set_geometry_scaler(). Left as None until then.
        self._geometry_scaler = None

        # Low-confidence combos (spec 8.4): computed once over ALL valid records
        # visible to this dataset instance (typically called with the full
        # training-fold record set).
        self._low_confidence_combos = self._compute_low_confidence_combos(
            min_samples=config.loss.min_sample_confidence_threshold
        )

        logger.info(
            f"ArchDataset ready: {len(self.records)} usable samples "
            f"(of {len(all_records)} structurally discovered), "
            f"{len(self.active_taxonomy.active_super_categories)} active super_categories."
        )

    def _compute_low_confidence_combos(self, min_samples: int) -> set[tuple[str, str]]:
        from collections import Counter
        combo_counts: Counter = Counter()
        for rec in self.records:
            sc = rec.metadata_json.get("super_category")
            oc = rec.metadata_json.get("object_category", "")
            combo_counts[(sc, oc)] += 1
        return {combo for combo, count in combo_counts.items() if count < min_samples}

    def set_geometry_scaler(self, scaler) -> None:
        """Injected by train.py after fitting a StandardScaler on the training fold only."""
        self._geometry_scaler = scaler

    def __len__(self) -> int:
        return len(self.records)

    def _load_views(self, renders_dir: Path, image_size: int) -> np.ndarray:
        view_files = sorted(
            p for p in renders_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )[: self.config.dataset.expected_views_per_sample]

        views = []
        for vf in view_files:
            img = Image.open(vf).convert("RGB").resize((image_size, image_size))
            views.append(np.asarray(img, dtype=np.float32) / 255.0)
        return np.stack(views, axis=0)  # (V, H, W, 3)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image_size = self.config.dataset.image_size

        views_np = self._load_views(rec.renders_dir, image_size)  # (V, H, W, 3)

        if self.transform is not None:
            views_np = self.transform(views_np)  # augmentation.py applies shared params across V

        if _TORCH_AVAILABLE:
            images = torch.from_numpy(views_np).permute(0, 3, 1, 2).float()  # (V, 3, H, W)
        else:
            images = views_np

        geometry_vector = None
        missing_geometry_fields: list[str] = []
        if self.config.model.use_geometry_fusion:
            raw_vector, missing_geometry_fields = gf.extract_geometry_vector(
                rec.geometry_json, self.geometry_feature_list
            )
            if self._geometry_scaler is not None:
                raw_vector = self._geometry_scaler.transform(raw_vector.reshape(1, -1))[0]
            geometry_vector = torch.from_numpy(raw_vector.astype(np.float32)) if _TORCH_AVAILABLE else raw_vector

        labels = self._build_labels(rec.metadata_json)

        sample_weight = 1.0
        if self.config.loss.use_confidence_score_sample_weighting:
            conf = rec.metadata_json.get("confidence_scores")
            if conf:
                # average across available confidence fields for a single scalar weight
                vals = [v for v in conf.values() if isinstance(v, (int, float))]
                if vals:
                    sample_weight = float(sum(vals) / len(vals))

        sc = rec.metadata_json.get("super_category")
        oc = rec.metadata_json.get("object_category", "")
        low_confidence_combo = (sc, oc) in self._low_confidence_combos

        return {
            "id": rec.id,
            "images": images,
            "geometry_vector": geometry_vector,
            "missing_geometry_fields": missing_geometry_fields,
            "labels": labels,
            "sample_weight": sample_weight,
            "low_confidence_combo": low_confidence_combo,
        }

    def _build_labels(self, metadata_json: dict) -> dict:
        sc = metadata_json.get("super_category")
        super_category_idx = self.active_taxonomy.super_category_index(sc)

        object_category_idx = -1
        if tx.has_object_category(sc):
            oc = metadata_json.get("object_category")
            if oc and oc in self.active_taxonomy.active_object_categories.get(sc, []):
                object_category_idx = self.active_taxonomy.object_category_index(sc, oc)

        style_values = metadata_json.get("style_class", []) or []
        style_vec = _multi_hot(style_values, tx.STYLE_CLASSES)

        materials = metadata_json.get("materials", {}) or {}
        primary = materials.get("primary")
        materials_primary_idx = tx.MATERIALS.index(primary) if primary in tx.MATERIALS else -1

        secondary_values = materials.get("secondary", []) or []
        materials_secondary_vec = _multi_hot(secondary_values, tx.MATERIALS)

        if _TORCH_AVAILABLE:
            return {
                "super_category": super_category_idx,
                "object_category": object_category_idx,
                "style_class": torch.from_numpy(style_vec),
                "materials_primary": materials_primary_idx,
                "materials_secondary": torch.from_numpy(materials_secondary_vec),
            }
        return {
            "super_category": super_category_idx,
            "object_category": object_category_idx,
            "style_class": style_vec,
            "materials_primary": materials_primary_idx,
            "materials_secondary": materials_secondary_vec,
        }
