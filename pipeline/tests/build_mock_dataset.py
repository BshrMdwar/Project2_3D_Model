"""
build_mock_dataset.py

Generates a tiny synthetic dataset (a handful of samples across a few
super_categories, including one Carpet sample with no object_category, and one
deliberately broken sample) under tests/fixtures/mock_dataset/, so dataset.py
can be exercised end-to-end without needing the real (large, external) dataset.

Run once: python tests/build_mock_dataset.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mock_dataset"


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_dummy_model_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"FAKE_MODEL_BINARY_DATA")


def _write_renders(dir_path: Path, num_views: int = 12, size: int = 64):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(1, num_views + 1):
        arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(dir_path / f"view_{i:02d}.png")


SAMPLES = [
    # Good Bed sample
    dict(
        id="bed_0000001",
        metadata={
            "id": "bed_0000001", "super_category": "Bed", "object_category": "King",
            "style_class": ["Minimalism"],
            "materials": {"primary": "Fabric", "secondary": ["Wood"]},
            "source": {"annotation_method": "Human"},
        },
        geometry={
            "model_id": "bed_0000001",
            "dimensions": {"width": 2.95, "depth": 2.08, "height": 1.14,
                           "aspect_hw": 0.38, "aspect_hd": 0.54, "aspect_wd": 1.41},
            "mesh_density": {"vertices": 378576, "faces": 371111,
                             "log_vertices": 12.84, "log_faces": 12.82,
                             "faces_per_volume": 52627.3, "faces_per_area": 14490.2},
            "geometry": {"surface_area": 25.6, "volume_estimate": 7.05, "occupancy_ratio": 1.0},
            "shape_descriptors": {"compactness": 2.98, "elongation": 2.58, "symmetry_axis": "Z"},
            "structure": {"connected_components": 72, "objects_count": 15},
            "materials_and_textures": {"avg_roughness": 0.51, "avg_metallic": 0.54},
            "physics_proxy": {"stability_score": 1.0},
        },
        num_views=12,
    ),
    # Good Bed sample, different style/subtype (Queen)
    dict(
        id="bed_0000002",
        metadata={
            "id": "bed_0000002", "super_category": "Bed", "object_category": "Queen",
            "style_class": ["Bauhaus", "Industrial"],
            "materials": {"primary": "Wood", "secondary": []},
            "source": {"annotation_method": "VLM"},
            "confidence_scores": {"super_category": 0.9, "object_category": 0.7},
        },
        geometry={
            "model_id": "bed_0000002",
            "dimensions": {"width": 2.0, "depth": 1.8, "height": 1.0,
                           "aspect_hw": 0.5, "aspect_hd": 0.55, "aspect_wd": 1.1},
            "mesh_density": {"vertices": 50000, "faces": 48000,
                             "log_vertices": 10.8, "log_faces": 10.7,
                             "faces_per_volume": 1000.0, "faces_per_area": 500.0},
            "geometry": {"surface_area": 10.0, "volume_estimate": 3.5, "occupancy_ratio": 0.8},
            "shape_descriptors": {"compactness": 1.5, "elongation": 1.2, "symmetry_axis": "X"},
            "structure": {"connected_components": 10, "objects_count": 3},
            "materials_and_textures": {"avg_roughness": 0.6, "avg_metallic": 0.1},
            "physics_proxy": {"stability_score": 0.95},
        },
        num_views=12,
    ),
    # Good Chair sample
    dict(
        id="chair_0000001",
        metadata={
            "id": "chair_0000001", "super_category": "Chair", "object_category": "Armchair",
            "style_class": ["Post Modern"],
            "materials": {"primary": "Fabric", "secondary": ["Metal"]},
            "source": {"annotation_method": "Human"},
        },
        geometry={
            "model_id": "chair_0000001",
            "dimensions": {"width": 0.8, "depth": 0.8, "height": 1.0,
                           "aspect_hw": 1.25, "aspect_hd": 1.25, "aspect_wd": 1.0},
            "mesh_density": {"vertices": 20000, "faces": 19000,
                             "log_vertices": 9.9, "log_faces": 9.85,
                             "faces_per_volume": 3000.0, "faces_per_area": 800.0},
            "geometry": {"surface_area": 5.0, "volume_estimate": 0.6, "occupancy_ratio": 0.6},
            "shape_descriptors": {"compactness": 3.0, "elongation": 1.0, "symmetry_axis": "Z"},
            "structure": {"connected_components": 5, "objects_count": 1},
            "materials_and_textures": {"avg_roughness": 0.4, "avg_metallic": 0.3},
            "physics_proxy": {"stability_score": 0.99},
        },
        num_views=12,
    ),
    # Good Carpet sample -- NO object_category by design
    dict(
        id="carpet_0000001",
        metadata={
            "id": "carpet_0000001", "super_category": "Carpet",
            "style_class": ["Industrial"],
            "materials": {"primary": "Fabric", "secondary": []},
            "source": {"annotation_method": "Human"},
        },
        geometry={
            "model_id": "carpet_0000001",
            "dimensions": {"width": 3.0, "depth": 2.0, "height": 0.02,
                           "aspect_hw": 0.006, "aspect_hd": 0.01, "aspect_wd": 1.5},
            "mesh_density": {"vertices": 400, "faces": 380,
                             "log_vertices": 5.99, "log_faces": 5.94,
                             "faces_per_volume": 100.0, "faces_per_area": 60.0},
            "geometry": {"surface_area": 6.0, "volume_estimate": 0.12, "occupancy_ratio": 1.0},
            "shape_descriptors": {"compactness": 1.1, "elongation": 1.5, "symmetry_axis": "Y"},
            "structure": {"connected_components": 1, "objects_count": 1},
            "materials_and_textures": {"avg_roughness": 0.8, "avg_metallic": 0.0},
            "physics_proxy": {"stability_score": 1.0},
        },
        num_views=12,
    ),
    # Storage sample (merged Shelf+Closet category)
    dict(
        id="storage_0000001",
        metadata={
            "id": "storage_0000001", "super_category": "Storage", "object_category": "Wardrobe",
            "style_class": ["French Classic"],
            "materials": {"primary": "Wood", "secondary": ["Glass"]},
            "source": {"annotation_method": "Human"},
        },
        geometry={
            "model_id": "storage_0000001",
            "dimensions": {"width": 1.5, "depth": 0.6, "height": 2.2,
                           "aspect_hw": 1.47, "aspect_hd": 3.67, "aspect_wd": 2.5},
            "mesh_density": {"vertices": 60000, "faces": 58000,
                             "log_vertices": 11.0, "log_faces": 10.97,
                             "faces_per_volume": 2000.0, "faces_per_area": 900.0},
            "geometry": {"surface_area": 12.0, "volume_estimate": 1.98, "occupancy_ratio": 0.9},
            "shape_descriptors": {"compactness": 2.0, "elongation": 2.9, "symmetry_axis": "Z"},
            "structure": {"connected_components": 20, "objects_count": 4},
            "materials_and_textures": {"avg_roughness": 0.3, "avg_metallic": 0.05},
            "physics_proxy": {"stability_score": 0.98},
        },
        num_views=12,
    ),
    # BROKEN: incomplete renders (only 5 of 12 views) -- must be excluded
    dict(
        id="chair_0000002_broken_renders",
        metadata={
            "id": "chair_0000002_broken_renders", "super_category": "Chair", "object_category": "Ottoman",
            "style_class": ["Minimalism"],
            "materials": {"primary": "Fabric", "secondary": []},
            "source": {"annotation_method": "Human"},
        },
        geometry={
            "model_id": "chair_0000002_broken_renders",
            "dimensions": {"width": 0.5, "depth": 0.5, "height": 0.4,
                           "aspect_hw": 0.8, "aspect_hd": 0.8, "aspect_wd": 1.0},
            "mesh_density": {"vertices": 5000, "faces": 4800,
                             "log_vertices": 8.5, "log_faces": 8.47,
                             "faces_per_volume": 500.0, "faces_per_area": 300.0},
            "geometry": {"surface_area": 2.0, "volume_estimate": 0.1, "occupancy_ratio": 0.5},
            "shape_descriptors": {"compactness": 2.5, "elongation": 1.0, "symmetry_axis": "Z"},
            "structure": {"connected_components": 3, "objects_count": 1},
            "materials_and_textures": {"avg_roughness": 0.5, "avg_metallic": 0.0},
            "physics_proxy": {"stability_score": 1.0},
        },
        num_views=5,  # deliberately incomplete
    ),
    # BROKEN: invalid object_category for its super_category (label validation should reject)
    dict(
        id="bed_0000003_bad_label",
        metadata={
            "id": "bed_0000003_bad_label", "super_category": "Bed", "object_category": "Armchair",  # invalid!
            "style_class": ["Minimalism"],
            "materials": {"primary": "Fabric", "secondary": []},
            "source": {"annotation_method": "Human"},
        },
        geometry={
            "model_id": "bed_0000003_bad_label",
            "dimensions": {"width": 2.0, "depth": 2.0, "height": 1.0,
                           "aspect_hw": 0.5, "aspect_hd": 0.5, "aspect_wd": 1.0},
            "mesh_density": {"vertices": 1000, "faces": 900,
                             "log_vertices": 6.9, "log_faces": 6.8,
                             "faces_per_volume": 100.0, "faces_per_area": 90.0},
            "geometry": {"surface_area": 4.0, "volume_estimate": 4.0, "occupancy_ratio": 1.0},
            "shape_descriptors": {"compactness": 1.0, "elongation": 1.0, "symmetry_axis": "Z"},
            "structure": {"connected_components": 1, "objects_count": 1},
            "materials_and_textures": {"avg_roughness": 0.5, "avg_metallic": 0.5},
            "physics_proxy": {"stability_score": 1.0},
        },
        num_views=12,
    ),
    # MISSING metadata entirely -- should be excluded by discover_samples (has model+renders only)
    dict(
        id="chair_0000003_no_metadata",
        metadata=None,  # sentinel: skip writing metadata.json
        geometry={
            "model_id": "chair_0000003_no_metadata",
            "dimensions": {"width": 1.0, "depth": 1.0, "height": 1.0,
                           "aspect_hw": 1.0, "aspect_hd": 1.0, "aspect_wd": 1.0},
        },
        num_views=12,
    ),
]


def build():
    for entry in SAMPLES:
        sid = entry["id"]
        _write_dummy_model_file(FIXTURE_ROOT / "models" / f"{sid}.fbx")

        if entry["metadata"] is not None:
            _write_json(FIXTURE_ROOT / "metadata" / f"{sid}.json", entry["metadata"])

        if entry["geometry"] is not None:
            _write_json(FIXTURE_ROOT / "geometry" / f"{sid}.json", entry["geometry"])

        _write_renders(FIXTURE_ROOT / "renders" / sid, num_views=entry["num_views"])

    print(f"Mock dataset built at {FIXTURE_ROOT}")
    print(f"Total sample folders: {len(SAMPLES)}")


if __name__ == "__main__":
    build()
