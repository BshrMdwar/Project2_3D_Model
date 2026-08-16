"""
explore_dataset.py

Generates a "Dataset Health Report" -- a human-readable summary of the current
dataset state, meant to be run BEFORE starting any training run (and re-run
periodically as more data is collected, since the dataset is explicitly
described as growing/non-final). Surfaces:

    - per-(super_category, object_category) sample counts (mirrors the shape
      of the Dataset_Furniture_Entry.pdf table, but computed live from disk)
    - which super_categories/object_categories are currently ACTIVE
      (dataset.ActiveTaxonomy) vs fully absent from the taxonomy
    - combos flagged as low-confidence (imbalance_handling.compute_low_confidence_combos)
    - geometry validation failure counts (geometry_validation.py)
    - label validation failure counts (label_validation.py)
    - basic style_class / materials distribution (multi-label, so counted
      per-tag not per-sample)
    - reference geometry statistics (mean/std per numeric field, for later
      outlier detection via geometry_validation.compute_reference_stats)

Usage:
    python explore_dataset.py --root_dir dataset/
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import taxonomy as tx
import label_validation as lv
import geometry_validation as gv
import imbalance_handling as ih
from dataset import discover_samples, ActiveTaxonomy

logging.basicConfig(level=logging.WARNING)  # keep report output clean; use logs only for real issues
logger = logging.getLogger(__name__)


def generate_report(root_dir: str, expected_views: int = 12, min_confidence_samples: int = 10) -> dict:
    records, discovery_warnings = discover_samples(
        root_dir=root_dir, expected_views=expected_views, require_geometry=False,
    )

    label_pass, label_fail = [], []
    geometry_pass, geometry_fail = [], []
    style_tag_counts: Counter = Counter()
    materials_primary_counts: Counter = Counter()
    materials_secondary_counts: Counter = Counter()
    combo_counts: Counter = Counter()

    for rec in records:
        result = lv.validate_label(rec.metadata_json, model_id=rec.id)
        (label_pass if result.is_valid else label_fail).append((rec.id, result.errors))

        if rec.geometry_json:
            geo_result = gv.validate_geometry(rec.geometry_json, model_id=rec.id)
            (geometry_pass if geo_result.is_valid else geometry_fail).append((rec.id, geo_result.errors))

        if result.is_valid:
            sc = rec.metadata_json.get("super_category")
            oc = rec.metadata_json.get("object_category", "")
            combo_counts[(sc, oc)] += 1

            for style in rec.metadata_json.get("style_class", []) or []:
                style_tag_counts[style] += 1

            materials = rec.metadata_json.get("materials", {}) or {}
            if materials.get("primary"):
                materials_primary_counts[materials["primary"]] += 1
            for mat in materials.get("secondary", []) or []:
                materials_secondary_counts[mat] += 1

    valid_metadata_jsons = [rec.metadata_json for rec in records if any(rec.id == pid for pid, _ in label_pass)]
    active_taxonomy = ActiveTaxonomy(valid_metadata_jsons) if valid_metadata_jsons else None

    low_confidence_combos = ih.compute_low_confidence_combos(
        super_categories=[sc for sc, oc in combo_counts.keys() for _ in range(combo_counts[(sc, oc)])],
        object_categories=[oc for sc, oc in combo_counts.keys() for _ in range(combo_counts[(sc, oc)])],
        min_samples=min_confidence_samples,
    )

    reference_stats = gv.compute_reference_stats([rec.geometry_json for rec in records if rec.geometry_json])

    all_super_categories = tx.get_all_super_categories()
    absent_super_categories = [
        sc for sc in all_super_categories
        if active_taxonomy is None or sc not in active_taxonomy.active_super_categories
    ]

    return {
        "total_discovered": len(records) + len(discovery_warnings),
        "total_structurally_valid": len(records),
        "discovery_warnings": discovery_warnings,
        "label_validation": {"passed": len(label_pass), "failed": len(label_fail), "failures": label_fail},
        "geometry_validation": {"passed": len(geometry_pass), "failed": len(geometry_fail), "failures": geometry_fail},
        "combo_counts": {f"{sc} / {oc or '(no object_category)'}": count for (sc, oc), count in sorted(combo_counts.items())},
        "active_super_categories": active_taxonomy.active_super_categories if active_taxonomy else [],
        "absent_super_categories": absent_super_categories,
        "low_confidence_combos": [f"{sc} / {oc or '(no object_category)'}" for sc, oc in sorted(low_confidence_combos)],
        "style_class_tag_counts": dict(style_tag_counts.most_common()),
        "materials_primary_counts": dict(materials_primary_counts.most_common()),
        "materials_secondary_tag_counts": dict(materials_secondary_counts.most_common()),
        "geometry_reference_stats": reference_stats,
    }


def print_report(report: dict) -> None:
    print("=" * 70)
    print("DATASET HEALTH REPORT")
    print("=" * 70)
    print(f"Total samples discovered on disk : {report['total_discovered']}")
    print(f"Structurally valid (usable)       : {report['total_structurally_valid']}")
    print(f"Excluded at discovery stage       : {len(report['discovery_warnings'])}")
    print(f"Label validation  - passed/failed : {report['label_validation']['passed']}/{report['label_validation']['failed']}")
    print(f"Geometry validation - passed/failed: {report['geometry_validation']['passed']}/{report['geometry_validation']['failed']}")
    print()
    print(f"Active super_categories ({len(report['active_super_categories'])}): {report['active_super_categories']}")
    print(f"Absent super_categories ({len(report['absent_super_categories'])}): {report['absent_super_categories']}")
    print()
    print("Sample counts per (super_category / object_category):")
    for combo, count in report["combo_counts"].items():
        flag = "  <-- LOW CONFIDENCE" if combo in report["low_confidence_combos"] else ""
        print(f"  {combo:50s} {count:5d}{flag}")
    print()
    print(f"Style class tag counts: {report['style_class_tag_counts']}")
    print(f"Materials (primary) counts: {report['materials_primary_counts']}")
    print(f"Materials (secondary) tag counts: {report['materials_secondary_tag_counts']}")
    print()
    if report["discovery_warnings"]:
        print(f"First 10 discovery warnings (of {len(report['discovery_warnings'])}):")
        for w in report["discovery_warnings"][:10]:
            print(f"  - {w}")
    if report["label_validation"]["failed"]:
        print(f"\nFirst 10 label validation failures (of {report['label_validation']['failed']}):")
        for sid, errors in report["label_validation"]["failures"][:10]:
            print(f"  - [{sid}] {errors}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate a Dataset Health Report.")
    parser.add_argument("--root_dir", type=str, default="dataset", help="Path to dataset/ directory")
    parser.add_argument("--expected_views", type=int, default=12)
    parser.add_argument("--min_confidence_samples", type=int, default=10)
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save the full report as JSON")
    args = parser.parse_args()

    report = generate_report(args.root_dir, args.expected_views, args.min_confidence_samples)
    print_report(report)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to {args.save_json}")


if __name__ == "__main__":
    main()
