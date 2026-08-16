"""
taxonomy.py

Single source of truth for all label taxonomy used across the training pipeline.
This file contains ONLY data + simple query helpers. No I/O, no torch, no training logic.

IMPORTANT DEVIATION FROM THE ORIGINAL SPEC (documented per explicit user decision):
- The original spec listed 11 super_categories: Chair, Bed, Table, Sofa, Light, Window,
  Door, Shelf, Cabinet, Closet, Carpet.
- By explicit user request, "Shelf" and "Closet" have been MERGED into a single new
  super_category called "Storage" (they serve the same functional purpose). This
  reduces the taxonomy to 10 super_categories.
- By explicit user request, "Carpet" has NO object_category (no sub-classification).
  It is classified only at the super_category level. This is represented by an
  empty list in SUPER_CATEGORIES, and every downstream consumer (heads.py, losses.py,
  dataset.py) must treat an empty sub-category list as "this super_category has no
  object_category head at all" -- not an error, not a placeholder.

Everything else in this file matches the original spec's taxonomy verbatim.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Core taxonomy: super_category -> list of valid object_category values.
# An empty list means this super_category has NO object_category classification
# (currently only Carpet, by explicit user decision).
# ---------------------------------------------------------------------------
SUPER_CATEGORIES: dict[str, list[str]] = {
    "Chair": [
        "Armchair", "Ottoman", "Vanity Chair", "Desk Chair",
        "Recliner Chair", "Bar Stool", "Dinner Chair", "Rocking Chair",
    ],
    "Bed": [
        "Single", "Twin XL", "Double", "Queen", "King",
        "Super King", "Emperor King", "Bunk Bed",
    ],
    "Table": [
        "Side Table", "Dinner Table", "Coffee Table", "Desk Table",
        "Meeting Table", "Bar Counter", "Nightstand Table",
    ],
    "Sofa": [
        "Love Seats", "Standard", "Large", "Sectional", "Recliner Sofas",
    ],
    "Light": [
        "Floor Mounted", "Wall Mounted", "Ceiling Mounted", "Free Floating",
    ],
    "Window": [
        "Casement", "Double Hung", "Awning", "Sliding", "Picture",
        "Bay and Bow", "Hopper", "Skylight and Roof",
    ],
    "Door": [
        "Panel Doors", "Flush Doors", "Glass Doors", "Louvered Doors",
    ],
    # --- merged super_category (Shelf + Closet), per explicit user decision ---
    "Storage": [
        "Standalone Shelves", "Wall Mounted Shelves", "Built In Shelves",
        "Walk In Closet", "Reach In Closet", "Wardrobe",
    ],
    "Cabinet": [
        "Floor Cabinets", "Wall Cabinets", "Tall Cabinets", "Corner Cabinets",
    ],
    # --- no object_category, per explicit user decision ---
    "Carpet": [],
}

STYLE_CLASSES: list[str] = [
    "Minimalism", "Bauhaus", "Post Modern",
    "French Classic", "Industrial", "Tropical",
]

MATERIALS: list[str] = [
    "Wood", "Metal", "Glass", "Concrete", "Stone", "Fabric", "Plastic", "3D Printed",
]


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def is_valid_super_category(super_category: str) -> bool:
    """Check whether a super_category string exists in the taxonomy."""
    return super_category in SUPER_CATEGORIES


def has_object_category(super_category: str) -> bool:
    """
    Whether this super_category has ANY object_category sub-classification at all.
    Returns False for Carpet (by design), and for any unknown super_category.
    """
    if not is_valid_super_category(super_category):
        return False
    return len(SUPER_CATEGORIES[super_category]) > 0


def get_object_categories(super_category: str) -> list[str]:
    """
    Return the full list of valid object_category values defined in the taxonomy
    for a given super_category. Returns [] if the super_category is unknown OR
    if it has no object_category classification (e.g. Carpet).
    """
    return SUPER_CATEGORIES.get(super_category, [])


def get_num_object_categories(super_category: str) -> int:
    """Number of object_category values defined in the taxonomy for this super_category."""
    return len(get_object_categories(super_category))


def is_valid_object_category(super_category: str, object_category: str) -> bool:
    """
    Whether `object_category` is a legal value for the given `super_category`
    according to the taxonomy. Always False if super_category is unknown or has
    no object_category classification.
    """
    return object_category in get_object_categories(super_category)


def get_all_super_categories() -> list[str]:
    """All super_category names defined in the taxonomy (fixed reference list)."""
    return list(SUPER_CATEGORIES.keys())


def get_num_super_categories() -> int:
    """Total number of super_categories defined in the taxonomy."""
    return len(SUPER_CATEGORIES)


def is_valid_style(style: str) -> bool:
    return style in STYLE_CLASSES


def is_valid_material(material: str) -> bool:
    return material in MATERIALS
