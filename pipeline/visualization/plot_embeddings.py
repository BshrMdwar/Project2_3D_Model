"""
visualization/plot_embeddings.py

Projects the fused embeddings (pooled image + geometry, right before the
output heads) down to 2D via t-SNE, colored by a chosen label field
(super_category by default). Useful for a qualitative sanity check: do
embeddings for the same super_category cluster together? Does adding geometry
fusion visibly separate previously-confused classes?

Usage:
    python visualization/plot_embeddings.py \\
        --checkpoint checkpoints/exp02_full_geometry/fold_0/self_contained.pt \\
        --color_by super_category \\
        --output tsne_plot.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # allow running as a script

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from inference import load_checkpoint, rebuild_model_from_checkpoint
from dataset import ArchDataset, ActiveTaxonomy


def extract_fused_embeddings(checkpoint_path: str, max_samples: int | None = None) -> dict:
    """
    Runs every sample in the FULL dataset (per the checkpoint's config) through
    the model up to (but not including) the output heads, collecting the fused
    embedding + label metadata for each.

    Returns:
        {
            "embeddings": np.ndarray (N, D),
            "super_category": list[str] (N,),
            "object_category": list[str] (N,),
            "style_class": list[list[str]] (N,),
        }
    """
    checkpoint = load_checkpoint(checkpoint_path)
    model, config, active_super_categories, active_object_categories = rebuild_model_from_checkpoint(checkpoint)

    active_taxonomy = ActiveTaxonomy.__new__(ActiveTaxonomy)
    active_taxonomy.active_super_categories = active_super_categories
    active_taxonomy.active_object_categories = active_object_categories

    dataset = ArchDataset(config, active_taxonomy=active_taxonomy, transform=None, is_training=False)
    if checkpoint.get("scaler") is not None:
        import pickle
        dataset.set_geometry_scaler(pickle.loads(checkpoint["scaler"]))

    embeddings, supers, objects, styles = [], [], [], []

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            images = sample["images"].unsqueeze(0)
            geometry = sample["geometry_vector"].unsqueeze(0) if sample["geometry_vector"] is not None else None

            view_embeddings = model.backbone(images)
            pooled = model.pooling(view_embeddings)
            if model.freeze_visual_backbone_entirely and model.geometry_only_projection is not None:
                geo_proj = model.geometry_only_projection(geometry)
                fused = torch.cat([pooled, geo_proj], dim=-1)
            else:
                fused = model.fusion(pooled, geometry)

            embeddings.append(fused[0].numpy())
            rec = dataset.records[i]
            supers.append(rec.metadata_json.get("super_category", "unknown"))
            objects.append(rec.metadata_json.get("object_category", "") or "(none)")
            styles.append(rec.metadata_json.get("style_class", []))

    return {
        "embeddings": np.stack(embeddings) if embeddings else np.zeros((0, 1)),
        "super_category": supers,
        "object_category": objects,
        "style_class": styles,
    }


def plot_tsne(data: dict, color_by: str, output_path: str, perplexity: float = 30.0) -> None:
    embeddings = data["embeddings"]
    n = embeddings.shape[0]
    if n < 3:
        print(f"Not enough samples ({n}) to run t-SNE meaningfully. Skipping plot.")
        return

    safe_perplexity = min(perplexity, max(1.0, (n - 1) / 3))
    tsne = TSNE(n_components=2, perplexity=safe_perplexity, random_state=42, init="pca")
    coords = tsne.fit_transform(embeddings)

    if color_by == "style_class":
        # multi-label -- color by first tag present, or "none"
        labels = [tags[0] if tags else "(none)" for tags in data["style_class"]]
    else:
        labels = data[color_by]

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    label_to_color = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    plt.figure(figsize=(10, 8))
    for lbl in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == lbl]
        plt.scatter(coords[idxs, 0], coords[idxs, 1], label=lbl, color=label_to_color[lbl], alpha=0.7, s=40)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.title(f"t-SNE of fused embeddings, colored by {color_by}")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved t-SNE plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize fused embeddings via t-SNE.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--color_by", type=str, default="super_category",
                         choices=["super_category", "object_category", "style_class"])
    parser.add_argument("--output", type=str, default="tsne_plot.png")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()

    data = extract_fused_embeddings(args.checkpoint, max_samples=args.max_samples)
    plot_tsne(data, args.color_by, args.output, perplexity=args.perplexity)


if __name__ == "__main__":
    main()
