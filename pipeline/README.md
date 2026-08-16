# Training Pipeline

Multi-task 3D architecture/furniture asset classifier: MVCNN-style multi-view
image backbone (+ optional geometry feature fusion) predicting super_category,
object_category (per super_category sub-head), style_class (multi-label),
materials_primary, and materials_secondary (multi-label).

## Status

**146/146 unit + integration tests passing**, including full end-to-end
training loop smoke tests (real PyTorch Lightning `Trainer.fit()` run against
a synthetic mock dataset) and full end-to-end inference tests (checkpoint
save -> load -> predict). Tested in this sandbox with a tiny fake backbone
(no network access here); real backbones (DINOv2/ConvNeXt/EfficientNet/CLIP)
need to be smoke-tested once on your machine with internet access, since they
download pretrained weights on first use.

## Setup

```bash
pip install -r requirements.txt
```

## Quick start

1. **Check your dataset first** (run this any time, especially as data grows):
   ```bash
   python explore_dataset.py --root_dir dataset/ --save_json health_report.json
   ```
   This prints a full breakdown: sample counts per (super_category,
   object_category), which categories are structurally active vs still empty,
   label/geometry validation failures, low-confidence combos, and style/material
   tag distributions.

2. **Run an experiment**:
   ```bash
   python train.py --experiment configs/experiments/exp01_baseline.yaml
   ```
   Six example experiment configs are provided (see `configs/experiments/`):
   - `exp01_baseline.yaml` — images only, no geometry
   - `exp02_full_geometry.yaml` — images + all geometry features
   - `exp03_geometry_shape_only.yaml` — images + shape-descriptor geometry subset
   - `exp04_geometry_materials_only.yaml` — images + material-property geometry subset
   - `exp05_attention_pooling.yaml` — attention-based view pooling instead of mean
   - `exp06_geometry_only.yaml` — visual backbone frozen entirely, geometry-only ablation

   Add a new experiment by copying one of these and adjusting fields — no code
   changes needed for any config-driven variation (backbone choice, pooling
   method, which geometry features to use, imbalance-handling toggles, etc).

3. **Run inference** on a trained checkpoint:
   ```python
   from inference import predict
   result = predict(
       "checkpoints/exp02_full_geometry/fold_0/self_contained.pt",
       render_image_paths=["view_01.png", "view_02.png", ...],  # up to 12
       geometry_json=parsed_geometry_dict,  # optional, only if the model needs it
   )
   print(result["predictions"])
   print(result["warnings"])
   ```
   Every checkpoint is self-contained (weights + config + active taxonomy +
   fitted geometry scaler) — inference needs nothing else.

4. **Visualize learned embeddings**:
   ```bash
   python visualization/plot_embeddings.py \
       --checkpoint checkpoints/exp02_full_geometry/fold_0/self_contained.pt \
       --color_by super_category --output tsne.png
   ```

## Key design decisions (confirmed with you during design)

- **Taxonomy**: 10 super_categories. `Shelf` + `Closet` merged into a single
  `Storage` category (they serve the same purpose). `Carpet` has no
  object_category sub-classification at all (`taxonomy.py`).
- **object_category head**: Option (B) — a **separate** `nn.Linear` sub-head
  per super_category (not one shared head with masking). See `models/heads.py`.
- **Dynamic head sizes**: output head sizes are computed from whichever
  super_categories/object_categories currently have ≥1 sample in the dataset
  (`dataset.ActiveTaxonomy`), not from the full taxonomy. Zero-sample classes
  are excluded automatically. Adding data for a brand-new class later requires
  retraining from scratch for that class (head size changes), not resuming.
- **K-Fold on near-empty classes**: per the original spec, any
  (super_category, object_category) combo with fewer samples than `k_folds`
  is included in **every** fold (train and val) rather than raising an error.
  This is logged clearly as a warning (`kfold_split.py`).
- **No undersampling / no dropped samples**: every valid sample is used,
  regardless of how rare its class is. Imbalance is handled entirely through
  loss weighting / focal loss / balanced sampling (`imbalance_handling.py`),
  never by removing data.
- **materials_breakdown** (variable-length geometry field) is intentionally
  **not supported** as a model feature — referencing it in a config raises a
  clear error at config-load time rather than failing deep into training.

## File overview

| File | Purpose |
|---|---|
| `taxonomy.py` | Single source of truth for category/style/material label lists |
| `config.py` | Typed dataclass config system, YAML base+experiment merging |
| `seed.py` | Reproducibility (random/numpy/torch seeding) |
| `geometry_features.py` | Extracts fixed-length numeric feature vectors from geometry JSON |
| `geometry_validation.py` | Sanity-checks geometry JSON (positivity, NaN/Inf, outliers) |
| `label_validation.py` | Validates metadata JSON against `taxonomy.py`'s hierarchy |
| `dataset.py` | `ArchDataset` (PyTorch Dataset), `ActiveTaxonomy`, sample discovery |
| `augmentation.py` | Multi-view-consistent image augmentation |
| `imbalance_handling.py` | Class weighting, focal loss, balanced sampling, low-confidence flagging |
| `kfold_split.py` | Stratified K-Fold splitting with rare-class handling |
| `feature_cache.py` | Disk cache for frozen-backbone embeddings |
| `metrics.py` | Per-head classification metrics, per-super_category object_category metrics |
| `models/backbones.py` | Swappable image backbones (DINOv2/ConvNeXt/EfficientNet/CLIP) |
| `models/fusion.py` | View pooling (mean/max/attention) + geometry fusion |
| `models/heads.py` | Multi-task output heads (super_category, per-category object_category sub-heads, style, materials) |
| `models/factory.py` | Assembles the complete model from config |
| `losses.py` | Multi-task loss aggregation |
| `train.py` | Main training orchestrator (PyTorch Lightning, K-Fold loop) |
| `inference.py` | Self-contained checkpoint loading + prediction |
| `explore_dataset.py` | Dataset Health Report CLI |
| `visualization/plot_embeddings.py` | t-SNE visualization of learned embeddings |

## Testing

```bash
for f in tests/test_*.py; do python3 "$f"; done
```

`tests/build_mock_dataset.py` generates a small synthetic dataset (good
samples, broken renders, missing metadata, invalid labels, a Carpet sample
with no object_category) under `tests/fixtures/mock_dataset/` — used by the
end-to-end tests. Real backbone downloads are avoided in tests via a tiny
fake `Backbone` subclass registered into `BACKBONE_REGISTRY`; swap back to a
real `backbone_name` in your experiment YAML for actual training runs.

## Running on Kaggle

The pipeline was written to be portable by default: relative paths (no
hardcoded Windows paths), `accelerator: "auto"` in `training.yaml` (Lightning
resolves CPU/GPU automatically), and no OS-specific code anywhere. To run on
Kaggle:
1. Upload `dataset/` as a Kaggle Dataset (or mount via Kaggle's dataset browser).
2. Upload this repo as a Kaggle Dataset or clone it into the notebook.
3. `pip install -r requirements.txt` in the first notebook cell.
4. Point `configs/base/dataset.yaml`'s `root_dir` at the mounted dataset path
   (e.g. `/kaggle/input/your-dataset-name`), or override it via a copied
   experiment YAML.
5. Run `python train.py --experiment ...` as usual.

First run on any new machine (Kaggle included) needs internet access once to
download pretrained backbone weights via `torch.hub` / `torchvision`.
