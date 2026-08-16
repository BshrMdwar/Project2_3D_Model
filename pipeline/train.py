"""
train.py

Main training entry point. Orchestrates:
    1. Load PipelineConfig from an experiment YAML.
    2. Seed everything (seed.py).
    3. Discover + validate the full dataset once (dataset.py), building the
       ONE shared ActiveTaxonomy used across every fold (so all folds agree
       on class indices/head sizes -- switching folds must never change what
       index "Bed" or "King" maps to).
    4. Build K stratified folds (kfold_split.py).
    5. For each fold:
        a. Build train/val ArchDataset splits sharing the same ActiveTaxonomy.
        b. Fit a StandardScaler for geometry features on the TRAINING split
           only (never on validation data -- avoids leakage), inject into
           both splits.
        c. Build the model (models/factory.py) fresh for this fold.
        d. Build the MultiTaskLoss (losses.py), computing class weights from
           the training split's labels only.
        e. Train with a PyTorch Lightning Trainer (early stopping, checkpointing).
        f. Save a self-contained checkpoint (model weights + config + active
           taxonomy + scaler + fold metadata) per design doc section 6, so
           inference.py needs nothing but this one file.
    6. Aggregate metrics across folds and report mean +/- std per metric.

Usage:
    python train.py --experiment configs/experiments/exp02_full_geometry.yaml
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    _LIGHTNING_AVAILABLE = True
except ImportError:
    _LIGHTNING_AVAILABLE = False

from sklearn.preprocessing import StandardScaler

import config as cfg
import seed as seed_mod
import taxonomy as tx
from dataset import ArchDataset, ActiveTaxonomy, discover_samples
from augmentation import MultiViewAugmentation
from kfold_split import make_kfold_splits
from models.factory import build_model
from losses import MultiTaskLoss
import imbalance_handling as ih
import metrics as metrics_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_full_active_taxonomy(config: cfg.PipelineConfig) -> ActiveTaxonomy:
    """
    Discover + validate the FULL dataset once to compute a single ActiveTaxonomy
    shared across every fold. This must happen before any fold-specific split,
    or different folds could end up with different head sizes/class orderings.
    """
    full_dataset = ArchDataset(config)  # discovers + validates everything, no record_ids filter
    return full_dataset.active_taxonomy, full_dataset


def fit_geometry_scaler(train_dataset: ArchDataset) -> StandardScaler | None:
    """Fit a StandardScaler on the TRAINING fold's geometry vectors only (no leakage)."""
    if not train_dataset.config.model.use_geometry_fusion:
        return None

    vectors = []
    for rec in train_dataset.records:
        from geometry_features import extract_geometry_vector
        vec, _ = extract_geometry_vector(rec.geometry_json, train_dataset.geometry_feature_list)
        vectors.append(vec)

    if not vectors:
        logger.warning("No geometry vectors available to fit scaler -- geometry fusion will get raw (unscaled) values.")
        return None

    scaler = StandardScaler()
    scaler.fit(np.stack(vectors))
    return scaler


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate: images/labels stack normally, but object_category_logits
    routing needs the raw super_category index per sample (handled downstream
    in the LightningModule, not here) -- geometry_vector may be None for the
    whole batch (fusion disabled) so we handle that case explicitly.
    """
    images = torch.stack([b["images"] for b in batch])

    geometry_vectors = None
    if batch[0]["geometry_vector"] is not None:
        geometry_vectors = torch.stack([b["geometry_vector"] for b in batch])

    labels = {
        "super_category": torch.tensor([b["labels"]["super_category"] for b in batch], dtype=torch.long),
        "object_category": torch.tensor([b["labels"]["object_category"] for b in batch], dtype=torch.long),
        "style_class": torch.stack([b["labels"]["style_class"] for b in batch]),
        "materials_primary": torch.tensor([b["labels"]["materials_primary"] for b in batch], dtype=torch.long),
        "materials_secondary": torch.stack([b["labels"]["materials_secondary"] for b in batch]),
    }
    sample_weights = torch.tensor([b["sample_weight"] for b in batch], dtype=torch.float32)
    ids = [b["id"] for b in batch]

    return {
        "ids": ids,
        "images": images,
        "geometry_vector": geometry_vectors,
        "labels": labels,
        "sample_weights": sample_weights,
    }


if _LIGHTNING_AVAILABLE:
    class LightningWrapper(pl.LightningModule):
        """Thin Lightning wrapper around CompleteModel + MultiTaskLoss for a single fold."""

        def __init__(self, model, loss_fn, config: cfg.PipelineConfig, active_taxonomy: ActiveTaxonomy):
            super().__init__()
            self.model = model
            self.loss_fn = loss_fn
            self.config = config
            self.active_taxonomy = active_taxonomy
            self._val_super_true, self._val_super_pred = [], []
            self._val_object_accumulator = metrics_mod.ObjectCategoryMetricsAccumulator()

        def forward(self, images, geometry_vector=None, super_category_gt=None):
            return self.model(images, geometry_vector=geometry_vector, super_category_gt=super_category_gt)

        def training_step(self, batch, batch_idx):
            outputs = self(batch["images"], batch["geometry_vector"], batch["labels"]["super_category"])
            losses = self.loss_fn(outputs, batch["labels"], batch["sample_weights"])
            self.log("train_total_loss", losses["total_loss"], prog_bar=True, batch_size=len(batch["ids"]))
            for key in ["super_category_loss", "object_category_loss", "style_loss",
                        "materials_primary_loss", "materials_secondary_loss"]:
                self.log(f"train_{key}", losses[key], batch_size=len(batch["ids"]))
            return losses["total_loss"]

        def validation_step(self, batch, batch_idx):
            outputs = self(batch["images"], batch["geometry_vector"], batch["labels"]["super_category"])
            losses = self.loss_fn(outputs, batch["labels"], batch["sample_weights"])
            self.log("val_total_loss", losses["total_loss"], prog_bar=True, batch_size=len(batch["ids"]))

            super_pred = outputs["super_category_logits"].argmax(dim=-1).cpu().tolist()
            super_true = batch["labels"]["super_category"].cpu().tolist()
            self._val_super_true.extend(super_true)
            self._val_super_pred.extend(super_pred)

            super_names = [self.active_taxonomy.active_super_categories[i] for i in super_true]
            object_pred = [
                (int(l.argmax().item()) if l is not None else -1)
                for l in outputs["object_category_logits"]
            ]
            object_true = batch["labels"]["object_category"].cpu().tolist()
            self._val_object_accumulator.add_batch(super_names, object_true, object_pred)

            return losses["total_loss"]

        def on_validation_epoch_end(self):
            num_super = len(self.active_taxonomy.active_super_categories)
            super_metrics = metrics_mod.compute_single_label_metrics(
                self._val_super_true, self._val_super_pred, num_super
            )
            self.log("val_super_category_f1", super_metrics["f1_macro"], prog_bar=True)
            self.log("val_super_category_accuracy", super_metrics["accuracy"])

            num_classes_per_super = {
                sc: self.active_taxonomy.num_object_categories(sc)
                for sc in self.active_taxonomy.active_super_categories
                if tx.has_object_category(sc) and self.active_taxonomy.num_object_categories(sc) > 0
            }
            object_report = self._val_object_accumulator.report(num_classes_per_super)
            self.log("val_object_category_macro_f1", object_report["macro_avg_f1"])

            self._val_super_true, self._val_super_pred = [], []
            self._val_object_accumulator = metrics_mod.ObjectCategoryMetricsAccumulator()

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
            return optimizer


def train_single_fold(
    config: cfg.PipelineConfig,
    active_taxonomy: ActiveTaxonomy,
    train_ids: list[str],
    val_ids: list[str],
    fold_index: int,
) -> dict:
    if not _LIGHTNING_AVAILABLE:
        raise ImportError(
            "pytorch_lightning is required for train.py. Install it with: "
            "pip install pytorch-lightning --break-system-packages"
        )

    train_transform = MultiViewAugmentation(config.augmentation)
    train_dataset = ArchDataset(
        config, active_taxonomy=active_taxonomy, record_ids=train_ids,
        transform=train_transform, is_training=True,
    )
    val_dataset = ArchDataset(
        config, active_taxonomy=active_taxonomy, record_ids=val_ids,
        transform=None, is_training=False,
    )

    scaler = fit_geometry_scaler(train_dataset)
    if scaler is not None:
        train_dataset.set_geometry_scaler(scaler)
        val_dataset.set_geometry_scaler(scaler)

    sampler = None
    shuffle = True
    if config.loss.use_balanced_batch_sampling:
        train_labels = [rec.metadata_json.get("super_category") for rec in train_dataset.records]
        train_label_indices = [active_taxonomy.super_category_index(sc) for sc in train_labels]
        sampler = ih.build_weighted_random_sampler(train_label_indices)
        shuffle = False

    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=config.training.num_workers, collate_fn=collate_fn,
        worker_init_fn=seed_mod.seed_worker if config.training.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers, collate_fn=collate_fn,
    )

    model = build_model(config, active_taxonomy)

    super_category_labels = [
        active_taxonomy.super_category_index(rec.metadata_json.get("super_category"))
        for rec in train_dataset.records
    ]
    object_category_labels_by_super: dict[str, list[int]] = {}
    materials_primary_labels = []
    for rec in train_dataset.records:
        sc = rec.metadata_json.get("super_category")
        oc = rec.metadata_json.get("object_category")
        if tx.has_object_category(sc) and oc in active_taxonomy.active_object_categories.get(sc, []):
            object_category_labels_by_super.setdefault(sc, []).append(
                active_taxonomy.object_category_index(sc, oc)
            )
        primary = (rec.metadata_json.get("materials", {}) or {}).get("primary")
        if primary in tx.MATERIALS:
            materials_primary_labels.append(tx.MATERIALS.index(primary))

    object_category_class_counts = {
        sc: active_taxonomy.num_object_categories(sc)
        for sc in model.heads.object_category_heads.keys()
    }

    loss_fn = MultiTaskLoss(
        loss_config=config.loss,
        num_super_categories=len(active_taxonomy.active_super_categories),
        object_category_class_counts=object_category_class_counts,
        super_category_labels_for_weights=super_category_labels,
        object_category_labels_for_weights=object_category_labels_by_super,
        materials_primary_labels_for_weights=materials_primary_labels,
    )

    lightning_module = LightningWrapper(model, loss_fn, config, active_taxonomy)

    checkpoint_dir = Path(config.system.checkpoint_dir) / config.experiment_name / f"fold_{fold_index}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor=config.training.monitor_metric, mode=config.training.monitor_mode,
            patience=config.training.early_stopping_patience,
        ),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir), filename="best",
            monitor=config.training.monitor_metric, mode=config.training.monitor_mode, save_top_k=1,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        precision=config.training.precision,
        callbacks=callbacks,
        default_root_dir=str(Path(config.system.tensorboard_dir) / config.experiment_name / f"fold_{fold_index}"),
        logger=pl.loggers.TensorBoardLogger(
            save_dir=config.system.tensorboard_dir, name=config.experiment_name, version=f"fold_{fold_index}"
        ),
    )

    trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_checkpoint_path = callbacks[1].best_model_path

    # Save a SELF-CONTAINED checkpoint (design doc section 6): model weights +
    # full config + active taxonomy + scaler + fold metadata, all in one file,
    # so inference.py needs nothing else to reconstruct the model.
    self_contained_path = checkpoint_dir / "self_contained.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "active_super_categories": active_taxonomy.active_super_categories,
        "active_object_categories": active_taxonomy.active_object_categories,
        "scaler": pickle.dumps(scaler) if scaler is not None else None,
        "experiment_name": config.experiment_name,
        "fold": fold_index,
    }, self_contained_path)

    val_metrics = dict(trainer.callback_metrics)
    return {k: (v.item() if torch.is_tensor(v) else v) for k, v in val_metrics.items()}


def run_experiment(experiment_yaml_path: str) -> dict:
    config = cfg.load_experiment_config(experiment_yaml_path)
    seed_mod.set_seed(config.system.seed, config.system.deterministic_cudnn)

    logger.info(f"Running experiment '{config.experiment_name}'")

    active_taxonomy, full_dataset = build_full_active_taxonomy(config)
    logger.info(
        f"Active super_categories ({len(active_taxonomy.active_super_categories)}): "
        f"{active_taxonomy.active_super_categories}"
    )

    sample_ids = [rec.id for rec in full_dataset.records]
    super_categories = [rec.metadata_json.get("super_category") for rec in full_dataset.records]
    object_categories = [rec.metadata_json.get("object_category", "") or "" for rec in full_dataset.records]

    folds = make_kfold_splits(
        sample_ids, super_categories, object_categories,
        k_folds=config.dataset.k_folds, seed=config.system.seed,
    )

    fold_results = []
    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        logger.info(f"--- Fold {fold_idx + 1}/{len(folds)}: train={len(train_ids)}, val={len(val_ids)} ---")
        result = train_single_fold(config, active_taxonomy, train_ids, val_ids, fold_idx)
        fold_results.append(result)

    aggregated = {}
    if fold_results:
        all_keys = set().union(*[r.keys() for r in fold_results])
        for key in all_keys:
            values = [r[key] for r in fold_results if key in r]
            if values:
                aggregated[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    logger.info(f"Experiment '{config.experiment_name}' complete. Aggregated metrics: {aggregated}")
    return {"fold_results": fold_results, "aggregated": aggregated}


def main():
    parser = argparse.ArgumentParser(description="Train a multi-task 3D asset classification model.")
    parser.add_argument("--experiment", type=str, required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()
    run_experiment(args.experiment)


if __name__ == "__main__":
    main()
