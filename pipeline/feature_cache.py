"""
feature_cache.py

Caches per-sample, per-view backbone embeddings to disk, keyed by
(sample_id, backbone_name). This matters a lot for this project's workflow:
when running MANY ablation experiments (spec section 14) that only differ in
downstream fusion/heads/pooling but share the SAME frozen (or partially-frozen)
backbone, recomputing the backbone forward pass every single epoch for every
experiment is wasteful -- especially on Kaggle's limited/timed GPU sessions.

Cache invalidation strategy: the cache key includes the backbone_name AND
whether that backbone had any unfrozen (trainable) layers. If
freeze_backbone_except_last_n_layers > 0, SOME backbone weights are actually
being trained, so cached embeddings from a PREVIOUS epoch are stale after every
optimizer step -- in that case, the cache is used only as a warm-start for the
FIRST epoch's frozen portion... but to keep this simple and safe against subtle
bugs, caching is only enabled by default when the backbone is entirely frozen
(freeze_backbone_except_last_n_layers == 0), which is exactly the setting used
for pure ablation studies on fusion/pooling/heads (spec 14 cases 1-2). Training
with partially-unfrozen backbone layers bypasses the cache entirely (correctness
over speed), controlled automatically -- no manual toggling needed to avoid this
footgun.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FeatureCache:
    def __init__(self, cache_dir: str | Path, backbone_name: str, backbone_is_fully_frozen: bool):
        self.cache_dir = Path(cache_dir)
        self.backbone_name = backbone_name
        self.enabled = backbone_is_fully_frozen
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(
                f"FeatureCache disabled for backbone '{backbone_name}': backbone has "
                f"trainable layers, so cached embeddings would go stale after the "
                f"first optimizer step. Recomputing every forward pass instead "
                f"(correctness over speed for partially fine-tuned backbones)."
            )

    def _cache_path(self, sample_id: str) -> Path:
        # backbone_name may contain characters unsafe for filenames in edge cases; hash it
        safe_backbone = hashlib.md5(self.backbone_name.encode()).hexdigest()[:8]
        return self.cache_dir / f"{sample_id}__{safe_backbone}.npy"

    def get(self, sample_id: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        path = self._cache_path(sample_id)
        if not path.exists():
            return None
        try:
            return np.load(path)
        except (OSError, ValueError) as e:
            logger.warning(f"FeatureCache: failed to load cached embedding for '{sample_id}' ({e}); recomputing.")
            return None

    def set(self, sample_id: str, embedding: np.ndarray) -> None:
        if not self.enabled:
            return
        path = self._cache_path(sample_id)
        try:
            np.save(path, embedding)
        except OSError as e:
            logger.warning(f"FeatureCache: failed to write cache for '{sample_id}' ({e}); continuing without caching it.")

    def clear(self) -> None:
        if not self.enabled or not self.cache_dir.exists():
            return
        count = 0
        for f in self.cache_dir.glob("*.npy"):
            f.unlink()
            count += 1
        logger.info(f"FeatureCache: cleared {count} cached embedding file(s) from {self.cache_dir}.")
