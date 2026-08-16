"""
augmentation.py

Data augmentation techniques for multi-view rendered images. Every technique is
independently toggleable via AugmentationConfig (config.py / configs/base/augmentation.yaml).

CRITICAL CONSTRAINT (per spec section 9): augmentation must be applied with the
SAME random parameters across all 12 views of a given sample within a single
training step (multi-view consistency). Each view getting independently random
augmentation would break the geometric consistency MVCNN-style pooling relies on.

This is achieved by sampling all random parameters ONCE per __call__ (i.e. once
per sample), then applying those same fixed parameters identically to every
view in the (V, H, W, 3) array.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SampledAugmentationParams:
    """Parameters sampled ONCE per sample, then applied identically to all views."""
    background_color: tuple[float, float, float] | None = None
    brightness_factor: float = 1.0
    contrast_factor: float = 1.0
    saturation_factor: float = 1.0
    scale_factor: float = 1.0
    do_horizontal_flip: bool = False
    rotation_degrees: float = 0.0


class MultiViewAugmentation:
    """
    Callable augmentation pipeline. Given a (V, H, W, 3) float32 array in [0, 1]
    (V views of the same sample), applies one set of randomly-sampled parameters
    identically across all V views, and returns the augmented (V, H, W, 3) array.

    Techniques (each independently toggleable, per spec section 9):
        1. Background synthesis    -- replace transparent/uniform bg with random solid/gradient color
        2. Color jitter             -- brightness/contrast/saturation
        3. Random scale/zoom + re-padding within frame
        4. Horizontal flip only (no vertical flip, no 90/180 rotation -- furniture has real orientation)
        5. Light rotation (+/- max_rotation_degrees)
    """

    def __init__(self, config, rng: np.random.Generator | None = None):
        """
        Args:
            config: an AugmentationConfig instance (config.py).
            rng: optional numpy Generator for reproducibility control independent
                of the global numpy random state (useful for DataLoader workers,
                see seed.py's seed_worker).
        """
        self.config = config
        self.rng = rng or np.random.default_rng()

    def _sample_params(self) -> SampledAugmentationParams:
        c = self.config
        params = SampledAugmentationParams()

        if c.use_background_synthesis:
            # Random solid background color; gradient synthesis is a simple two-color
            # lerp applied at render-composite time (kept simple/solid here since the
            # base renders already come with a uniform/transparent background per spec).
            params.background_color = tuple(self.rng.uniform(0.0, 1.0, size=3).tolist())

        if c.use_color_jitter:
            s = c.color_jitter_strength
            params.brightness_factor = float(self.rng.uniform(1 - s, 1 + s))
            params.contrast_factor = float(self.rng.uniform(1 - s, 1 + s))
            params.saturation_factor = float(self.rng.uniform(1 - s, 1 + s))

        if c.use_random_scale:
            lo, hi = c.scale_range
            params.scale_factor = float(self.rng.uniform(lo, hi))

        if c.use_horizontal_flip:
            params.do_horizontal_flip = bool(self.rng.random() < c.horizontal_flip_prob)

        if c.use_rotation:
            params.rotation_degrees = float(
                self.rng.uniform(-c.max_rotation_degrees, c.max_rotation_degrees)
            )

        return params

    def __call__(self, views: np.ndarray) -> np.ndarray:
        """
        Args:
            views: (V, H, W, 3) float32 array in [0, 1].
        Returns:
            (V, H, W, 3) float32 array, same shape, augmented.
        """
        params = self._sample_params()
        out = views.copy()

        for v in range(out.shape[0]):
            out[v] = self._apply_to_single_view(out[v], params)

        return out

    def _apply_to_single_view(self, img: np.ndarray, params: SampledAugmentationParams) -> np.ndarray:
        if params.background_color is not None:
            img = self._apply_background(img, params.background_color)

        if self.config.use_color_jitter:
            img = self._apply_color_jitter(
                img, params.brightness_factor, params.contrast_factor, params.saturation_factor
            )

        if self.config.use_random_scale:
            img = self._apply_scale(img, params.scale_factor)

        if params.do_horizontal_flip:
            img = img[:, ::-1, :]

        if self.config.use_rotation:
            img = self._apply_rotation(img, params.rotation_degrees)

        return np.clip(img, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _apply_background(img: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
        """
        Composite over a solid background where the render is near-uniform/transparent.
        Approximation: treat near-white or near-black uniform regions as background
        (renders are already background-removed upstream per spec section 9's premise).
        For a production system this would use an actual alpha channel; this
        implementation assumes RGB-only input (no alpha) and blends based on a
        luminance threshold as a reasonable placeholder -- flagged clearly here
        for future refinement once real alpha-channel renders are available.
        """
        # NOTE: placeholder heuristic; replace with proper alpha compositing once
        # renders/ includes an alpha channel. Currently a light blend toward the
        # synthetic background color to diversify backgrounds without needing alpha.
        bg = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        blend_strength = 0.15  # keep subtle since we lack a real alpha mask
        return img * (1 - blend_strength) + bg * blend_strength

    @staticmethod
    def _apply_color_jitter(img: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
        img = img * brightness

        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * contrast + mean

        gray = img.mean(axis=2, keepdims=True)
        img = gray + (img - gray) * saturation

        return img

    @staticmethod
    def _apply_scale(img: np.ndarray, scale_factor: float) -> np.ndarray:
        """Zoom in/out then re-pad/crop back to original size, centered."""
        h, w = img.shape[:2]
        new_h, new_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))

        # simple nearest-neighbor resize via index mapping (no extra deps needed here;
        # actual pipeline can swap in PIL/cv2 resize for higher quality)
        row_idx = np.clip((np.arange(new_h) * h / new_h).astype(int), 0, h - 1)
        col_idx = np.clip((np.arange(new_w) * w / new_w).astype(int), 0, w - 1)
        resized = img[row_idx][:, col_idx]

        out = np.zeros_like(img)
        if scale_factor >= 1.0:
            # crop centered region back to (h, w)
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            out = resized[start_h:start_h + h, start_w:start_w + w]
        else:
            # pad centered into (h, w)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            out[start_h:start_h + new_h, start_w:start_w + new_w] = resized

        return out

    @staticmethod
    def _apply_rotation(img: np.ndarray, degrees: float) -> np.ndarray:
        """Light rotation via PIL (imported lazily to keep numpy-only path light)."""
        from PIL import Image
        h, w = img.shape[:2]
        pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        rotated = pil_img.rotate(degrees, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
        return np.asarray(rotated, dtype=np.float32) / 255.0
