"""
augmentation.py — Safe 3D augmentation for pulmonary nodule patches.

GOLDEN RULE: Safe augmentations preserve shape, texture, density (HU values),
and margins. Unsafe augmentations distort any of these.

Applied ONLY to malignant-class patches during training to address the
class imbalance in LIDC-IDRI (majority of nodules are benign).

Safety classification (from design document):
  SAFE      — horizontal flip, vertical flip, 90° rotations, small rotations
               (±15°), small translations
  CAUTION   — small Gaussian noise (preserves HU if std is tiny),
               small zoom (preserves proportions if range is tight)
  UNSAFE    — elastic deformation (distorts shape, a key malignancy indicator),
               strong brightness/contrast (alters HU values),
               cutout/masking (removes critical feature information)
"""

import numpy as np
from typing import Tuple


class Safe3DAugmentation:
    """
    Augmentation pipeline for 3D CT nodule patches (D, H, W).

    All transforms preserve clinically meaningful features.
    Each transform is applied independently with its own probability.
    """

    def __init__(
        self,
        flip_prob:       float = 0.5,
        rotate90_prob:   float = 0.5,
        small_rot_prob:  float = 0.4,
        translate_prob:  float = 0.3,
        noise_prob:      float = 0.3,
        noise_std:       float = 0.01,   # Very small — must not shift HU values
        zoom_prob:       float = 0.2,
        zoom_range:      Tuple[float, float] = (0.9, 1.1),  # Tight range only
        random_seed:     int = None,
    ):
        self.flip_prob      = flip_prob
        self.rotate90_prob  = rotate90_prob
        self.small_rot_prob = small_rot_prob
        self.translate_prob = translate_prob
        self.noise_prob     = noise_prob
        self.noise_std      = noise_std
        self.zoom_prob      = zoom_prob
        self.zoom_range     = zoom_range
        self.rng            = np.random.default_rng(random_seed)

    def __call__(self, patch: np.ndarray) -> np.ndarray:
        """
        Apply a random subset of safe augmentations to a 3D patch.

        Args:
            patch: np.ndarray of shape (D, H, W), values in [0, 1] after
                   HU normalization.

        Returns:
            Augmented patch with same shape (D, H, W).
        """
        patch = patch.copy()

        # ── SAFE: axis-aligned flips ──────────────────────────────────────────
        if self.rng.random() < self.flip_prob:
            patch = np.flip(patch, axis=2).copy()   # left-right
        if self.rng.random() < self.flip_prob:
            patch = np.flip(patch, axis=1).copy()   # anterior-posterior
        if self.rng.random() < self.flip_prob:
            patch = np.flip(patch, axis=0).copy()   # superior-inferior

        # ── SAFE: 90° in-plane rotations ──────────────────────────────────────
        if self.rng.random() < self.rotate90_prob:
            k = self.rng.integers(1, 4)  # 90, 180, or 270 degrees
            patch = np.rot90(patch, k=k, axes=(1, 2)).copy()

        # ── SAFE: small random rotation (±15°) ───────────────────────────────
        if self.rng.random() < self.small_rot_prob:
            patch = self._small_rotation(patch, max_angle=15.0)

        # ── SAFE: small spatial translation ───────────────────────────────────
        if self.rng.random() < self.translate_prob:
            patch = self._small_translation(patch, max_shift=3)

        # ── CAUTION: small Gaussian noise ────────────────────────────────────
        # Simulates scanner acquisition noise. std=0.01 on normalized [0,1]
        # data corresponds to ~10 HU — clinically negligible.
        if self.rng.random() < self.noise_prob:
            noise = self.rng.normal(0.0, self.noise_std, patch.shape).astype(np.float32)
            patch = np.clip(patch + noise, 0.0, 1.0)

        # ── CAUTION: small zoom ───────────────────────────────────────────────
        # Tight zoom range (0.9–1.1) preserves relative nodule proportions.
        if self.rng.random() < self.zoom_prob:
            patch = self._small_zoom(patch)

        return patch

    # ── Private helpers ───────────────────────────────────────────────────────

    def _small_rotation(self, patch: np.ndarray, max_angle: float) -> np.ndarray:
        """Rotate in the axial (H, W) plane by a random small angle."""
        from scipy.ndimage import rotate
        angle = self.rng.uniform(-max_angle, max_angle)
        # order=1 (bilinear) balances speed and quality; mode='nearest' avoids
        # black borders that would introduce artificial HU boundaries.
        patch = rotate(patch, angle, axes=(1, 2), reshape=False,
                       order=1, mode="nearest")
        return patch

    def _small_translation(self, patch: np.ndarray, max_shift: int) -> np.ndarray:
        """Shift the nodule center slightly within the patch volume."""
        from scipy.ndimage import shift
        shifts = self.rng.integers(-max_shift, max_shift + 1, size=3)
        patch = shift(patch, shifts.tolist(), order=1, mode="nearest")
        return patch

    def _small_zoom(self, patch: np.ndarray) -> np.ndarray:
        """Zoom in or out slightly, then crop/pad back to original shape."""
        from scipy.ndimage import zoom as scipy_zoom
        factor = self.rng.uniform(self.zoom_range[0], self.zoom_range[1])
        original_shape = patch.shape
        zoomed = scipy_zoom(patch, factor, order=1)
        return self._match_shape(zoomed, original_shape)

    @staticmethod
    def _match_shape(arr: np.ndarray, target: tuple) -> np.ndarray:
        """
        Center-crop or center-pad arr to match target shape along each axis.
        Used after zoom to restore the fixed 64×64×64 patch size.
        """
        for dim in range(3):
            current = arr.shape[dim]
            want    = target[dim]
            if current > want:
                # Center crop
                start = (current - want) // 2
                idx   = range(start, start + want)
                arr   = np.take(arr, idx, axis=dim)
            elif current < want:
                # Center pad with edge values (avoids hard HU boundaries)
                pad_before = (want - current) // 2
                pad_after  = want - current - pad_before
                pad_width  = [(0, 0)] * 3
                pad_width[dim] = (pad_before, pad_after)
                arr = np.pad(arr, pad_width, mode="edge")
        return arr
