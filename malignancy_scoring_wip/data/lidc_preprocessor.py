"""
lidc_preprocessor.py — Extract 3D patches and radiomics from LIDC-IDRI.

This script is a TEMPORARY Bloc 2 substitute used only until the real Bloc 2
(nnU-Net segmentation + PyRadiomics pipeline) is ready. It produces .npz files
in the same format that Bloc 2 will eventually produce, so the Bloc 3 dataset
class requires no changes when the handshake with Bloc 2 is done.

Output format per nodule (matches Bloc 2 output spec from Section 11.1):
    {
        nodule_id:       str,          unique identifier: {patient_id}_{nodule_idx}
        patient_id:      str,          LIDC-IDRI patient identifier
        patch_3d:        float32 arr,  shape (64, 64, 64), values in [0, 1]
        radiomics:       float32 arr,  shape (93,)
        mask_3d:         uint8 arr,    shape (64, 64, 64), binary consensus mask
        confidence_seg:  float,        1.0 (ground truth, not nnU-Net output)
        location:        [x, y, z],    nodule centroid in voxel coords
        malignancy_score: float,       mean of 4 radiologist scores (1–5)
        binary_label:    int,          set later by splits.assign_all_binary_labels
    }

Usage:
    python scripts/preprocess_lidc.py --lidc_root /path/to/LIDC-IDRI
                                      --output_dir data/preprocessed
                                      --patch_size 64
"""

import os
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# PyRadiomics for radiomic feature extraction
import radiomics
from radiomics import featureextractor

# pylidc for LIDC-IDRI annotation parsing
try:
    import pylidc as pl
    PYLIDC_AVAILABLE = True
except ImportError:
    PYLIDC_AVAILABLE = False
    print("[WARNING] pylidc not installed. Run: pip install pylidc")


# ─── Hounsfield Unit constants ────────────────────────────────────────────────

# Lung window: standard clinical window for pulmonary CT reading
HU_MIN = -1000   # air
HU_MAX =   400   # soft tissue / mild calcification

PATCH_SIZE = 64   # voxels per side (D, H, W)


# ─── PyRadiomics configuration ────────────────────────────────────────────────

# We configure PyRadiomics to extract exactly 93 features.
# Feature groups: first-order statistics, shape (3D), GLCM texture.
RADIOMICS_PARAMS = {
    "imageType":      {"Original": {}},
    "featureClass": {
        "firstorder":  {},   # 18 features — intensity statistics
        "shape":       {},   # 14 features — 3D geometric descriptors
        "glcm":        {},   # 24 features — grey-level co-occurrence matrix
        "glrlm":       {},   # 16 features — grey-level run-length matrix
        "glszm":       {},   # 16 features — grey-level size-zone matrix
        "ngtdm":       {},   # 5  features — neighbourhood grey-tone diff matrix
    },
    # Required settings for 3D patches
    "setting": {
        "binWidth":           25,
        "resampledPixelSpacing": None,   # already resampled in preprocessing
        "interpolator":       "sitkBSpline",
        "verbose":            False,
    },
}


# ─── Main preprocessing function ─────────────────────────────────────────────

def preprocess_lidc(
    lidc_root:    str,
    output_dir:   str,
    patch_size:   int  = PATCH_SIZE,
    min_nodule_mm: float = 3.0,
    verbose:      bool = True,
) -> List[Dict]:
    """
    Iterate over all LIDC-IDRI patients, extract nodule patches and radiomics,
    save each nodule as a .npz file.

    Args:
        lidc_root:     Path to LIDC-IDRI dataset root.
        output_dir:    Directory where .npz files will be written.
        patch_size:    Cubic patch side length in voxels.
        min_nodule_mm: Skip nodules smaller than this (millimeters).
        verbose:       Print progress.

    Returns:
        List of metadata dicts (one per nodule) — also saved as metadata.json.
    """
    if not PYLIDC_AVAILABLE:
        raise RuntimeError("pylidc is required. Install with: pip install pylidc")

    os.makedirs(output_dir, exist_ok=True)

    # Configure pylidc to find the dataset
    import pylidc as pl
    pl.query(pl.Scan)   # warm up connection

    extractor = _build_radiomics_extractor()
    metadata  = []

    scans = pl.query(pl.Scan).all()
    if verbose:
        print(f"[Preprocessor] Found {len(scans)} scans in LIDC-IDRI.")

    for scan in tqdm(scans, desc="Processing scans", disable=not verbose):
        patient_id = scan.patient_id

        # Load the CT volume as a numpy array with voxel spacing info
        try:
            vol, spacing = _load_scan_volume(scan)
        except Exception as e:
            print(f"[WARN] Could not load scan {patient_id}: {e}")
            continue

        # Apply lung window clipping and normalize to [0, 1]
        vol_norm = _clip_and_normalize(vol)

        # Get nodule clusters (each cluster = one nodule agreed upon by >= 3
        # of the 4 radiologists)
        nodule_clusters = scan.cluster_annotations(clevel=0.5)

        for nidx, annotations in enumerate(nodule_clusters):
            # Require at least 3 radiologists to have annotated the nodule
            if len(annotations) < 3:
                continue

            # Average malignancy score across radiologists (each rates 1–5)
            mal_scores = [ann.malignancy for ann in annotations
                          if ann.malignancy is not None]
            if not mal_scores:
                continue
            mean_score = float(np.mean(mal_scores))

            # Build consensus segmentation mask by majority vote
            consensus_mask = _majority_vote_mask(annotations, vol.shape)

            # Compute nodule centroid from mask
            centroid = _mask_centroid(consensus_mask)
            if centroid is None:
                continue

            # Skip sub-threshold nodules (< 3 mm)
            nodule_diam_mm = _estimate_diameter_mm(consensus_mask, spacing)
            if nodule_diam_mm < min_nodule_mm:
                continue

            # Extract cubic patch centered on centroid
            patch, mask_patch = _extract_patch(vol_norm, consensus_mask,
                                               centroid, patch_size)
            if patch is None:
                continue

            # Compute radiomic features from patch + mask
            try:
                radio_vec = _extract_radiomics(extractor, patch, mask_patch, spacing)
            except Exception as e:
                print(f"[WARN] Radiomics failed for {patient_id} nodule {nidx}: {e}")
                continue

            nodule_id = f"{patient_id}_{nidx:03d}"

            # Save as .npz (same format Bloc 2 will eventually produce)
            npz_path = os.path.join(output_dir, f"{nodule_id}.npz")
            np.savez_compressed(
                npz_path,
                patch_3d        = patch.astype(np.float32),
                mask_3d         = mask_patch.astype(np.uint8),
                radiomics       = radio_vec.astype(np.float32),
                location        = np.array(centroid, dtype=np.float32),
                confidence_seg  = np.float32(1.0),   # ground truth, not nnU-Net
            )

            meta = {
                "nodule_id":        nodule_id,
                "patient_id":       patient_id,
                "npz_path":         npz_path,
                "malignancy_score": mean_score,
                "binary_label":     None,  # assigned later by splits module
                "location":         list(centroid),
                "nodule_diam_mm":   nodule_diam_mm,
                "num_annotators":   len(annotations),
            }
            metadata.append(meta)

    # Save metadata index
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"[Preprocessor] Done. {len(metadata)} nodules saved to {output_dir}")
        print(f"[Preprocessor] Metadata index: {meta_path}")

    return metadata


# ─── Helper functions ─────────────────────────────────────────────────────────

def _load_scan_volume(scan) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a pylidc scan as a 3D numpy array with voxel spacing."""
    vol     = scan.to_volume()                # (D, H, W), raw HU values
    spacing = (scan.slice_thickness,          # z-spacing (mm)
               scan.pixel_spacing,            # y-spacing (mm)
               scan.pixel_spacing)            # x-spacing (mm)
    return vol.astype(np.float32), spacing


def _clip_and_normalize(vol: np.ndarray) -> np.ndarray:
    """
    Clip HU values to the lung window [HU_MIN, HU_MAX] and normalize to [0, 1].
    Clipping removes bone and air extremes that are irrelevant for nodule analysis.
    """
    vol = np.clip(vol, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)
    return vol.astype(np.float32)


def _majority_vote_mask(
    annotations: list,
    vol_shape:   Tuple[int, ...]
) -> np.ndarray:
    """
    Create a consensus binary mask by majority vote across radiologist annotations.
    A voxel is considered part of the nodule if >= 50% of annotators marked it.
    """
    vote_sum = np.zeros(vol_shape, dtype=np.float32)
    for ann in annotations:
        mask = ann.boolean_mask(pad=[(0, 0), (0, 0), (0, 0)])
        # Each annotation mask may have a bounding box offset
        bbox  = ann.bbox()
        slices = tuple(slice(b.start, b.stop) for b in bbox)
        vote_sum[slices] += mask.astype(np.float32)

    consensus = (vote_sum / len(annotations)) >= 0.5
    return consensus.astype(np.uint8)


def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Return the integer centroid (z, y, x) of a binary mask."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    centroid = coords.mean(axis=0)
    return tuple(int(c) for c in centroid)


def _estimate_diameter_mm(
    mask:    np.ndarray,
    spacing: Tuple[float, float, float]
) -> float:
    """
    Approximate nodule diameter in mm from mask volume.
    Assumes a sphere: d = 2 * (3V / 4π)^(1/3).
    """
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    nodule_vol_mm3 = mask.sum() * voxel_vol_mm3
    if nodule_vol_mm3 <= 0:
        return 0.0
    radius_mm = (3 * nodule_vol_mm3 / (4 * np.pi)) ** (1 / 3)
    return 2.0 * radius_mm


def _extract_patch(
    vol:       np.ndarray,
    mask:      np.ndarray,
    centroid:  Tuple[int, int, int],
    size:      int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract a cubic patch of shape (size, size, size) centered on centroid.
    Returns (None, None) if the centroid is too close to the volume boundary.
    """
    half = size // 2
    z, y, x = centroid

    # Check bounds — skip nodules too close to volume edge
    D, H, W = vol.shape
    if (z - half < 0 or z + half >= D or
            y - half < 0 or y + half >= H or
            x - half < 0 or x + half >= W):
        return None, None

    patch      = vol [z - half : z + half, y - half : y + half, x - half : x + half]
    mask_patch = mask[z - half : z + half, y - half : y + half, x - half : x + half]

    return patch.copy(), mask_patch.copy()


def _build_radiomics_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    """Create and configure the PyRadiomics feature extractor."""
    import logging
    # Suppress verbose PyRadiomics output
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.loadParams(params=RADIOMICS_PARAMS)
    return extractor


def _extract_radiomics(
    extractor:  featureextractor.RadiomicsFeatureExtractor,
    patch:      np.ndarray,
    mask_patch: np.ndarray,
    spacing:    Tuple[float, float, float],
) -> np.ndarray:
    """
    Compute radiomic features for a single nodule patch.

    Returns:
        np.ndarray of shape (93,) with numerical feature values.
    """
    # Convert numpy arrays to SimpleITK images with correct spacing
    patch_itk = sitk.GetImageFromArray(patch)
    patch_itk.SetSpacing([spacing[2], spacing[1], spacing[0]])  # (x, y, z)

    mask_itk = sitk.GetImageFromArray(mask_patch.astype(np.uint8))
    mask_itk.SetSpacing([spacing[2], spacing[1], spacing[0]])

    result = extractor.execute(patch_itk, mask_itk, label=1)

    # Keep only numerical features (exclude metadata strings)
    features = [
        float(v)
        for k, v in result.items()
        if not k.startswith("diagnostics_")
    ]

    return np.array(features, dtype=np.float32)


# ─── Load saved metadata ──────────────────────────────────────────────────────

def load_metadata(preprocessed_dir: str) -> List[Dict]:
    """
    Load the nodule metadata index produced by preprocess_lidc().

    Args:
        preprocessed_dir: Directory where preprocessing output was saved.

    Returns:
        List of nodule metadata dicts.
    """
    meta_path = os.path.join(preprocessed_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"metadata.json not found in {preprocessed_dir}. "
            "Run scripts/preprocess_lidc.py first."
        )
    with open(meta_path) as f:
        return json.load(f)
