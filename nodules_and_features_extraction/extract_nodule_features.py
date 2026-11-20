"""
extract_nodule_features.py
==========================
Given a 3D CT volume and its binary nodule mask, extracts every nodule
as an individual entity and computes a full feature set for each one.

Usage
-----
  python extract_nodule_features.py \
      --volume  path/to/ct_volume.nii.gz \
      --mask    path/to/nodule_mask.nii.gz \
      --output  path/to/output_folder

Output folder structure
-----------------------
  output_folder/
    nodule_001/
      patch_3d.npy          64x64x64 CT crop  (float32, HU values)
      mask_3d.npy           64x64x64 binary mask crop  (uint8)
      patch_3d.nii.gz       same crop as NIfTI (for visual inspection)
      mask_3d.nii.gz        same mask crop as NIfTI
      features.json         all scalar features (geometry + HU stats)
      radiomics.json        PyRadiomics feature vector (~93-107 features)
    nodule_002/
      ...
    summary.json            one row per nodule — all features combined
    summary.csv             same as summary.json but CSV format
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

warnings.filterwarnings("ignore")

# ── radiomics is optional — warn clearly if missing ──────────────────
try:
    from radiomics import featureextractor as _fx
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False
    print("WARNING: pyradiomics not installed. Radiomics features will be skipped.")
    print("         Install with: pip install pyradiomics")

PATCH_SIZE = 64   # 64x64x64 voxel cube per nodule


# ── helpers ───────────────────────────────────────────────────────────

def load_volume(path):
    """Load NIfTI → SimpleITK image + numpy array (Z,Y,X)"""
    img   = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(img).astype(np.float32)
    return img, array


def load_mask(path):
    """Load binary mask NIfTI → SimpleITK image + uint8 numpy (Z,Y,X)"""
    img   = sitk.ReadImage(path)
    array = (sitk.GetArrayFromImage(img) > 0).astype(np.uint8)
    return img, array


def extract_patch(ct_array, mask_array, centroid_zyx, size=64):
    """
    Crop a cubic patch of `size` voxels centered on centroid_zyx.
    Pads with -1000 HU (air) for CT and 0 for mask if near the edge.

    Returns
    -------
    patch : np.ndarray  shape (size, size, size) float32
    pmask : np.ndarray  shape (size, size, size) uint8
    """
    half = size // 2
    z, y, x = [int(round(c)) for c in centroid_zyx]

    pad = half + 2
    ct_pad   = np.pad(ct_array,   pad, mode='constant', constant_values=-1000)
    mask_pad = np.pad(mask_array, pad, mode='constant', constant_values=0)

    zp, yp, xp = z + pad, y + pad, x + pad

    patch = ct_pad  [zp-half:zp+half, yp-half:yp+half, xp-half:xp+half]
    pmask = mask_pad[zp-half:zp+half, yp-half:yp+half, xp-half:xp+half]

    assert patch.shape == (size, size, size), \
        f"Patch shape mismatch: {patch.shape}"

    start_zyx = (z - half, y - half, x - half)
    return patch.astype(np.float32), pmask.astype(np.uint8), start_zyx


def array_to_nifti(array, reference_sitk, output_path, origin=None):
    """Save numpy array as NIfTI with geometry from reference image.
    
    If origin is provided, it is used (needed for patches smaller than
    the full volume).  Otherwise the reference origin is copied.
    """
    out = sitk.GetImageFromArray(array)
    out.SetSpacing(reference_sitk.GetSpacing())
    out.SetDirection(reference_sitk.GetDirection())
    if origin is not None:
        out.SetOrigin(origin)
    else:
        out.SetOrigin(reference_sitk.GetOrigin())
    sitk.WriteImage(out, output_path)


def compute_geometry(component_mask, spacing_xyz):
    """
    Compute geometric features from a binary component mask.

    Parameters
    ----------
    component_mask : np.ndarray  (Z,Y,X) bool
    spacing_xyz    : tuple  (x_mm, y_mm, z_mm)  SimpleITK convention

    Returns
    -------
    dict of scalar geometric features
    """
    sx, sy, sz = spacing_xyz   # SimpleITK: x=col, y=row, z=slice

    voxel_count = int(component_mask.sum())
    volume_mm3  = float(voxel_count * sx * sy * sz)

    centroid_zyx = ndimage.center_of_mass(component_mask)

    zz, yy, xx = np.where(component_mask)

    # Bounding box extents in mm
    z_extent_mm = float((zz.max() - zz.min() + 1) * sz)
    y_extent_mm = float((yy.max() - yy.min() + 1) * sy)
    x_extent_mm = float((xx.max() - xx.min() + 1) * sx)

    # Approximate diameter = max extent
    diameter_mm = max(z_extent_mm, y_extent_mm, x_extent_mm)

    # Approximate sphericity
    # V = 4/3 * pi * r^3  =>  r = (3V / 4pi)^(1/3)
    # Surface area approximated from voxel surface count
    equiv_radius_mm = (3 * volume_mm3 / (4 * np.pi)) ** (1/3) if volume_mm3 > 0 else 0
    equiv_volume    = (4/3) * np.pi * (equiv_radius_mm ** 3)
    sphericity      = float(equiv_volume / volume_mm3) if volume_mm3 > 0 else 0

    return {
        "centroid_z_vox"  : float(centroid_zyx[0]),
        "centroid_y_vox"  : float(centroid_zyx[1]),
        "centroid_x_vox"  : float(centroid_zyx[2]),
        "voxel_count"     : voxel_count,
        "volume_mm3"      : round(volume_mm3, 3),
        "diameter_mm"     : round(diameter_mm, 3),
        "z_extent_mm"     : round(z_extent_mm, 3),
        "y_extent_mm"     : round(y_extent_mm, 3),
        "x_extent_mm"     : round(x_extent_mm, 3),
        "equiv_radius_mm" : round(equiv_radius_mm, 3),
        "sphericity"      : round(sphericity, 4),
    }


def compute_hu_stats(ct_array, component_mask):
    """HU intensity statistics inside the nodule voxels."""
    voxels = ct_array[component_mask > 0].astype(np.float64)
    if len(voxels) == 0:
        return {}
    return {
        "hu_mean"   : round(float(voxels.mean()),   2),
        "hu_std"    : round(float(voxels.std()),    2),
        "hu_min"    : round(float(voxels.min()),    2),
        "hu_max"    : round(float(voxels.max()),    2),
        "hu_median" : round(float(np.median(voxels)), 2),
        "hu_p25"    : round(float(np.percentile(voxels, 25)), 2),
        "hu_p75"    : round(float(np.percentile(voxels, 75)), 2),
    }


def compute_radiomics(ct_sitk, mask_sitk, label_value=1):
    """
    Run PyRadiomics on the full volume with a single-label mask.
    Returns two dicts: (features, diagnostics).
    """
    if not RADIOMICS_AVAILABLE:
        return {}, {}

    settings = {
        "binWidth": 25,
        "interpolator": sitk.sitkBSpline,
        "resampledPixelSpacing": None,
        "textureSampleDistance": 1,
        "force2D": False,
        "force2Ddimension": 0,
        "minimumROIDimensions": 1,
        "minimumROISize": 2,
        "label": label_value,
    }

    extractor = _fx.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllFeatures()

    try:
        result = extractor.execute(ct_sitk, mask_sitk)
    except Exception as e:
        print(f"    WARNING: PyRadiomics failed: {e}")
        return {}, {}

    features = {}
    diagnostics = {}
    for k, v in result.items():
        if k.startswith("diagnostics"):
            if isinstance(v, np.ndarray):
                diagnostics[k] = v.tolist()
            elif isinstance(v, dict):
                diagnostics[k] = v
            else:
                try:
                    diagnostics[k] = float(v)
                except (TypeError, ValueError):
                    diagnostics[k] = str(v)
        elif isinstance(v, np.ndarray):
            if v.ndim == 0:
                features[k] = float(v.item())
            else:
                features[k] = v.tolist()
        elif np.isscalar(v):
            try:
                features[k] = float(v)
            except (TypeError, ValueError):
                pass

    return features, diagnostics


# ── per-nodule processing ─────────────────────────────────────────────

def process_nodule(
    nod_idx,
    component_mask,
    ct_array,
    ct_sitk,
    mask_sitk_full,
    spacing_xyz,
    output_dir,
):
    """
    Extract everything for a single nodule component.

    Returns flat dict for the summary CSV/JSON.
    """
    nod_id  = f"nodule_{nod_idx:03d}"
    nod_dir = os.path.join(output_dir, nod_id)
    os.makedirs(nod_dir, exist_ok=True)

    print(f"  [{nod_id}] extracting features...")

    # ── geometry ─────────────────────────────────────────────────────
    geo = compute_geometry(component_mask, spacing_xyz)
    centroid_zyx = (
        geo["centroid_z_vox"],
        geo["centroid_y_vox"],
        geo["centroid_x_vox"],
    )

    # ── HU statistics ─────────────────────────────────────────────────
    hu  = compute_hu_stats(ct_array, component_mask)

    # ── scalar features JSON ──────────────────────────────────────────
    features = {"nodule_id": nod_id, **geo, **hu}
    with open(os.path.join(nod_dir, "features.json"), "w") as f:
        json.dump(features, f, indent=2)

    # ── 64³ patch extraction ──────────────────────────────────────────
    patch, pmask, start_zyx = extract_patch(ct_array, component_mask, centroid_zyx, PATCH_SIZE)

    np.save(os.path.join(nod_dir, "patch_3d.npy"), patch)
    np.save(os.path.join(nod_dir, "mask_3d.npy"),  pmask)

    # Compute physical origin of the patch so NIfTI aligns in Slicer
    sz, sy, sx = start_zyx
    patch_origin = ct_sitk.TransformIndexToPhysicalPoint((sx, sy, sz))

    # Save as NIfTI for visual inspection in 3D Slicer / ITK-SNAP
    array_to_nifti(patch, ct_sitk,        os.path.join(nod_dir, "patch_3d.nii.gz"), origin=patch_origin)
    array_to_nifti(pmask, mask_sitk_full,  os.path.join(nod_dir, "mask_3d.nii.gz"), origin=patch_origin)

    # ── PyRadiomics on the isolated nodule mask ───────────────────────
    # Build a single-label mask image for this nodule only
    iso_mask_arr = component_mask.astype(np.uint8)
    iso_mask_img = sitk.GetImageFromArray(iso_mask_arr)
    iso_mask_img.CopyInformation(ct_sitk)

    radio, radio_diag = compute_radiomics(ct_sitk, iso_mask_img, label_value=1)

    with open(os.path.join(nod_dir, "radiomics.json"), "w") as f:
        json.dump(radio, f, indent=2)
    if radio_diag:
        with open(os.path.join(nod_dir, "radiomics_diagnostics.json"), "w") as f:
            json.dump(radio_diag, f, indent=2)

    print(f"  [{nod_id}] done — "
          f"vol={geo['volume_mm3']:.1f}mm³  "
          f"diam={geo['diameter_mm']:.1f}mm  "
          f"HU_mean={hu.get('hu_mean','?')}  "
          f"radiomics={len(radio)} features")

    return {**features, **_flatten_radio_for_summary(radio)}


def _flatten_radio_for_summary(radio):
    """Flatten array-valued radiomics features so they fit in a CSV row."""
    flat = {}
    for k, v in radio.items():
        if isinstance(v, list):
            flat[k] = json.dumps(v)
        else:
            flat[k] = v
    return flat


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract per-nodule features from CT volume + binary mask"
    )
    parser.add_argument("--volume", required=True,
                        help="Path to 3D CT volume (.nii.gz)")
    parser.add_argument("--mask",   required=True,
                        help="Path to binary nodule mask (.nii.gz)")
    parser.add_argument("--output", required=True,
                        help="Output folder (created if not exists)")
    parser.add_argument("--min_voxels", type=int, default=10,
                        help="Minimum voxels to consider a component a real "
                             "nodule (default: 10). Smaller components are noise.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────
    print(f"\nLoading volume : {args.volume}")
    ct_sitk, ct_array = load_volume(args.volume)
    spacing_xyz = ct_sitk.GetSpacing()   # (x, y, z) in mm
    print(f"  Volume shape  : {ct_array.shape}  (Z, Y, X)")
    print(f"  Spacing       : {spacing_xyz}  mm  (x, y, z)")

    print(f"\nLoading mask   : {args.mask}")
    mask_sitk, mask_array = load_mask(args.mask)
    print(f"  Mask shape    : {mask_array.shape}")
    print(f"  Foreground vox: {mask_array.sum()}")

    if ct_array.shape != mask_array.shape:
        print(f"ERROR: Volume shape {ct_array.shape} != mask shape {mask_array.shape}")
        sys.exit(1)

    # ── connected components → individual nodules ─────────────────────
    print("\nRunning connected component analysis...")
    labeled, n_components = ndimage.label(mask_array)
    print(f"  Found {n_components} connected component(s)")

    # ── process each component ────────────────────────────────────────
    all_rows = []
    nod_idx  = 0

    for comp_label in range(1, n_components + 1):
        component = (labeled == comp_label)
        voxel_count = int(component.sum())

        if voxel_count < args.min_voxels:
            print(f"  [component {comp_label}] only {voxel_count} voxels "
                  f"< min {args.min_voxels} — skipping (noise)")
            continue

        nod_idx += 1
        row = process_nodule(
            nod_idx      = nod_idx,
            component_mask = component,
            ct_array     = ct_array,
            ct_sitk      = ct_sitk,
            mask_sitk_full = mask_sitk,
            spacing_xyz  = spacing_xyz,
            output_dir   = args.output,
        )
        all_rows.append(row)

    # ── summary files ─────────────────────────────────────────────────
    print(f"\nSaving summary files...")

    summary_json_path = os.path.join(args.output, "summary.json")
    summary_csv_path  = os.path.join(args.output, "summary.csv")

    with open(summary_json_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(summary_csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Volume           : {args.volume}")
    print(f"  Mask             : {args.mask}")
    print(f"  Valid nodules    : {nod_idx}  (of {n_components} components)")
    print(f"  Output folder    : {args.output}")
    print(f"  Summary JSON     : {summary_json_path}")
    print(f"  Summary CSV      : {summary_csv_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
