import os
import json
import warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

warnings.filterwarnings("ignore")

try:
    from radiomics import featureextractor as _fx
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False
    print("WARNING: pyradiomics not installed. Radiomics features will be skipped.")

PATCH_SIZE = 64


def load_volume(path):
    img = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(img).astype(np.float32)
    return img, array


def load_mask(path):
    img = sitk.ReadImage(path)
    array = (sitk.GetArrayFromImage(img) > 0).astype(np.uint8)
    return img, array


def extract_patch(ct_array, mask_array, centroid_zyx, size=64):
    half = size // 2
    z, y, x = [int(round(c)) for c in centroid_zyx]

    pad = half + 2
    ct_pad = np.pad(ct_array, pad, mode="constant", constant_values=-1000)
    mask_pad = np.pad(mask_array, pad, mode="constant", constant_values=0)

    zp, yp, xp = z + pad, y + pad, x + pad

    patch = ct_pad[zp - half:zp + half, yp - half:yp + half, xp - half:xp + half]
    pmask = mask_pad[zp - half:zp + half, yp - half:yp + half, xp - half:xp + half]

    start_zyx = (z - half, y - half, x - half)
    return patch.astype(np.float32), pmask.astype(np.uint8), start_zyx


def array_to_nifti(array, reference_sitk, output_path, origin=None):
    out = sitk.GetImageFromArray(array)
    out.SetSpacing(reference_sitk.GetSpacing())
    out.SetDirection(reference_sitk.GetDirection())
    if origin is not None:
        out.SetOrigin(origin)
    else:
        out.SetOrigin(reference_sitk.GetOrigin())
    sitk.WriteImage(out, output_path)


def compute_geometry(component_mask, spacing_xyz):
    sx, sy, sz = spacing_xyz

    voxel_count = int(component_mask.sum())
    volume_mm3 = float(voxel_count * sx * sy * sz)

    centroid_zyx = ndimage.center_of_mass(component_mask)

    zz, yy, xx = np.where(component_mask)

    z_extent_mm = float((zz.max() - zz.min() + 1) * sz)
    y_extent_mm = float((yy.max() - yy.min() + 1) * sy)
    x_extent_mm = float((xx.max() - xx.min() + 1) * sx)

    diameter_mm = max(z_extent_mm, y_extent_mm, x_extent_mm)

    equiv_radius_mm = (3 * volume_mm3 / (4 * np.pi)) ** (1 / 3) if volume_mm3 > 0 else 0
    equiv_volume = (4 / 3) * np.pi * (equiv_radius_mm ** 3)
    sphericity = float(equiv_volume / volume_mm3) if volume_mm3 > 0 else 0

    return {
        "centroid_z_vox": float(centroid_zyx[0]),
        "centroid_y_vox": float(centroid_zyx[1]),
        "centroid_x_vox": float(centroid_zyx[2]),
        "voxel_count": voxel_count,
        "volume_mm3": round(volume_mm3, 3),
        "diameter_mm": round(diameter_mm, 3),
        "z_extent_mm": round(z_extent_mm, 3),
        "y_extent_mm": round(y_extent_mm, 3),
        "x_extent_mm": round(x_extent_mm, 3),
        "equiv_radius_mm": round(equiv_radius_mm, 3),
        "sphericity": round(sphericity, 4),
    }


def compute_hu_stats(ct_array, component_mask):
    voxels = ct_array[component_mask > 0].astype(np.float64)
    if len(voxels) == 0:
        return {}
    return {
        "hu_mean": round(float(voxels.mean()), 2),
        "hu_std": round(float(voxels.std()), 2),
        "hu_min": round(float(voxels.min()), 2),
        "hu_max": round(float(voxels.max()), 2),
        "hu_median": round(float(np.median(voxels)), 2),
        "hu_p25": round(float(np.percentile(voxels, 25)), 2),
        "hu_p75": round(float(np.percentile(voxels, 75)), 2),
    }


def compute_radiomics(ct_sitk, mask_sitk, label_value=1):
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


def process_nodule(nod_idx, component_mask, ct_array, ct_sitk, mask_sitk_full, spacing_xyz, output_dir):
    nod_id = f"nodule_{nod_idx:03d}"
    nod_dir = os.path.join(output_dir, nod_id)
    os.makedirs(nod_dir, exist_ok=True)

    print(f"  [{nod_id}] extracting features...")

    geo = compute_geometry(component_mask, spacing_xyz)
    centroid_zyx = (geo["centroid_z_vox"], geo["centroid_y_vox"], geo["centroid_x_vox"])
    hu = compute_hu_stats(ct_array, component_mask)

    features = {"nodule_id": nod_id, **geo, **hu}
    with open(os.path.join(nod_dir, "features.json"), "w") as f:
        json.dump(features, f, indent=2)

    patch, pmask, start_zyx = extract_patch(ct_array, component_mask, centroid_zyx, PATCH_SIZE)

    np.save(os.path.join(nod_dir, "patch_3d.npy"), patch)
    np.save(os.path.join(nod_dir, "mask_3d.npy"), pmask)

    sz, sy, sx = start_zyx
    patch_origin = ct_sitk.TransformIndexToPhysicalPoint((sx, sy, sz))

    array_to_nifti(patch, ct_sitk, os.path.join(nod_dir, "patch_3d.nii.gz"), origin=patch_origin)
    array_to_nifti(pmask, mask_sitk_full, os.path.join(nod_dir, "mask_3d.nii.gz"), origin=patch_origin)

    iso_mask_arr = component_mask.astype(np.uint8)
    iso_mask_img = sitk.GetImageFromArray(iso_mask_arr)
    iso_mask_img.CopyInformation(ct_sitk)

    radio, radio_diag = compute_radiomics(ct_sitk, iso_mask_img, label_value=1)

    with open(os.path.join(nod_dir, "radiomics.json"), "w") as f:
        json.dump(radio, f, indent=2)
    if radio_diag:
        with open(os.path.join(nod_dir, "radiomics_diagnostics.json"), "w") as f:
            json.dump(radio_diag, f, indent=2)

    print(f"  [{nod_id}] done — vol={geo['volume_mm3']:.1f}mm3 diam={geo['diameter_mm']:.1f}mm")

    flat_radio = {}
    for k, v in radio.items():
        flat_radio[k] = json.dumps(v) if isinstance(v, list) else v

    return {**features, **flat_radio}


def extract_all_features(volume_path, mask_path, output_dir, min_voxels=10):
    """
    Extract per-nodule features from CT volume + filtered nodule mask.
    Returns list of nodule feature dicts and saves per-nodule files + summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Loading volume: {volume_path}")
    ct_sitk, ct_array = load_volume(volume_path)
    spacing_xyz = ct_sitk.GetSpacing()

    print(f"  Loading mask: {mask_path}")
    mask_sitk, mask_array = load_mask(mask_path)

    if ct_array.shape != mask_array.shape:
        raise ValueError(f"Volume shape {ct_array.shape} != mask shape {mask_array.shape}")

    labeled, n_components = ndimage.label(mask_array)
    print(f"  Found {n_components} connected component(s)")

    all_rows = []
    nod_idx = 0

    for comp_label in range(1, n_components + 1):
        component = (labeled == comp_label)
        voxel_count = int(component.sum())

        if voxel_count < min_voxels:
            print(f"  [component {comp_label}] {voxel_count} vox < {min_voxels} — skipping")
            continue

        nod_idx += 1
        row = process_nodule(
            nod_idx=nod_idx,
            component_mask=component,
            ct_array=ct_array,
            ct_sitk=ct_sitk,
            mask_sitk_full=mask_sitk,
            spacing_xyz=spacing_xyz,
            output_dir=output_dir,
        )
        all_rows.append(row)

    summary_json_path = os.path.join(output_dir, "summary.json")
    summary_csv_path = os.path.join(output_dir, "summary.csv")

    with open(summary_json_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(summary_csv_path, index=False)

    print(f"  Valid nodules: {nod_idx} (of {n_components} components)")
    return all_rows
