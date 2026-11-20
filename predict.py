"""
Inference Pipeline (Fold 1): DICOM -> Lung Mask -> Nodule Mask
================================================================
Self-contained single-fold inference using fold 1 model.

PHASE 1: DICOM to NIfTI conversion
PHASE 2: TotalSegmentator extracts lung mask (--ml --fast, 5 lobes)
PHASE 3: Re-load CT + mask, mask non-lung tissue to -1024 HU, crop
PHASE 4: Create overlapping 3D patches (80x192x192) from cropped lung CT
PHASE 5: nnU-Net fold 1 inference on all patches
PHASE 6: Stitch predictions (OR logic), apply lung mask constraint
PHASE 7: Map back to native CT space, save outputs

Usage:
  python predict.py --dicom_dir "C:/path/to/dicom" --case_id case_001
  python predict.py --dicom_dir "C:/path/to/dicom" --case_id case_001 --threshold 0.2

Outputs (in native CT space):
  {prefix}_lung_mask.nii.gz    - Binary lung mask
  {prefix}_nodule_mask.nii.gz   - Binary nodule mask
"""

import os
import sys
import argparse
import tempfile
import subprocess
import shutil
import numpy as np
import SimpleITK as sitk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

PATCH_SIZE = (80, 192, 192)
OVERLAP = (20, 48, 48)
DEFAULT_THRESHOLD = 0.15


def convert_dicom_to_nifti(dicom_dir, output_path):
    print("  Converting DICOM to NIfTI...")
    target_dir = dicom_dir
    for root, dirs, files in os.walk(dicom_dir):
        if any(f.endswith('.dcm') for f in files):
            target_dir = root
            break

    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(target_dir)
    if not series_IDs:
        raise ValueError(f"No DICOM series found in {target_dir}")

    largest_series = []
    for series_id in series_IDs:
        filenames = reader.GetGDCMSeriesFileNames(target_dir, series_id)
        if len(filenames) > len(largest_series):
            largest_series = filenames

    dicom_names = largest_series

    if not dicom_names:
        raise ValueError(f"No DICOM files found in {dicom_dir} or its subdirectories.")
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    sitk.WriteImage(img, output_path)
    num_slices = len(dicom_names)
    size = img.GetSize()
    spacing = img.GetSpacing()
    print(f"  Loaded {num_slices} slices")
    print(f"  Size: {size}  Spacing: {[round(float(s), 3) for s in spacing]}")


def run_totalsegmentator(input_nifti, output_file):
    cmd = [
        "TotalSegmentator",
        "-i", input_nifti,
        "-o", output_file,
        "--fast", "--ml",
        "-rs", "lung_upper_lobe_left", "lung_lower_lobe_left",
               "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
    ]
    print("  Running TotalSegmentator...")
    subprocess.run(cmd, check=True)
    print("  TotalSegmentator complete.")


def run_nnunet(input_dir, output_dir, fold):
    nnunet_exe = os.path.join(os.path.dirname(sys.executable), "nnUNet_predict")
    if os.name == "nt":
        nnunet_exe += ".exe"

    env = os.environ.copy()
    env["RESULTS_FOLDER"] = os.path.abspath(MODEL_DIR)

    cmd = [
        nnunet_exe,
        "-i", os.path.abspath(input_dir),
        "-o", os.path.abspath(output_dir),
        "-t", "Task001_NoduleDetector",
        "-m", "3d_fullres",
        "-f", str(fold),
        "--disable_tta",
        "--save_npz",
        "--num_threads_preprocessing", "1",
        "--num_threads_nifti_save", "1",
    ]
    print(f"  Running nnU-Net fold {fold} ({nnunet_exe})...")
    subprocess.run(cmd, env=env, check=True)


def generate_patch_coords(volume_shape, patch_size, overlap):
    D, H, W = volume_shape
    pz, py, px = patch_size
    oz, oy, ox = overlap

    def get_starts(dim_size, patch_dim, overlap_dim):
        stride = patch_dim - overlap_dim
        if stride <= 0:
            stride = 1
        starts = list(range(0, max(1, dim_size - patch_dim + 1), stride))
        if not starts or starts[-1] + patch_dim < dim_size:
            starts.append(max(0, dim_size - patch_dim))
        seen = set()
        unique = []
        for s in starts:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    z_starts = get_starts(D, pz, oz)
    y_starts = get_starts(H, py, oy)
    x_starts = get_starts(W, px, ox)

    coords = []
    for z0 in z_starts:
        z1 = min(z0 + pz, D)
        for y0 in y_starts:
            y1 = min(y0 + py, H)
            for x0 in x_starts:
                x1 = min(x0 + px, W)
                coords.append((z0, z1, y0, y1, x0, x1))
    return coords


def main():
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Inference Pipeline: DICOM -> Lung Mask -> Nodule Mask (Fold 1)")
    parser.add_argument("--dicom_dir", required=True, help="Folder containing DICOM .dcm slices")
    parser.add_argument("--case_id", required=True, help="Case ID (e.g. case_003). Outputs go to inference/results/")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Probability threshold for nodule detection (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    fold = 1

    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_prefix = os.path.join(results_dir, args.case_id)

    fold_dir = os.path.join(MODEL_DIR, "nnUNet", "3d_fullres", "Task001_NoduleDetector",
                            "nnUNetTrainerV2__nnUNetPlansv2.1", "fold_1")
    if not os.path.isdir(fold_dir):
        print(f"ERROR: Model not found at {fold_dir}")
        sys.exit(1)

    temp_dir = tempfile.mkdtemp()
    try:
        ct_nifti_path = os.path.join(temp_dir, "full_ct.nii.gz")
        lung_mask_path = os.path.join(temp_dir, "lung_mask.nii.gz")

        # ── PHASE 1: DICOM -> NIfTI ──────────────────────────────────
        print("=" * 60)
        print("PHASE 1: Converting DICOM to NIfTI")
        print("=" * 60)
        convert_dicom_to_nifti(args.dicom_dir, ct_nifti_path)

        # ── PHASE 2: TotalSegmentator -> Lung Mask ───────────────────
        print()
        print("=" * 60)
        print("PHASE 2: TotalSegmentator Lung Extraction")
        print("=" * 60)
        run_totalsegmentator(ct_nifti_path, lung_mask_path)

        # ── PHASE 3: Load CT + mask, crop, prepare ────────────────────
        print()
        print("=" * 60)
        print("PHASE 3: Loading results, masking and cropping")
        print("=" * 60)

        original_ct = sitk.ReadImage(ct_nifti_path)
        original_array = sitk.GetArrayFromImage(original_ct)
        original_spacing = original_ct.GetSpacing()
        original_direction = original_ct.GetDirection()

        lung_mask_img = sitk.ReadImage(lung_mask_path)
        combined_mask = sitk.GetArrayFromImage(lung_mask_img) > 0

        if combined_mask.shape != original_array.shape:
            raise ValueError(
                f"Mask shape {combined_mask.shape} != CT shape {original_array.shape}"
            )

        if not np.any(combined_mask):
            print("  ERROR: No lungs detected. Saving empty masks and aborting.")
            empty = sitk.GetImageFromArray(np.zeros(original_array.shape, dtype=np.uint8))
            empty.CopyInformation(original_ct)
            for suffix in ["_lung_mask", "_nodule_mask"]:
                sitk.WriteImage(empty, f"{output_prefix}{suffix}.nii.gz")
            sys.exit(1)

        lung_voxels = int(np.sum(combined_mask))
        print(f"  Lung voxels detected: {lung_voxels}")

        z_idx, y_idx, x_idx = np.where(combined_mask)
        margin = 10
        z_min = max(0, int(np.min(z_idx)) - margin)
        z_max = min(original_array.shape[0], int(np.max(z_idx)) + margin + 1)
        y_min = max(0, int(np.min(y_idx)) - margin)
        y_max = min(original_array.shape[1], int(np.max(y_idx)) + margin + 1)
        x_min = max(0, int(np.min(x_idx)) - margin)
        x_max = min(original_array.shape[2], int(np.max(x_idx)) + margin + 1)
        print(f"  Bounding box: z=[{z_min},{z_max}) y=[{y_min},{y_max}) x=[{x_min},{x_max})")

        lung_mask_output = f"{output_prefix}_lung_mask.nii.gz"
        lung_mask_out = sitk.GetImageFromArray(combined_mask.astype(np.uint8))
        lung_mask_out.CopyInformation(original_ct)
        sitk.WriteImage(lung_mask_out, lung_mask_output)
        print(f"  Lung mask saved -> {lung_mask_output}")

        masked_array = original_array.copy()
        masked_array[~combined_mask] = -1024

        cropped_array = masked_array[z_min:z_max, y_min:y_max, x_min:x_max]
        cropped_lung_mask = combined_mask[z_min:z_max, y_min:y_max, x_min:x_max]
        print(f"  Cropped volume shape: {cropped_array.shape}")

        del original_ct, original_array, masked_array, combined_mask, lung_mask_img, lung_mask_out

        # ── PHASE 4: Create overlapping patches ───────────────────────
        print()
        print("=" * 60)
        print("PHASE 4: Creating overlapping patches")
        print("=" * 60)

        coords = generate_patch_coords(cropped_array.shape, PATCH_SIZE, OVERLAP)
        print(f"  Volume shape: {cropped_array.shape}")
        print(f"  Patch size (Z,Y,X): {PATCH_SIZE}")
        print(f"  Overlap (Z,Y,X): {OVERLAP}")
        print(f"  Total patches: {len(coords)}")

        nnunet_in_dir = os.path.join(temp_dir, "nnunet_in")
        nnunet_out_dir = os.path.join(temp_dir, "nnunet_out")
        processed_patches_dir = os.path.join(results_dir, f"{args.case_id}_processed_patches")

        os.makedirs(nnunet_in_dir, exist_ok=True)
        os.makedirs(nnunet_out_dir, exist_ok=True)
        os.makedirs(processed_patches_dir, exist_ok=True)

        pz, py, px = PATCH_SIZE
        for i, (z0, z1, y0, y1, x0, x1) in enumerate(coords):
            patch_array = cropped_array[z0:z1, y0:y1, x0:x1].copy()
            valid_shape = (z1 - z0, y1 - y0, x1 - x0)

            if valid_shape != (pz, py, px):
                padded = np.full((pz, py, px), -1024, dtype=np.int16)
                padded[:valid_shape[0], :valid_shape[1], :valid_shape[2]] = patch_array
                patch_array = padded

            patch_img = sitk.GetImageFromArray(patch_array)
            patch_img.SetSpacing(original_spacing)
            patch_img.SetDirection(original_direction)
            patch_img.SetOrigin((0.0, 0.0, 0.0))

            patch_filename = f"patch_{i:04d}_0000.nii.gz"
            sitk.WriteImage(patch_img, os.path.join(nnunet_in_dir, patch_filename))
            sitk.WriteImage(patch_img, os.path.join(processed_patches_dir, patch_filename))

        print(f"  Saved {len(coords)} patches to nnU-Net input directory and {processed_patches_dir}")

        # ── PHASE 5: nnU-Net single-fold inference ─────────────────────────
        print()
        print("=" * 60)
        print(f"PHASE 5: nnU-Net single-fold inference (Fold {fold})")
        print("=" * 60)

        run_nnunet(nnunet_in_dir, nnunet_out_dir, fold)
        print("  nnU-Net complete.")

        # ── PHASE 6: Stitch + lung mask constraint ────────────────────
        print()
        print("=" * 60)
        print("PHASE 6: Stitching patch predictions")
        print("=" * 60)

        stitched_pred = np.zeros(cropped_array.shape, dtype=np.uint8)
        missing = 0
        for i, (z0, z1, y0, y1, x0, x1) in enumerate(coords):
            pred_path = os.path.join(nnunet_out_dir, f"patch_{i:04d}.npz")
            if not os.path.exists(pred_path):
                missing += 1
                continue

            npz_data = np.load(pred_path)
            probs = npz_data['softmax']
            nodule_probs = probs[1]

            valid_z = z1 - z0
            valid_y = y1 - y0
            valid_x = x1 - x0
            pred_valid = nodule_probs[:valid_z, :valid_y, :valid_x] > args.threshold
            stitched_pred[z0:z1, y0:y1, x0:x1] |= pred_valid.astype(np.uint8)

        if missing > 0:
            print(f"  WARNING: {missing}/{len(coords)} patch predictions missing!")

        stitched_pred[~cropped_lung_mask] = 0
        print(f"  Nodule voxels (cropped lung): {int(np.sum(stitched_pred > 0))}")

        # ── PHASE 7: Map to native CT space + save ─────────────────────
        print()
        print("=" * 60)
        print("PHASE 7: Mapping to native CT space")
        print("=" * 60)

        lung_mask_loaded = sitk.ReadImage(lung_mask_output)
        combined_mask_loaded = sitk.GetArrayFromImage(lung_mask_loaded) > 0

        final_mask_array = np.zeros(combined_mask_loaded.shape, dtype=np.uint8)
        final_mask_array[z_min:z_max, y_min:y_max, x_min:x_max] = stitched_pred
        final_mask_array[~combined_mask_loaded] = 0

        nodule_mask_output = f"{output_prefix}_nodule_mask.nii.gz"
        nodule_mask_img = sitk.GetImageFromArray(final_mask_array)
        nodule_mask_img.CopyInformation(lung_mask_loaded)
        sitk.WriteImage(nodule_mask_img, nodule_mask_output)
        print(f"  Nodule mask saved -> {nodule_mask_output}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    total_nodule_voxels = int(np.sum(final_mask_array > 0))
    print()
    print("=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"  Lung mask    : {lung_mask_output}")
    print(f"  Nodule mask  : {nodule_mask_output}")
    print(f"  Fold         : {fold}")
    print(f"  Threshold    : {args.threshold}")
    print(f"  Lung voxels  : {lung_voxels}")
    print(f"  Nodule voxels: {total_nodule_voxels}")
    print()
    print("  Open .nii.gz files in 3D Slicer / ITK-SNAP for visualization.")
    print("=" * 60)


if __name__ == "__main__":
    main()