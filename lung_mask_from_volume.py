"""
Lung Mask Extraction from 3D NIfTI Volume
==========================================
Input:  A 3D CT volume (.nii.gz) in native size (already constructed)
Output: Binary lung mask (.nii.gz) in the same native size

Usage:
  python lung_mask_from_volume.py --input volume.nii.gz
  python lung_mask_from_volume.py --input volume.nii.gz --output custom_lung_mask.nii.gz
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
import SimpleITK as sitk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "result_lungmask")


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


def main():
    parser = argparse.ArgumentParser(description="Extract lung mask from a 3D CT volume")
    parser.add_argument("--input", required=True, help="Path to 3D CT volume (.nii.gz)")
    parser.add_argument("--output", default=None, help="Path for output lung mask (.nii.gz). Defaults to result_lungmask/<input_stem>_lung_mask.nii.gz")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        stem = os.path.splitext(os.path.splitext(os.path.basename(input_path))[0])[0]
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{stem}_lung_mask.nii.gz")

    if not os.path.isfile(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    temp_dir = tempfile.mkdtemp()
    try:
        lung_mask_path = os.path.join(temp_dir, "lung_mask.nii.gz")

        print("=" * 60)
        print("STEP 1: Running TotalSegmentator on input volume")
        print("=" * 60)
        run_totalsegmentator(input_path, lung_mask_path)

        print()
        print("=" * 60)
        print("STEP 2: Combining lobes into binary lung mask")
        print("=" * 60)

        original_ct = sitk.ReadImage(input_path)
        original_array = sitk.GetArrayFromImage(original_ct)

        lung_mask_img = sitk.ReadImage(lung_mask_path)
        combined_mask = sitk.GetArrayFromImage(lung_mask_img) > 0

        if combined_mask.shape != original_array.shape:
            raise ValueError(
                f"Mask shape {combined_mask.shape} != CT shape {original_array.shape}"
            )

        if not np.any(combined_mask):
            print("  WARNING: No lungs detected. Saving empty mask.")
            combined_mask = np.zeros(original_array.shape, dtype=np.uint8)
        else:
            print(f"  Lung voxels detected: {int(np.sum(combined_mask))}")

        lung_mask_out = sitk.GetImageFromArray(combined_mask.astype(np.uint8))
        lung_mask_out.CopyInformation(original_ct)
        sitk.WriteImage(lung_mask_out, output_path)
        print(f"  Lung mask saved -> {output_path}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Lung voxels: {int(np.sum(combined_mask))}")


if __name__ == "__main__":
    main()