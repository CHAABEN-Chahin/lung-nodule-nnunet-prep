import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def run_totalsegmentator(input_nifti, output_file):
    from totalsegmentator.python_api import totalsegmentator
    import gc

    print("  Running TotalSegmentator (fastest mode, CPU, low-memory)...")
    input_img = nib.load(input_nifti)

    output_img = totalsegmentator(
        input_img,
        ml=True,
        fastest=True,
        device="cpu",
        roi_subset=[
            "lung_upper_lobe_left", "lung_lower_lobe_left",
            "lung_upper_lobe_right", "lung_middle_lobe_right",
            "lung_lower_lobe_right",
        ],
        nr_thr_resamp=1,
        nr_thr_saving=1,
        nora_tag="None",
        quiet=True,
    )

    nib.save(output_img, output_file)
    del input_img, output_img
    gc.collect()
    print("  TotalSegmentator complete.")


def extract_lung_and_lobe_masks(ct_path, lung_mask_path, lobe_mask_path):
    """
    Run TotalSegmentator on CT volume to extract:
    1. Binary lung mask (all lobes merged)
    2. Multi-label lobe mask (each lobe has a distinct integer label)
    """
    run_totalsegmentator(ct_path, lobe_mask_path)

    original_ct = sitk.ReadImage(ct_path)
    original_array = sitk.GetArrayFromImage(original_ct)

    lobe_img = sitk.ReadImage(lobe_mask_path)
    lobe_array = sitk.GetArrayFromImage(lobe_img)

    binary_mask = (lobe_array > 0).astype(np.uint8)

    if binary_mask.shape != original_array.shape:
        raise ValueError(
            f"Mask shape {binary_mask.shape} != CT shape {original_array.shape}"
        )

    if not np.any(binary_mask):
        print("  WARNING: No lungs detected.")
        return 0

    lung_voxels = int(np.sum(binary_mask))
    print(f"  Lung voxels detected: {lung_voxels}")

    unique_labels = np.unique(lobe_array[lobe_array > 0])
    print(f"  Lobe labels found: {unique_labels.tolist()}")

    lung_mask_out = sitk.GetImageFromArray(binary_mask)
    lung_mask_out.CopyInformation(original_ct)
    sitk.WriteImage(lung_mask_out, lung_mask_path)
    print(f"  Lung mask saved -> {lung_mask_path}")

    lobe_out = sitk.GetImageFromArray(lobe_array)
    lobe_out.CopyInformation(original_ct)
    sitk.WriteImage(lobe_out, lobe_mask_path)
    print(f"  Lobe masks saved -> {lobe_mask_path}")

    return lung_voxels
