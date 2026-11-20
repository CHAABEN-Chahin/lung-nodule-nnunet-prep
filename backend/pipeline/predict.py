import os
import sys
import subprocess
import numpy as np
import SimpleITK as sitk


PATCH_SIZE = (80, 192, 192)
OVERLAP = (20, 48, 48)
DEFAULT_THRESHOLD = 0.15


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


def run_nnunet(input_dir, output_dir, model_dir, fold=1):
    """Single-threaded nnU-Net v1 inference — no multiprocessing (Windows-safe)."""
    import torch
    import gc

    _original_torch_load = torch.load
    def _safe_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)
    torch.load = _safe_load

    os.environ["RESULTS_FOLDER"] = os.path.abspath(model_dir)

    from nnunet.training.model_restore import load_model_and_checkpoint_files
    from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
    from batchgenerators.utilities.file_and_folder_operations import subfiles, join

    model_folder = os.path.join(
        os.path.abspath(model_dir),
        "nnUNet", "3d_fullres", "Task001_NoduleDetector",
        "nnUNetTrainerV2__nnUNetPlansv2.1",
    )

    print(f"  Running nnU-Net fold {fold} (single-threaded)...")

    trainer, params = load_model_and_checkpoint_files(
        model_folder, [fold], mixed_precision=True,
        checkpoint_name="model_final_checkpoint",
    )
    trainer.load_checkpoint_ram(params[0], False)

    if "segmentation_export_params" in trainer.plans:
        force_separate_z = trainer.plans["segmentation_export_params"]["force_separate_z"]
        interpolation_order = trainer.plans["segmentation_export_params"]["interpolation_order"]
        interpolation_order_z = trainer.plans["segmentation_export_params"]["interpolation_order_z"]
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0

    input_files = sorted(subfiles(os.path.abspath(input_dir), suffix=".nii.gz"))
    case_ids = [os.path.basename(f)[:-12] for f in input_files]

    print(f"  {len(input_files)} patches to process")

    for i, (input_file, case_id) in enumerate(zip(input_files, case_ids)):
        output_file = join(os.path.abspath(output_dir), case_id + ".nii.gz")
        npz_file = join(os.path.abspath(output_dir), case_id + ".npz")

        print(f"  [{i+1}/{len(input_files)}] {case_id}")

        d, _, dct = trainer.preprocess_patient([input_file])
        d = d.astype(np.float32)

        softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
            d, do_mirroring=False,
            mirror_axes=trainer.data_aug_params["mirror_axes"],
            use_sliding_window=True, step_size=0.5,
            use_gaussian=True, all_in_gpu=False,
            mixed_precision=True,
        )[1]

        transpose_forward = trainer.plans.get("transpose_forward")
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get("transpose_backward")
            softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

        if hasattr(trainer, "regions_class_order"):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None

        np.savez_compressed(npz_file, softmax=softmax)

        save_segmentation_nifti_from_softmax(
            softmax, output_file, dct, interpolation_order,
            region_class_order, None, None,
            npz_file, None, force_separate_z, interpolation_order_z,
        )

    torch.load = _original_torch_load
    gc.collect()
    print("  nnU-Net complete.")


def create_patches_and_infer(
    ct_array, lung_mask_array, original_spacing, original_direction,
    model_dir, temp_dir, threshold=DEFAULT_THRESHOLD
):
    """
    Mask non-lung tissue, crop, create patches, run nnU-Net, stitch.
    Returns the stitched nodule mask in native CT space.
    """
    masked_array = ct_array.copy()
    masked_array[~lung_mask_array] = -1024

    z_idx, y_idx, x_idx = np.where(lung_mask_array)
    margin = 10
    z_min = max(0, int(np.min(z_idx)) - margin)
    z_max = min(ct_array.shape[0], int(np.max(z_idx)) + margin + 1)
    y_min = max(0, int(np.min(y_idx)) - margin)
    y_max = min(ct_array.shape[1], int(np.max(y_idx)) + margin + 1)
    x_min = max(0, int(np.min(x_idx)) - margin)
    x_max = min(ct_array.shape[2], int(np.max(x_idx)) + margin + 1)
    print(f"  Bounding box: z=[{z_min},{z_max}) y=[{y_min},{y_max}) x=[{x_min},{x_max})")

    cropped_array = masked_array[z_min:z_max, y_min:y_max, x_min:x_max]
    cropped_lung_mask = lung_mask_array[z_min:z_max, y_min:y_max, x_min:x_max]
    print(f"  Cropped volume shape: {cropped_array.shape}")

    coords = generate_patch_coords(cropped_array.shape, PATCH_SIZE, OVERLAP)
    print(f"  Total patches: {len(coords)}")

    nnunet_in_dir = os.path.join(temp_dir, "nnunet_in")
    nnunet_out_dir = os.path.join(temp_dir, "nnunet_out")
    os.makedirs(nnunet_in_dir, exist_ok=True)
    os.makedirs(nnunet_out_dir, exist_ok=True)

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
        sitk.WriteImage(patch_img, os.path.join(nnunet_in_dir, f"patch_{i:04d}_0000.nii.gz"))

    print(f"  Saved {len(coords)} patches")

    run_nnunet(nnunet_in_dir, nnunet_out_dir, model_dir, fold=1)

    stitched_pred = np.zeros(cropped_array.shape, dtype=np.uint8)
    missing = 0
    for i, (z0, z1, y0, y1, x0, x1) in enumerate(coords):
        pred_path = os.path.join(nnunet_out_dir, f"patch_{i:04d}.npz")
        if not os.path.exists(pred_path):
            missing += 1
            continue
        npz_data = np.load(pred_path)
        probs = npz_data["softmax"]
        nodule_probs = probs[1]
        valid_z, valid_y, valid_x = z1 - z0, y1 - y0, x1 - x0
        pred_valid = nodule_probs[:valid_z, :valid_y, :valid_x] > threshold
        stitched_pred[z0:z1, y0:y1, x0:x1] |= pred_valid.astype(np.uint8)

    if missing > 0:
        print(f"  WARNING: {missing}/{len(coords)} patch predictions missing!")

    stitched_pred[~cropped_lung_mask] = 0
    print(f"  Nodule voxels (cropped): {int(np.sum(stitched_pred > 0))}")

    final_mask = np.zeros(ct_array.shape, dtype=np.uint8)
    final_mask[z_min:z_max, y_min:y_max, x_min:x_max] = stitched_pred
    final_mask[~lung_mask_array] = 0

    print(f"  Nodule voxels (native): {int(np.sum(final_mask > 0))}")
    return final_mask
