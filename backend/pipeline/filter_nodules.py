import numpy as np
import torch
import scipy.ndimage
from pipeline.classifier import NoduleClassifier


def extract_centered_crop(volume, center_z, center_y, center_x, depth=32, height=48, width=48):
    z_min, z_max = center_z - depth // 2, center_z + depth // 2
    y_min, y_max = center_y - height // 2, center_y + height // 2
    x_min, x_max = center_x - width // 2, center_x + width // 2

    v_depth, v_height, v_width = volume.shape

    pad_z_before = max(0, -z_min)
    pad_z_after = max(0, z_max - v_depth)
    pad_y_before = max(0, -y_min)
    pad_y_after = max(0, y_max - v_height)
    pad_x_before = max(0, -x_min)
    pad_x_after = max(0, x_max - v_width)

    crop_z_min, crop_z_max = max(0, z_min), min(v_depth, z_max)
    crop_y_min, crop_y_max = max(0, y_min), min(v_height, y_max)
    crop_x_min, crop_x_max = max(0, x_min), min(v_width, x_max)

    crop = volume[crop_z_min:crop_z_max, crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    if any([pad_z_before, pad_z_after, pad_y_before, pad_y_after, pad_x_before, pad_x_after]):
        crop = np.pad(
            crop,
            ((pad_z_before, pad_z_after), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
            mode="constant",
            constant_values=-1000,
        )
    return crop


def filter_false_positives(
    volume, mask, classifier_path, voxel_spacing_mm=(1.0, 1.0, 1.0), threshold=0.2
):
    """
    Filter false positive nodule candidates using CNN classifier + size filter.
    Returns the filtered binary mask (only true nodules remain).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoduleClassifier().to(device)

    state_dict = torch.load(classifier_path, map_location=device, weights_only=False)
    if "model_state" in state_dict:
        model.load_state_dict(state_dict["model_state"])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    labeled_mask, num_features = scipy.ndimage.label(mask)
    print(f"  Found {num_features} initial candidates.")

    voxel_vol_mm3 = voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]
    min_vol_mm3 = (4 / 3) * np.pi * (1.5) ** 3
    min_voxels = int(np.ceil(min_vol_mm3 / voxel_vol_mm3))
    print(f"  Min voxels for 3mm diameter: {min_voxels}")

    filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    true_count = 0
    fp_count = 0
    noise_count = 0

    for i in range(1, num_features + 1):
        comp_mask = labeled_mask == i
        num_voxels = np.sum(comp_mask)

        if num_voxels < min_voxels:
            noise_count += 1
            continue

        coords = np.where(comp_mask)
        center_z = int(np.mean(coords[0]))
        center_y = int(np.mean(coords[1]))
        center_x = int(np.mean(coords[2]))

        crop = extract_centered_crop(volume, center_z, center_y, center_x)
        tensor_crop = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            _, probs = model(tensor_crop)

        prob_nodule = probs[0, 1].item()
        is_nodule = prob_nodule >= threshold

        if is_nodule:
            true_count += 1
            filtered_mask[comp_mask] = 1
        else:
            fp_count += 1

        label = "TRUE" if is_nodule else "FP"
        print(f"  Candidate {i}: {num_voxels} vox, prob={prob_nodule:.4f} -> {label}")

    print(f"  Noise (<3mm): {noise_count}")
    print(f"  True nodules: {true_count}")
    print(f"  False positives: {fp_count}")

    return filtered_mask
