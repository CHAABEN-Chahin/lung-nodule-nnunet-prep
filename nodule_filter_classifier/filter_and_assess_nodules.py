import numpy as np
import torch
import argparse
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
from classifier import NoduleClassifier

def extract_centered_crop(volume, center_z, center_y, center_x, depth=32, height=48, width=48):
    """
    Extracts a crop of the specified size centered around the given coordinates.
    Pads the array with the minimum density value if the crop exceeds the volume boundaries.
    """
    z_min, z_max = center_z - depth // 2, center_z + depth // 2
    y_min, y_max = center_y - height // 2, center_y + height // 2
    x_min, x_max = center_x - width // 2, center_x + width // 2

    v_depth, v_height, v_width = volume.shape

    # Calculate required padding
    pad_z_before = max(0, -z_min)
    pad_z_after = max(0, z_max - v_depth)
    
    pad_y_before = max(0, -y_min)
    pad_y_after = max(0, y_max - v_height)

    pad_x_before = max(0, -x_min)
    pad_x_after = max(0, x_max - v_width)
    
    # Calculate crop indices within the volume
    crop_z_min, crop_z_max = max(0, z_min), min(v_depth, z_max)
    crop_y_min, crop_y_max = max(0, y_min), min(v_height, y_max)
    crop_x_min, crop_x_max = max(0, x_min), min(v_width, x_max)

    # Extract crop
    crop = volume[crop_z_min:crop_z_max, crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    # Pad if necessary
    if any([pad_z_before, pad_z_after, pad_y_before, pad_y_after, pad_x_before, pad_x_after]):
        crop = np.pad(
            crop,
            ((pad_z_before, pad_z_after), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
            mode='constant',
            constant_values=-1000 # Typical minimum value for HU (air)
        )
    return crop


def process_and_visualize(volume, mask, model_path, voxel_spacing_mm=(1.0, 1.0, 1.0), output_mask_path='filtered_nodules.nii.gz', affine=np.eye(4), threshold=0.2):
    """
    volume: 3D numpy array of the CT scan (Hounsfield Units)
    mask: 3D numpy array of the segmentation (1 for nodule candidate, 0 for background)
    model_path: Path to the NoduleClassifier trained weights
    voxel_spacing_mm: (z_spacing, y_spacing, x_spacing) in mm. 
                      Used to filter out noise spots < 3mm.
    output_mask_path: Path to save the final extracted .nii.gz mask without noise & false positives.
    affine: 4x4 affine matrix linking the volume coordinates to world coordinates.
    threshold: The probability threshold (0.0 to 1.0) above which a candidate is considered a true nodule.
    """
    # 1. Load the False-Positive Reduction Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NoduleClassifier().to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state' in state_dict:
        model.load_state_dict(state_dict['model_state'])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    # 2. Extract Connected Components from the mask
    labeled_mask, num_features = scipy.ndimage.label(mask)
    print(f"Found {num_features} initial candidates in the segmentation mask.")

    # 3. Process each candidate
    true_nodules_points = []
    false_positives_points = []
    skipped_noise = 0
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)

    # Calculate spherical volume of a 3mm diameter spot
    voxel_vol_mm3 = voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]
    min_vol_mm3 = (4/3) * np.pi * (1.5)**3 # Volume for a radius of 1.5mm
    min_voxels = int(np.ceil(min_vol_mm3 / voxel_vol_mm3))
    print(f"Filtering out candidates smaller than ~{min_voxels} voxels (< 3mm diameter).")

    for i in range(1, num_features + 1):
        comp_mask = (labeled_mask == i)
        num_voxels = np.sum(comp_mask)
        
        # Determine filtering size (Noise)
        if num_voxels < min_voxels:
            skipped_noise += 1
            continue
            
        # Get centroid of candidate
        coords = np.where(comp_mask)
        center_z = int(np.mean(coords[0]))
        center_y = int(np.mean(coords[1]))
        center_x = int(np.mean(coords[2]))
        
        # 4. Extract Crop and Classify
        crop = extract_centered_crop(volume, center_z, center_y, center_x)
        
        # Add batch and channel dimension for PyTorch (1, 1, D, H, W)
        tensor_crop = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, probs = model(tensor_crop)
            
        prob_nodule = probs[0, 1].item()
        is_nodule = prob_nodule >= threshold
        
        print(f"Candidate {i} (Vol: {num_voxels} voxels): Nodule Probability = {prob_nodule:.4f} -> {'True Nodule' if is_nodule else 'False Positive'}")
        
        # Save points for 3D visualization
        points = np.column_stack(coords)
        if is_nodule:
            true_nodules_points.append(points)
            filtered_mask[comp_mask] = 1 # Keep it in the clean mask
        else:
            false_positives_points.append(points)
            
    print(f"\nSummary:")
    print(f" - Ignored noise spots (<3mm): {skipped_noise}")
    print(f" - True Nodules kept: {len(true_nodules_points)}")
    print(f" - False Positives rejected: {len(false_positives_points)}")

    if output_mask_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_mask_path)), exist_ok=True)
        filtered_nii = nib.Nifti1Image(filtered_mask, affine)
        nib.save(filtered_nii, output_mask_path)
        print(f"\n[+] Saved the cleaned true nodule mask to: {output_mask_path}")

    # 5. Visualize Results in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot True Nodules (Green)
    for pts in true_nodules_points:
        ax.scatter(pts[:, 2], pts[:, 1], pts[:, 0], c='green', alpha=0.1, s=1, label='True Nodule')
        
    # Plot False Positives (Red)
    for pts in false_positives_points:
        ax.scatter(pts[:, 2], pts[:, 1], pts[:, 0], c='red', alpha=0.1, s=1, label='False Positive')

    # Fix duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    if unique:
        ax.legend(*zip(*unique))

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Nodule Detection Results')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter false positive nodules from segmentation mask using 3D CNN.")
    parser.add_argument('--volume', '-v', type=str, required=True, help="Path to the original CT volume (.nii or .nii.gz)")
    parser.add_argument('--mask', '-m', type=str, required=True, help="Path to the predicted nodule mask (.nii or .nii.gz)")
    parser.add_argument('--output', '-o', type=str, default=os.path.join('results', 'filtered_nodules.nii.gz'), help="Path to save the cleaned mask")
    parser.add_argument('--model', '-w', type=str, default='nodule_classifier.state', help="Path to the model weights")
    parser.add_argument('--spacing', '-s', type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Voxel spacing in mm (z y x)")
    parser.add_argument('--threshold', '-t', type=float, default=0.2, help="Probability threshold to keep a nodule (default: 0.2)")
    
    args = parser.parse_args()

    print(f"Loading Volume: {args.volume}")
    vol_nii = nib.load(args.volume)
    volume_data = vol_nii.get_fdata()

    print(f"Loading Mask: {args.mask}")
    mask_nii = nib.load(args.mask)
    mask_data = mask_nii.get_fdata()

    # The NoduleClassifier expects [depth, height, width] Z, Y, X
    # Ensure dimensions match (sometimes nibabel loads X, Y, Z depending on the standard)
    # Assuming the input is correctly oriented, otherwise add np.transpose(..., (2, 1, 0)) here if needed
    
    process_and_visualize(
        volume=volume_data, 
        mask=mask_data, 
        model_path=os.path.join(os.path.dirname(__file__), args.model) if not os.path.isabs(args.model) else args.model,
        voxel_spacing_mm=tuple(args.spacing),
        output_mask_path=args.output,
        affine=mask_nii.affine,
        threshold=args.threshold
    )
