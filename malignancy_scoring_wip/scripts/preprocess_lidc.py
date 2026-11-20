"""
preprocess_lidc.py — CLI script to run LIDC-IDRI preprocessing.

Usage:
    python scripts/preprocess_lidc.py \\
        --lidc_root  /path/to/LIDC-IDRI \\
        --output_dir data/preprocessed  \\
        --patch_size 64

This script calls lidc_preprocessor.preprocess_lidc() which:
  1. Loads each CT scan via pylidc.
  2. Extracts a 64×64×64 patch around each annotated nodule (>3mm).
  3. Builds a consensus segmentation mask (majority vote of 4 radiologists).
  4. Computes 93 radiomic features using PyRadiomics.
  5. Saves each nodule as a .npz file + a metadata.json index.

Run this ONCE before training. It takes several hours for the full dataset.
Progress is shown with a tqdm bar.
"""

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.lidc_preprocessor import preprocess_lidc
from data.splits             import assign_all_binary_labels, split_patients
import json


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess LIDC-IDRI dataset for Bloc 3.")
    p.add_argument("--lidc_root",   type=str, required=True,
                   help="Path to downloaded LIDC-IDRI dataset root.")
    p.add_argument("--output_dir",  type=str, default="data/preprocessed",
                   help="Directory to save .npz patch files.")
    p.add_argument("--patch_size",  type=int, default=64,
                   help="Cubic patch side length in voxels (default 64).")
    p.add_argument("--min_nodule",  type=float, default=3.0,
                   help="Minimum nodule diameter in mm (default 3.0).")
    p.add_argument("--seed",        type=int, default=42,
                   help="Random seed for score-3 label assignment.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[Preprocessor] LIDC root  : {args.lidc_root}")
    print(f"[Preprocessor] Output dir : {args.output_dir}")
    print(f"[Preprocessor] Patch size : {args.patch_size}³")
    print(f"[Preprocessor] Min nodule : {args.min_nodule} mm")
    print()

    # Run preprocessing
    metadata = preprocess_lidc(
        lidc_root    = args.lidc_root,
        output_dir   = args.output_dir,
        patch_size   = args.patch_size,
        min_nodule_mm = args.min_nodule,
        verbose      = True,
    )

    # Assign binary labels and perform patient-level split
    metadata = assign_all_binary_labels(metadata, random_seed=args.seed)
    train_ids, val_ids, test_ids = split_patients(metadata, random_seed=args.seed)

    # Save split assignments alongside metadata
    splits = {
        "train_patient_ids": train_ids,
        "val_patient_ids":   val_ids,
        "test_patient_ids":  test_ids,
    }
    splits_path = os.path.join(args.output_dir, "splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[Preprocessor] Splits saved → {splits_path}")

    # Update metadata.json with binary labels
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Preprocessor] Metadata updated → {meta_path}")


if __name__ == "__main__":
    main()
