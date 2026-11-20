"""
dataset.py — PyTorch Dataset for Bloc 3 malignancy scoring.

Reads the .npz files produced by lidc_preprocessor.py (or eventually by Bloc 2).
The data format is identical in both cases — no changes needed at handshake time.

Each sample returns:
    patch    : torch.FloatTensor  shape (1, D, H, W) — 1-channel CT patch
    radiomics: torch.FloatTensor  shape (93,)
    label    : torch.LongTensor   scalar — 0 (benign) or 1 (malignant)
    score    : torch.FloatTensor  scalar — mean malignancy score 1.0–5.0
    nodule_id: str                for traceability
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Callable

from .augmentation import Safe3DAugmentation
from .splits import assign_all_binary_labels, split_patients


class NoduleDataset(Dataset):
    """
    PyTorch Dataset wrapping preprocessed LIDC-IDRI nodule patches.

    Augmentation is applied ONLY to malignant-class samples during training
    to address the class imbalance in LIDC-IDRI.
    """

    def __init__(
        self,
        records:    List[Dict],
        augment:    bool = False,
        augmentor:  Optional[Safe3DAugmentation] = None,
    ):
        """
        Args:
            records:   List of metadata dicts (from lidc_preprocessor.load_metadata).
                       Each dict must contain 'npz_path', 'malignancy_score',
                       'binary_label', and 'nodule_id'.
            augment:   If True, apply safe 3D augmentation to malignant samples.
            augmentor: Custom Safe3DAugmentation instance. If None and augment=True,
                       a default instance is created.
        """
        # Filter out records with no binary_label (shouldn't happen after preprocessing)
        self.records = [r for r in records if r.get("binary_label") is not None]

        self.augment   = augment
        self.augmentor = augmentor if augmentor else Safe3DAugmentation()

        # Pre-compute class counts for logging and WeightedSampler
        self.n_benign    = sum(1 for r in self.records if r["binary_label"] == 0)
        self.n_malignant = sum(1 for r in self.records if r["binary_label"] == 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # ── Load .npz file ────────────────────────────────────────────────────
        data     = np.load(rec["npz_path"])
        patch    = data["patch_3d"].astype(np.float32)    # (D, H, W)
        radiomics = data["radiomics"].astype(np.float32)  # (93,)

        label = int(rec["binary_label"])
        score = float(rec["malignancy_score"])

        # ── Augmentation (malignant class only, training only) ────────────────
        if self.augment and label == 1:
            patch = self.augmentor(patch)

        # ── Convert to tensors ────────────────────────────────────────────────
        # Add channel dimension: (D, H, W) → (1, D, H, W)
        patch_tensor     = torch.from_numpy(patch).unsqueeze(0)
        radiomics_tensor = torch.from_numpy(radiomics)
        label_tensor     = torch.tensor(label, dtype=torch.long)
        score_tensor     = torch.tensor(score, dtype=torch.float32)

        return patch_tensor, radiomics_tensor, label_tensor, score_tensor, rec["nodule_id"]

    def class_counts(self) -> Dict[str, int]:
        """Return counts for logging and class weight calculation."""
        return {"benign": self.n_benign, "malignant": self.n_malignant}


# ─── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(
    metadata:      List[Dict],
    train_ids:     List[str],
    val_ids:       List[str],
    test_ids:      List[str],
    batch_size:    int  = 16,
    num_workers:   int  = 4,
    random_seed:   int  = 42,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from split patient IDs.

    Args:
        metadata:    Full list of nodule metadata dicts (from load_metadata).
        train_ids:   Patient IDs assigned to training set.
        val_ids:     Patient IDs assigned to validation set.
        test_ids:    Patient IDs assigned to test set.
        batch_size:  Samples per batch.
        num_workers: Parallel data loading workers.
        random_seed: For binary label assignment of score-3 nodules.

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to DataLoaders.
    """
    # Assign binary labels (handles score-3 50/50 split)
    metadata = assign_all_binary_labels(metadata, random_seed=random_seed)

    # Build per-split record lists (patient-level partitioning)
    train_set_ids = set(train_ids)
    val_set_ids   = set(val_ids)
    test_set_ids  = set(test_ids)

    train_records = [r for r in metadata if r["patient_id"] in train_set_ids]
    val_records   = [r for r in metadata if r["patient_id"] in val_set_ids]
    test_records  = [r for r in metadata if r["patient_id"] in test_set_ids]

    # Datasets
    train_ds = NoduleDataset(train_records, augment=True)
    val_ds   = NoduleDataset(val_records,   augment=False)
    test_ds  = NoduleDataset(test_records,  augment=False)

    print(f"[Dataset] Train: {len(train_ds)} nodules "
          f"({train_ds.n_benign} benign / {train_ds.n_malignant} malignant)")
    print(f"[Dataset] Val:   {len(val_ds)} nodules "
          f"({val_ds.n_benign} benign / {val_ds.n_malignant} malignant)")
    print(f"[Dataset] Test:  {len(test_ds)} nodules "
          f"({test_ds.n_benign} benign / {test_ds.n_malignant} malignant)")

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,      # avoid single-sample batches in BatchNorm
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return loaders
