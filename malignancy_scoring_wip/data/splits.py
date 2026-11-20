"""
splits.py — Patient-level stratified split for LIDC-IDRI.

CRITICAL RULE: All splits are at the PATIENT level, not the nodule level.
LIDC-IDRI patients can have multiple nodules. A nodule-level split would
place nodules from the same patient in both train and test sets, causing
data leakage and artificially inflated test performance.

Score-3 handling: Score 3 sits at the benign/malignant boundary, consistent
with the uncertainty flag design. These nodules are split 50/50 between the
benign and malignant groups rather than treated as a separate class.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split


def assign_binary_label(score: float) -> int:
    """
    Convert average malignancy score (1.0–5.0) to binary label.

    Benign  (0): score <= 2.5  (radiologist scores 1 or 2)
    Malignant (1): score > 2.5  (radiologist scores 3, 4, or 5)

    Score 3 nodules are assigned label 1 here for stratification purposes,
    but are physically split 50/50 by split_score3_nodules() afterward.
    """
    return 0 if score <= 2.5 else 1


def split_score3_nodules(nodule_records: List[Dict], random_seed: int = 42) -> List[Dict]:
    """
    For nodules whose average malignancy score falls in (2.5, 3.5],
    assign binary labels 50% benign / 50% malignant.

    This is consistent with the uncertainty flag design — score 3 represents
    the ambiguous boundary between classes.

    Args:
        nodule_records: List of dicts, each with at least 'malignancy_score'.
                        Will have 'binary_label' set in place.
        random_seed:    Reproducibility seed.

    Returns:
        nodule_records with 'binary_label' populated for score-3 nodules.
    """
    score3 = [n for n in nodule_records if 2.5 < n["malignancy_score"] <= 3.5]

    rng = np.random.default_rng(random_seed)
    rng.shuffle(score3)  # in-place shuffle for reproducibility

    half = len(score3) // 2
    for i, nodule in enumerate(score3):
        nodule["binary_label"] = 0 if i < half else 1

    return nodule_records


def assign_all_binary_labels(nodule_records: List[Dict], random_seed: int = 42) -> List[Dict]:
    """
    Populate 'binary_label' for every nodule record.

    Score 1–2   → 0 (benign)
    Score 3     → 50/50 split (handled by split_score3_nodules)
    Score 4–5   → 1 (malignant)

    Args:
        nodule_records: List of dicts with 'malignancy_score' key.
        random_seed:    Seed for the score-3 50/50 split.

    Returns:
        nodule_records with 'binary_label' set for all entries.
    """
    # Assign clear-cut labels first
    for n in nodule_records:
        score = n["malignancy_score"]
        if score <= 2.5:
            n["binary_label"] = 0
        elif score > 3.5:
            n["binary_label"] = 1
        # Score 3 nodules get 'binary_label' = None temporarily

    # Handle ambiguous score-3 nodules
    nodule_records = split_score3_nodules(nodule_records, random_seed)

    return nodule_records


def split_patients(
    nodule_records: List[Dict],
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
    test_ratio:   float = 0.15,
    random_seed:  int   = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split LIDC-IDRI patients into train / val / test sets with stratification.

    Stratification ensures that class distribution (benign vs malignant) is
    preserved in all three splits. Each patient is assigned its dominant class
    label (highest malignancy score across all its nodules).

    Args:
        nodule_records: List of dicts, each containing:
                          - 'patient_id'       (str)
                          - 'malignancy_score' (float, 1.0–5.0)
        train_ratio:    Fraction of patients for training.
        val_ratio:      Fraction of patients for validation.
        test_ratio:     Fraction of patients for testing.
        random_seed:    Reproducibility seed.

    Returns:
        (train_ids, val_ids, test_ids) — three lists of patient ID strings.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + val + test ratios must sum to 1.0"

    # ── Group nodules by patient ──────────────────────────────────────────────
    patient_scores: Dict[str, List[float]] = {}
    for rec in nodule_records:
        pid = rec["patient_id"]
        patient_scores.setdefault(pid, []).append(rec["malignancy_score"])

    patient_ids = np.array(list(patient_scores.keys()))

    # Dominant class = label of the most malignant nodule in the patient
    patient_labels = np.array([
        assign_binary_label(max(scores))
        for scores in patient_scores.values()
    ])

    # ── First split: train vs (val + test) ───────────────────────────────────
    temp_size = val_ratio + test_ratio
    train_ids, temp_ids, _, temp_labels = train_test_split(
        patient_ids,
        patient_labels,
        test_size=temp_size,
        stratify=patient_labels,
        random_state=random_seed,
    )

    # ── Second split: val vs test ─────────────────────────────────────────────
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1.0 - relative_val),
        stratify=temp_labels,
        random_state=random_seed,
    )

    print(
        f"[Splits] Patients  — train: {len(train_ids)} | "
        f"val: {len(val_ids)} | test: {len(test_ids)}"
    )

    # Count nodules per split for logging
    split_map = {pid: "train" for pid in train_ids}
    split_map.update({pid: "val" for pid in val_ids})
    split_map.update({pid: "test" for pid in test_ids})

    counts = {"train": 0, "val": 0, "test": 0}
    for rec in nodule_records:
        counts[split_map[rec["patient_id"]]] += 1

    print(
        f"[Splits] Nodules   — train: {counts['train']} | "
        f"val: {counts['val']} | test: {counts['test']}"
    )

    return list(train_ids), list(val_ids), list(test_ids)
