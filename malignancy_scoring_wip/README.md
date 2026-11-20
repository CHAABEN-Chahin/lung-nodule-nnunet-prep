# Bloc 3 — Malignancy Scoring Module

**Projet Préparation Métier** | Pulmonary Analysis from CT Scanners via 3D Reconstruction & Deep Learning  
*Chehine Chaaben & Yesmine Haddar | Supervised by Mr. Hichem Besbes & Mr. Riadh Abdelfattah*

---

## Position in the Pipeline

```
Bloc 1: CT slices ──► preprocessing ──► 3D volume (NIfTI)
Bloc 2: 3D volume ──► nnU-Net v2 segmentation ──► nodule masks
                  ──► patch extractor ──► 3D patches
                  ──► feature extractor ──► 93d radiomics vector
Bloc 3: [3D patch + 93d radiomics] ──► malignancy scoring   ← YOU ARE HERE
Bloc 4: scores + features ──► LLM ──► diagnostic report
```

Bloc 3 is the **malignancy scoring module** of Phase 1. It takes a 3D nodule patch and a radiomic feature vector, and outputs a malignancy score, a binary decision (benign/malignant), a confidence value, and an uncertainty flag.

---

## Project Structure

```
bloc3/
├── config.py                  # All hyperparameters in one place
├── train.py                   # Main training entry point
├── requirements.txt
│
├── data/
│   ├── lidc_preprocessor.py  # Extract patches from LIDC-IDRI (temporary Bloc 2 substitute)
│   ├── dataset.py             # PyTorch Dataset and DataLoader factory
│   ├── augmentation.py        # Safe 3D augmentations
│   └── splits.py              # Patient-level stratified train/val/test split
│
├── models/
│   ├── backbone.py            # 3D ResNet-18 + Models Genesis / MedicalNet loaders
│   ├── fusion.py              # Concatenation fusion module (512d CNN + 93d radiomics)
│   ├── heads.py               # Regression head (score 1–5) + Binary head (benign/malignant)
│   └── bloc3.py               # Full model assembling all components
│
├── training/
│   ├── loss.py                # Combined MSE + WeightedCrossEntropy loss
│   ├── metrics.py             # Sensitivity, AUC-ROC, F1, MAE + threshold checks
│   └── trainer.py             # Two-stage training loop with early stopping
│
└── scripts/
    └── preprocess_lidc.py     # CLI to run preprocessing on LIDC-IDRI
```

---

## Setup

```bash
pip install -r requirements.txt
```

Configure `pylidc` to find your LIDC-IDRI dataset:
```bash
# Create ~/.pylidcrc (Linux/Mac) or %USERPROFILE%\.pylidcrc (Windows)
[dicom]
path = /path/to/LIDC-IDRI
warn = True
```

---

## Step 1 — Preprocess LIDC-IDRI (run once)

```bash
python scripts/preprocess_lidc.py \
    --lidc_root  /path/to/LIDC-IDRI \
    --output_dir data/preprocessed  \
    --patch_size 64
```

This script extracts a `64×64×64` voxel patch and a 93-dimensional radiomic feature vector for each nodule (>3mm), and saves them as `.npz` files. It also performs the patient-level train/val/test split and saves `splits.json`.

**Takes several hours on the full dataset (~1018 patients, 2635 nodules).**

---

## Step 2 — Download Pretrained Backbone Weights

Before training, download pretrained weights for the backbone you want to use:

| Backbone | Repository | File to download |
|---|---|---|
| Models Genesis | https://github.com/MrGiovanni/ModelsGenesis | `genesis_chest_ct.pt` |
| MedicalNet ResNet18 | https://github.com/Tencent/MedicalNet | `resnet_18.pth` |

Place the downloaded file in a `weights/` folder.

---

## Step 3 — Train

```bash
# Models Genesis backbone
python train.py \
    --backbone models_genesis \
    --weights  weights/genesis_chest_ct.pt \
    --name     bloc3_genesis

# MedicalNet ResNet18 backbone
python train.py \
    --backbone medicalnet_resnet18 \
    --weights  weights/resnet_18.pth \
    --name     bloc3_medicalnet
```

Training runs **Stage 1** (frozen backbone) then **Stage 2** (unfrozen backbone) automatically.

---

## Architecture

### Input
| Input | Shape | Source |
|---|---|---|
| 3D nodule patch | `(1, 64, 64, 64)` | Bloc 2 patch extractor |
| Radiomics vector | `(93,)` | Bloc 2 / PyRadiomics |

### Model
```
3D Patch (1,64,64,64) ──► Backbone (ResNet3D18) ──► 512d CNN vector
                                                           │
Radiomics (93,)        ──────────────────────────► Fusion MLP ──► 256d
                                                           │
                                        ┌──────────────────┤
                                 Regression Head      Binary Head
                                  score [1.0–5.0]   probs [0,1]²
                                                      + uncertainty flag
```

### Output
| Output | Type | Description |
|---|---|---|
| `malignancy_score` | float [1.0–5.0] | Continuous risk score |
| `confidence` | float [0–1] | P(malignant) |
| `decision` | str | `'benign'` / `'malignant'` / `'uncertain'` |
| `uncertainty_flag` | bool | `True` if confidence ∈ [0.30, 0.70] → radiologist review |

---

## Training Strategy

| | Stage 1 | Stage 2 |
|---|---|---|
| Backbone | **Frozen** | **Unfrozen** |
| Backbone LR | 0 | `1e-5` |
| Fusion + Heads LR | `1e-4` | `1e-4` |
| Purpose | Learn to use pretrained features | Fine-tune for malignancy task |

**Early stopping** monitors both validation loss (patience=10) AND sensitivity (min 0.85).  
Weights are restored from the **best-sensitivity epoch**, not best-loss.

---

## Evaluation Metrics (Priority Order)

| Priority | Metric | Minimum Threshold |
|---|---|---|
| 1 | **Sensitivity** | ≥ 0.85 |
| 2 | AUC-ROC | ≥ 0.80 |
| 3 | F1-Score | ≥ 0.75 |
| 4 | MAE | ≤ 0.60 |

A backbone that fails **any single threshold** is eliminated from comparison, regardless of performance on other metrics.

---

## Backbone Comparison Protocol

Run training twice — once per backbone — with **identical** fusion, heads, loss, data, splits, optimizer, and epoch count. Only the backbone changes. Select the backbone with the highest sensitivity among those passing all thresholds.

---

## Handshake with Bloc 2 (pending)

The current preprocessing (`lidc_preprocessor.py`) is a **temporary substitute** for Bloc 2. It produces `.npz` files in the same format that Bloc 2 will eventually produce (see `data/lidc_preprocessor.py`, Section 11.1 of design doc). When the handshake with Bloc 2 is ready, `NoduleDataset` requires **no changes** — only the data source switches.
