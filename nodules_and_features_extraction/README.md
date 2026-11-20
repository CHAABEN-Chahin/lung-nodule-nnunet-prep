# extract_nodule_features.py

Extracts every nodule from a 3D CT volume + binary mask as an individual
entity, computing a full feature set for each one.

---

## What it does

1. Loads your CT volume and nodule mask
2. Runs connected component analysis — each blob in the mask = one nodule
3. For each nodule, extracts:
   - **Geometric features** — volume, diameter, centroid, extents, sphericity
   - **HU statistics** — mean, std, min, max, median, p25, p75
   - **64×64×64 patch** — CT crop centered on the nodule (`.npy` + `.nii.gz`)
   - **64×64×64 mask** — binary mask crop same size (`.npy` + `.nii.gz`)
   - **PyRadiomics vector** — ~93–107 texture/shape features (if installed)
4. Saves a combined `summary.json` and `summary.csv` with all nodules

---

## Usage

```bash
python extract_nodule_features.py \
    --volume  /path/to/ct_volume.nii.gz \
    --mask    /path/to/nodule_mask.nii.gz \
    --output  /path/to/output_folder
```

### Optional argument

| Argument | Default | Description |
|---|---|---|
| `--min_voxels` | 10 | Components smaller than this are treated as noise and skipped |

---

## Output structure

```
output_folder/
  nodule_001/
    patch_3d.npy          # 64×64×64 CT crop, float32 HU values
    mask_3d.npy           # 64×64×64 binary mask, uint8
    patch_3d.nii.gz       # same patch as NIfTI (open in 3D Slicer / ITK-SNAP)
    mask_3d.nii.gz        # same mask as NIfTI
    features.json         # geometry + HU stats (scalar values)
    radiomics.json        # PyRadiomics feature vector
  nodule_002/
    ...
  summary.json            # all nodules, all features in one file
  summary.csv             # same, CSV format — ready for pandas / Excel
```

---

## Install dependencies

```bash
pip install SimpleITK numpy scipy pandas pyradiomics
```

> **pyradiomics is optional.** If not installed, `radiomics.json` will be
> empty and a warning is shown. Everything else still works.

---

## Feature reference

### Geometry features (in `features.json`)

| Feature | Unit | Description |
|---|---|---|
| `centroid_z/y/x_vox` | voxels | Center of mass in voxel coordinates |
| `voxel_count` | voxels | Total number of foreground voxels |
| `volume_mm3` | mm³ | Physical volume = voxel_count × spacing |
| `diameter_mm` | mm | Max extent across all 3 axes |
| `z/y/x_extent_mm` | mm | Bounding box size per axis |
| `equiv_radius_mm` | mm | Radius of sphere with same volume |
| `sphericity` | 0–1 | How spherical the nodule is |

### HU statistics (in `features.json`)

| Feature | Description |
|---|---|
| `hu_mean` | Mean Hounsfield Unit inside nodule |
| `hu_std` | Standard deviation |
| `hu_min` / `hu_max` | Range |
| `hu_median` | Median HU |
| `hu_p25` / `hu_p75` | 25th / 75th percentile |

### PyRadiomics features (in `radiomics.json`)

~93–107 features including:
- First-order statistics (energy, entropy, kurtosis, skewness…)
- GLCM texture (contrast, correlation, homogeneity…)
- GLRLM (run-length features)
- Shape features (surface area, compactness…)

Full feature list: https://pyradiomics.readthedocs.io/en/latest/features.html

---

## Notes

- The `.nii.gz` patch files copy the original image's spacing/origin/direction
  so they open correctly in 3D Slicer and ITK-SNAP.
- The `.npy` files are the raw numpy arrays — directly loadable for model training.
- Noise filtering: components with fewer than `--min_voxels` voxels are skipped
  and a message is printed. Increase this threshold if you have many small artifacts.
