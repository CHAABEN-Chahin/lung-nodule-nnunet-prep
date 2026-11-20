# Model weights

The weights are not stored in git (the nnU-Net checkpoint alone is about 250 MB).
Place them here before running inference or the backend:

```
model/
  nnUNet/3d_fullres/Task001_NoduleDetector/nnUNetTrainerV2__nnUNetPlansv2.1/
    plans.pkl
    fold_1/
      model_final_checkpoint.model
      model_final_checkpoint.model.pkl
      postprocessing.json
```

`predict.py` and `backend/config.py` both look for this folder next to them.

The false-positive classifier weights go in `nodule_filter_classifier/nodule_classifier.state` (not included either).
