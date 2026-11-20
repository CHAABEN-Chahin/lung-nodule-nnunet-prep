# Nodule False Positive Reduction Classifier

This folder contains the 3D CNN model used to classify segmentation candidates as true nodule vs. non-nodule.

## Contents

- `classifier.py` — PyTorch definitions for `NoduleClassifier` and `NoduleBlock`.
- `nodule_classifier.state` — Model weights (not included in this repository).
- `filter_and_assess_nodules.py` — End-to-end script: loads a CT + nodule mask, filters noise, runs the classifier, saves a cleaned mask and shows a 3D visualization.
- `requirements.txt` — Python dependencies.

## Model Input / Output

The `NoduleClassifier` takes a 5D tensor `(batch_size, 1, 32, 48, 48)` (HU values) and returns:

1. `linear_output` — raw logits, shape `(batch_size, 2)`
2. `softmax_head` — class probabilities, shape `(batch_size, 2)` where index 1 = nodule probability

**Shape constraint**: `(batch_size, 1, 32, 48, 48)`

## Code Example

```python
import torch
from classifier import NoduleClassifier

model = NoduleClassifier()
state_dict = torch.load('nodule_classifier.state')
if 'model_state' in state_dict:
    model.load_state_dict(state_dict['model_state'])
else:
    model.load_state_dict(state_dict)
model.eval()

# Dummy input (batch=1, channels=1, depth=32, height=48, width=48)
input_tensor = torch.randn(1, 1, 32, 48, 48)
with torch.no_grad():
    logits, probabilities = model(input_tensor)
print(f"Probabilities: {probabilities}")
# e.g., tensor([[0.012, 0.988]]) -> High confidence in nodule presence
```

## Run the filter script

```bash
python filter_and_assess_nodules.py \
    --volume path/to/ct_volume.nii.gz \
    --mask   path/to/nodule_mask.nii.gz \
    --output results/filtered_nodules.nii.gz \
    --model  nodule_classifier.state \
    --threshold 0.2
```
