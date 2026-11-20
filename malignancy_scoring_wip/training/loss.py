"""
loss.py — Combined loss function for Bloc 3.

Total Loss = lambda_1 × MSE(regression) + lambda_2 × WeightedCrossEntropy(binary)

Why a combined loss?
  The regression head and binary head must both improve simultaneously.
  The regression score provides nuance (1–5 range) while the binary head
  provides the direct clinical decision.

Why lambda_2 > lambda_1 (default 2.0 vs 1.0)?
  Clinical priority: the binary decision (benign/malignant) matters more
  than the exact score precision. Sensitivity > specificity in oncology —
  missing a malignant nodule is more dangerous than a false positive.

Why NOT equal lambdas?
  MSE and WeightedCrossEntropy operate on different numerical scales.
  Equal lambdas would let whichever loss happens to be numerically larger
  dominate training, regardless of clinical priority.

Lambda tuning protocol (from design doc, Section 7):
  Step 1: Start with lambda_1=1.0, lambda_2=2.0 (literature-based).
  Step 2: Run 5–10 epochs, observe loss magnitudes.
          If MSE >> CrossEntropy → increase lambda_2 or decrease lambda_1.
          If CrossEntropy >> MSE → vice versa.
  Step 3: Sequential short-epoch grid search (6 experiments × 20 epochs).
"""

import torch
import torch.nn as nn
from typing import Tuple


class Bloc3Loss(nn.Module):
    """
    Combined regression + binary classification loss.

    Args:
        lambda_1:         Weight for MSE regression loss.
        lambda_2:         Weight for WeightedCrossEntropy binary loss.
        class_weights:    Tensor([w_benign, w_malignant]) for cross-entropy.
                          Higher weight on malignant = higher sensitivity.
        device:           Torch device for class_weights tensor.
    """

    def __init__(
        self,
        lambda_1:      float = 1.0,
        lambda_2:      float = 2.0,
        class_weights: Tuple[float, float] = (1.0, 3.0),
        device:        str   = "cuda",
    ):
        super().__init__()

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        # MSE for regression — measures distance on the 1.0–5.0 score scale
        self.mse = nn.MSELoss()

        # Weighted cross-entropy for binary classification
        # Higher weight on class 1 (malignant) penalises false negatives more
        weights = torch.tensor(list(class_weights), dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weights.to(device))

    def forward(
        self,
        score:      torch.Tensor,   # (B, 1) — regression output [1.0, 5.0]
        probs:      torch.Tensor,   # (B, 2) — binary head logits / probs
        true_score: torch.Tensor,   # (B,)   — ground truth score [1.0, 5.0]
        true_label: torch.Tensor,   # (B,)   — ground truth binary label {0, 1}
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            score:      (B, 1) regression prediction
            probs:      (B, 2) binary class probabilities
            true_score: (B,)  ground truth malignancy score
            true_label: (B,)  ground truth binary label (long)

        Returns:
            total_loss:  scalar — lambda_1 * mse + lambda_2 * ce
            loss_reg:    scalar — unweighted MSE (for logging)
            loss_cls:    scalar — unweighted CE  (for logging)
        """
        # Regression loss — compare predicted score to radiologist consensus
        loss_reg = self.mse(score.squeeze(1), true_score.float())

        # Binary classification loss — probs are already softmax probabilities,
        # but CrossEntropyLoss expects raw logits → pass the pre-softmax scores.
        # NOTE: BinaryHead outputs softmax probs; CE expects logits.
        # We work around this by using NLLLoss on log(probs) instead:
        loss_cls = nn.functional.nll_loss(
            torch.log(probs + 1e-8),   # log-probs, (B, 2)
            true_label.long(),
            weight=self.ce.weight,
        )

        total_loss = self.lambda_1 * loss_reg + self.lambda_2 * loss_cls

        return total_loss, loss_reg.detach(), loss_cls.detach()

    def update_lambdas(self, lambda_1: float, lambda_2: float):
        """Update loss weights during lambda tuning experiments."""
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        print(f"[Loss] Updated: lambda_1={lambda_1}, lambda_2={lambda_2}")
