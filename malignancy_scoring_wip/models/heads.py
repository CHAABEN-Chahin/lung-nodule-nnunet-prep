"""
heads.py — Output heads for Bloc 3.

Two heads, both fed by the fusion module output:

  RegressionHead  →  malignancy score 1.0–5.0 (continuous)
  BinaryHead      →  benign / malignant + confidence (0–1) + uncertainty flag

Why a dual-head design?
  A single classification head would not capture the clinical nuance between
  a score of 2 labeled "benign" and a score of 1 labeled "benign".
  The regression head provides the granular score; the binary head adds the
  safety mechanism (uncertainty flag) on top.

Why the binary head takes the regression score as extra input?
  This ensures the binary decision is always consistent with the regression
  score and cannot contradict it. Without this, the two heads could
  independently predict conflicting results.

Uncertainty flag (core clinical safety mechanism):
  confidence < 0.30           → Benign,    flag OFF
  0.30 ≤ confidence ≤ 0.70   → Uncertain, flag ON  (radiologist review)
  confidence > 0.70           → Malignant, flag OFF
"""

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """
    Predicts a continuous malignancy score in [1.0, 5.0].

    Architecture:
        fusion_output (hidden_dim)
          → Linear(hidden_dim → 256) + ReLU + Dropout(0.3)
          → Linear(256 → 1)
          → sigmoid × 4 + 1      (maps sigmoid output [0,1] to [1.0, 5.0])

    Input:  (B, hidden_dim)
    Output: (B, 1)  — score in [1.0, 5.0]
    """

    def __init__(self, hidden_dim: int = 256, dropout_rate: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (B, hidden_dim)

        Returns:
            score: (B, 1) — malignancy score in [1.0, 5.0]
        """
        raw   = self.net(fused)                # (B, 1), unbounded
        score = torch.sigmoid(raw) * 4.0 + 1.0  # (B, 1), range [1.0, 5.0]
        return score


class BinaryHead(nn.Module):
    """
    Predicts benign / malignant probability + generates the uncertainty flag.

    Takes BOTH the fusion output AND the regression score as input.
    The extra regression score input (1d) ensures the binary decision cannot
    contradict the continuous score prediction.

    Architecture:
        [fusion_output (hidden_dim) || regression_score (1)] = (hidden_dim + 1)
          → Linear(hidden_dim+1 → 256) + ReLU + Dropout(0.3)
          → Linear(256 → 2) + Softmax
          → [p_benign, p_malignant]

    Input:  fused (B, hidden_dim), regression_score (B, 1)
    Output: probs (B, 2)  — [P(benign), P(malignant)]
    """

    def __init__(self, hidden_dim: int = 256, dropout_rate: float = 0.3):
        super().__init__()

        # +1 for the concatenated regression score
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(
        self,
        fused:             torch.Tensor,
        regression_score:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fused:            (B, hidden_dim)
            regression_score: (B, 1) — from RegressionHead

        Returns:
            probs: (B, 2) — [P(benign), P(malignant)] summing to 1.0
        """
        # Concatenate fusion features with regression score → (B, hidden_dim+1)
        combined = torch.cat([fused, regression_score], dim=1)
        logits   = self.net(combined)                          # (B, 2)
        probs    = torch.softmax(logits, dim=1)                # (B, 2)
        return probs


# ─── Inference utilities ──────────────────────────────────────────────────────

def make_decision(
    confidence:          torch.Tensor,
    benign_threshold:    float = 0.30,
    malignant_threshold: float = 0.70,
):
    """
    Convert malignant confidence score to a clinical decision + uncertainty flag.

    Args:
        confidence:          (B,) — P(malignant) from BinaryHead
        benign_threshold:    Below this → "benign", flag OFF
        malignant_threshold: Above this → "malignant", flag OFF
                             Between the two → "uncertain", flag ON

    Returns:
        decisions:        List[str] — 'benign', 'malignant', or 'uncertain'
        uncertainty_flags: List[bool]
    """
    decisions        = []
    uncertainty_flags = []

    for conf in confidence.tolist():
        if conf < benign_threshold:
            decisions.append("benign")
            uncertainty_flags.append(False)
        elif conf > malignant_threshold:
            decisions.append("malignant")
            uncertainty_flags.append(False)
        else:
            decisions.append("uncertain")
            uncertainty_flags.append(True)   # triggers radiologist review

    return decisions, uncertainty_flags
