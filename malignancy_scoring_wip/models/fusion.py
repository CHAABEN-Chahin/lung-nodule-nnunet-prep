"""
fusion.py — Fusion Module for Bloc 3.

Combines the 512d CNN feature vector (from backbone) with the 93d radiomics
vector (from PyRadiomics / Bloc 2) into a single 605d representation.

Why separate pathways?
  The 3D patch has spatial structure → must go through a CNN.
  The radiomics vector is a flat set of handcrafted numerical descriptors
  (size, shape, texture, margins, location) with no spatial relationship
  between values → feeding it into a CNN would be meaningless.

Fusion strategy (phased approach from design doc):
  Phase 1 (current): Simple concatenation [512d + 93d] = 605d → MLP
                     Implement first — establish a baseline.
  Phase 2 (future):  Attention-based fusion if Phase 1 baseline is solid
                     and dataset size supports it.

  Attention-based fusion is NOT implemented now because:
    - Harder to train, requires more data for reliable attention weights.
    - Less clinically interpretable.
    - More prone to overfitting on ~1844 training nodules.
"""

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """
    Phase 1 fusion: concatenate CNN features and radiomics, then process
    through a 2-layer MLP with BatchNorm, ReLU, and Dropout.

    Input:
        cnn_features: (B, 512) — output from backbone
        radiomics:    (B,  93) — PyRadiomics feature vector
    Output:
        fused:        (B, hidden_dim) — combined representation

    Note: Dropout is applied here (as specified in the design doc).
          It is NOT applied in the backbone.
    """

    def __init__(
        self,
        cnn_dim:     int   = 512,
        radio_dim:   int   = 93,
        hidden_dim:  int   = 256,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        fusion_dim = cnn_dim + radio_dim   # 605

        # Two-layer MLP that processes the concatenated vector
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(
        self,
        cnn_features: torch.Tensor,
        radiomics:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cnn_features: (B, 512)
            radiomics:    (B, 93)

        Returns:
            fused: (B, hidden_dim)
        """
        # Concatenate along feature dimension → (B, 605)
        combined = torch.cat([cnn_features, radiomics], dim=1)

        # Process through MLP → (B, hidden_dim)
        fused = self.mlp(combined)
        return fused
