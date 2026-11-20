"""
bloc3.py — Full Bloc 3 model: assembles backbone + fusion + heads.

Data flow:
    3D patch (B,1,D,H,W)  ──► Backbone ──────────────► (B, 512)
                                                              │
    Radiomics  (B, 93)    ──────────────────────────► Fusion ─► (B, hidden_dim)
                                                              │
                                             ┌────────────────┴──────────────────┐
                                       RegressionHead                      BinaryHead
                                        score (B,1)               probs (B,2) incl. score
                                       [1.0 → 5.0]              [P(benign), P(malignant)]
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .backbone import get_backbone, ResNet3D18
from .fusion   import FusionModule
from .heads    import RegressionHead, BinaryHead, make_decision
from config    import ModelConfig


class Bloc3Model(nn.Module):
    """
    Full Bloc 3 malignancy scoring model.

    Wraps backbone + fusion + regression head + binary head into a single
    nn.Module. The backbone can be frozen (Stage 1) or unfrozen (Stage 2).

    Args:
        cfg: ModelConfig dataclass with all architectural hyperparameters.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        # ── Backbone (pretrained 3D CNN) ──────────────────────────────────────
        self.backbone = get_backbone(
            name         = cfg.backbone,
            weights_path = cfg.backbone_weights_path or None,
        )

        # ── Fusion module ─────────────────────────────────────────────────────
        self.fusion = FusionModule(
            cnn_dim      = cfg.cnn_features_dim,
            radio_dim    = cfg.radiomics_dim,
            hidden_dim   = cfg.hidden_dim,
            dropout_rate = cfg.dropout_rate,
        )

        # ── Output heads ──────────────────────────────────────────────────────
        self.regression_head = RegressionHead(
            hidden_dim   = cfg.hidden_dim,
            dropout_rate = cfg.dropout_rate,
        )
        self.binary_head = BinaryHead(
            hidden_dim   = cfg.hidden_dim,
            dropout_rate = cfg.dropout_rate,
        )

        self.cfg = cfg

    # ─── Training forward pass ────────────────────────────────────────────────

    def forward(
        self,
        patch:     torch.Tensor,
        radiomics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass used during training and evaluation.

        Args:
            patch:     (B, 1, D, H, W) — normalized CT patch
            radiomics: (B, 93)          — radiomic feature vector

        Returns:
            score: (B, 1)  — regression malignancy score [1.0, 5.0]
            probs: (B, 2)  — [P(benign), P(malignant)]
        """
        # 1. Extract spatial features from 3D patch
        cnn_features = self.backbone(patch)            # (B, 512)

        # 2. Fuse CNN features with radiomics
        fused = self.fusion(cnn_features, radiomics)   # (B, hidden_dim)

        # 3. Regression head → continuous score
        score = self.regression_head(fused)            # (B, 1)

        # 4. Binary head — also receives regression score for consistency
        probs = self.binary_head(fused, score)         # (B, 2)

        return score, probs

    # ─── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        patch:               torch.Tensor,
        radiomics:           torch.Tensor,
        benign_threshold:    float = 0.30,
        malignant_threshold: float = 0.70,
    ) -> Dict:
        """
        Full inference: forward pass + clinical decision + uncertainty flag.

        Returns a dict matching the Bloc 3 output format (Section 11.4):
            malignancy_score: float  — 1.0–5.0
            confidence:       float  — P(malignant)
            decision:         str    — 'benign' | 'malignant' | 'uncertain'
            uncertainty_flag: bool
            cnn_features:     tensor — 512d features (passed to Bloc 4 if needed)
        """
        self.eval()
        score, probs = self.forward(patch, radiomics)

        confidence   = probs[:, 1]   # P(malignant), shape (B,)
        decisions, flags = make_decision(
            confidence,
            benign_threshold    = benign_threshold,
            malignant_threshold = malignant_threshold,
        )

        # Also expose backbone features for Bloc 4 (LLM diagnostic layer)
        cnn_features = self.backbone(patch)

        return {
            "malignancy_score":  score.squeeze(1).cpu().tolist(),
            "confidence":        confidence.cpu().tolist(),
            "decision":          decisions,
            "uncertainty_flag":  flags,
            "cnn_features":      cnn_features.cpu(),
        }

    # ─── Stage management (two-stage training) ────────────────────────────────

    def freeze_backbone(self):
        """
        Stage 1: freeze backbone weights entirely.
        Fusion and heads learn to use pretrained features without
        disturbing the backbone's learned representations.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Bloc3] Backbone FROZEN — Stage 1 training.")

    def unfreeze_backbone(self):
        """
        Stage 2: unfreeze backbone for gentle fine-tuning.
        Use a much smaller learning rate for backbone than for fusion+heads
        (1e-5 vs 1e-4) to avoid catastrophic forgetting of pretrained knowledge.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Bloc3] Backbone UNFROZEN — Stage 2 fine-tuning.")

    def get_param_groups(
        self,
        backbone_lr: float,
        heads_lr:    float,
        weight_decay: float,
    ) -> list:
        """
        Return parameter groups for Stage 2 optimizer with differential
        learning rates.

        Backbone:         small lr (backbone_lr)  — gentle fine-tuning
        Fusion + heads:   larger lr (heads_lr)    — still learning more

        Note: weight_decay is applied to fusion + heads only.
              The backbone was already regularized during pretraining, and
              its very small learning rate acts as implicit regularization.
        """
        return [
            {
                "params":       self.backbone.parameters(),
                "lr":           backbone_lr,
                "weight_decay": 0.0,       # no extra regularization on backbone
            },
            {
                "params": list(self.fusion.parameters()) +
                          list(self.regression_head.parameters()) +
                          list(self.binary_head.parameters()),
                "lr":           heads_lr,
                "weight_decay": weight_decay,
            },
        ]
