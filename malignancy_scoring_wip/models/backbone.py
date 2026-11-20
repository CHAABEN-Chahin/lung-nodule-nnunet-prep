"""
backbone.py — 3D CNN Backbone for Bloc 3.

Two backbones are supported for comparison under identical experimental
conditions (only the backbone changes between experiments):

  A) Models Genesis  — ResNet3D, self-supervised pretraining by restoring
                       corrupted medical images; trained partly on LIDC data.
     Weights: https://github.com/MrGiovanni/ModelsGenesis

  B) MedicalNet ResNet18 — ResNet18 3D, supervised pretraining on 8 medical
                            segmentation datasets with expert annotations.
     Weights: https://github.com/Tencent/MedicalNet

Both output a 512-dimensional feature vector, ensuring the fusion module and
output heads are architecturally identical across experiments for a fair,
scientifically meaningful comparison.

Architecture choice rationale:
  ResNet10 → too shallow, underfitting risk on 2635 nodules.
  ResNet18 → chosen: right capacity for dataset size.
  ResNet50 → too deep, overfitting risk on 2635 nodules.
"""

import torch
import torch.nn as nn
from typing import Optional


# ─── 3D ResNet Building Blocks ────────────────────────────────────────────────

class BasicBlock3D(nn.Module):
    """
    Standard 3D residual block for ResNet-18/34.
    Two 3×3×3 convolutions with a shortcut connection.
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample   # for matching dimensions on skip path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out      = self.conv1(x)
        out      = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class ResNet3D18(nn.Module):
    """
    3D ResNet-18 encoder.

    Matches the architecture used by MedicalNet ResNet18 and is compatible
    with Models Genesis encoder weights (both ResNet-based).

    Input:  (B, 1, D, H, W) — single-channel 3D CT patch
    Output: (B, 512)         — feature vector via Global Average Pooling
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # Stem: large 7×7×7 conv → MaxPool (same as 2D ResNet-18 stem)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Residual layers (channel progression: 64 → 128 → 256 → 512)
        self.layer1 = self._make_layer(64,  64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)

        # Collapse spatial dimensions → 512d vector
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self._init_weights()

    def _make_layer(self, in_ch: int, out_ch: int,
                    n_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        layers = [BasicBlock3D(in_ch, out_ch, stride, downsample)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock3D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming initialization for conv layers (used when no pretrained weights)."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W)

        Returns:
            features: (B, 512)
        """
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.flatten(1)   # (B, 512)
        return x


# ─── Weight loading helpers ───────────────────────────────────────────────────

def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    """Remove a key prefix from a state dict (e.g. 'module.' from DataParallel)."""
    return {k[len(prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)}


def load_models_genesis(weights_path: Optional[str] = None) -> ResNet3D18:
    """
    Load Models Genesis pretrained backbone.

    Models Genesis was trained via self-supervised learning by restoring
    corrupted 3D medical images, including data from LIDC.

    Pretrained weights: https://github.com/MrGiovanni/ModelsGenesis
    Download: genesis_chest_ct.pt (or similar filename from their releases)

    Args:
        weights_path: Local path to the .pt / .pth file.
                      None → random initialisation (debug only).
    """
    backbone = ResNet3D18(in_channels=1)

    if not weights_path:
        print("[Backbone] Models Genesis — no weights path, using random init.")
        return backbone

    state_dict = torch.load(weights_path, map_location="cpu")

    # Models Genesis saves a full encoder-decoder; extract encoder part.
    # Try common key patterns (varies slightly by release version).
    for prefix in ("encoder.", "module.encoder.", "module.", ""):
        candidate = _strip_prefix(state_dict, prefix) if prefix else state_dict
        result    = backbone.load_state_dict(candidate, strict=False)
        if len(result.missing_keys) < len(backbone.state_dict()):
            print(f"[Backbone] Models Genesis loaded (prefix='{prefix}'). "
                  f"Missing: {len(result.missing_keys)}, "
                  f"Unexpected: {len(result.unexpected_keys)}")
            break

    return backbone


def load_medicalnet_resnet18(weights_path: Optional[str] = None) -> ResNet3D18:
    """
    Load MedicalNet ResNet18 pretrained backbone.

    MedicalNet ResNet18 was supervised-pretrained on 8 medical segmentation
    datasets with expert annotations.

    Pretrained weights: https://github.com/Tencent/MedicalNet
    Download: resnet_18.pth from their release assets.

    Args:
        weights_path: Local path to the .pth file.
                      None → random initialisation (debug only).
    """
    backbone = ResNet3D18(in_channels=1)

    if not weights_path:
        print("[Backbone] MedicalNet ResNet18 — no weights path, using random init.")
        return backbone

    state_dict = torch.load(weights_path, map_location="cpu")

    # MedicalNet may wrap weights in DataParallel 'module.' prefix
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = _strip_prefix(state_dict, "module.")

    # MedicalNet saves the full model including a classification head —
    # strict=False lets us ignore the final FC layer (we don't use it).
    result = backbone.load_state_dict(state_dict, strict=False)
    print(f"[Backbone] MedicalNet ResNet18 loaded. "
          f"Missing: {len(result.missing_keys)}, "
          f"Unexpected: {len(result.unexpected_keys)}")

    return backbone


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_backbone(name: str, weights_path: Optional[str] = None) -> ResNet3D18:
    """
    Factory function — returns the chosen pretrained backbone.

    Args:
        name:          'models_genesis' or 'medicalnet_resnet18'
        weights_path:  Path to pretrained .pth file.

    Returns:
        ResNet3D18 encoder with (optionally) loaded pretrained weights.
    """
    if name == "models_genesis":
        return load_models_genesis(weights_path)
    elif name == "medicalnet_resnet18":
        return load_medicalnet_resnet18(weights_path)
    else:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            "Choose 'models_genesis' or 'medicalnet_resnet18'."
        )
