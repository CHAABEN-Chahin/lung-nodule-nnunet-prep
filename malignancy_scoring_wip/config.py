"""
config.py — Central configuration for Bloc 3: Malignancy Scoring Module.

All hyperparameters, paths, and architectural constants live here.
Change values here rather than hunting through training code.
"""

from dataclasses import dataclass, field


# ─── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    # Path to the downloaded LIDC-IDRI dataset (DICOM files)
    lidc_root: str = "data/LIDC-IDRI"

    # Where preprocessed .npz patch files will be saved by lidc_preprocessor.py
    preprocessed_dir: str = "data/preprocessed"

    # Fixed patch size fed to the backbone (D x H x W in voxels)
    patch_size: tuple = (64, 64, 64)

    # PyRadiomics extracts exactly this many features per nodule
    radiomics_dim: int = 93

    # Only keep nodules larger than this (millimeters)
    min_nodule_size_mm: float = 3.0

    # Patient-level split ratios (must sum to 1.0)
    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    test_ratio:  float = 0.15

    # Seed for reproducible splits and score-3 shuffling
    random_seed: int = 42

    # Number of DataLoader worker processes
    num_workers: int = 4


# ─── Model ────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # Backbone to use: 'models_genesis' or 'medicalnet_resnet18'
    # Only this changes between experiments — all other params stay fixed
    backbone: str = "models_genesis"

    # Path to pretrained backbone weights (.pth file)
    # Set to "" to use random initialization (for debugging only)
    backbone_weights_path: str = ""

    # Backbone output dimension (both backbones output 512d)
    cnn_features_dim: int = 512

    # PyRadiomics vector dimension
    radiomics_dim: int = 93

    # Fusion dimension = CNN features + radiomics (512 + 93 = 605)
    fusion_dim: int = 605  # must equal cnn_features_dim + radiomics_dim

    # Hidden dimension for MLP in fusion module and both output heads
    hidden_dim: int = 256

    # Dropout rate — applied to fusion MLP and heads ONLY, NOT to backbone
    # (backbone was already regularized during pretraining)
    dropout_rate: float = 0.3


# ─── Training ─────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    batch_size: int = 16

    # ── Loss weights ──────────────────────────────────────────────────────────
    # MSE (regression) and WeightedCrossEntropy (binary) operate on different
    # numerical scales — equal lambdas would let the larger-scale loss dominate.
    # lambda_2 > lambda_1 reflects clinical priority: binary decision matters
    # more than exact score precision, and sensitivity > specificity in oncology.
    lambda_1: float = 1.0   # weight for regression MSE loss
    lambda_2: float = 2.0   # weight for binary WeightedCrossEntropy loss

    # Higher weight on malignant misclassification — missing a malignant nodule
    # is clinically more dangerous than a false positive.
    benign_class_weight:    float = 1.0
    malignant_class_weight: float = 3.0

    # ── Stage 1: frozen backbone ───────────────────────────────────────────────
    # Fusion and heads learn to use pretrained features without disturbing backbone.
    stage1_lr:         float = 1e-4
    stage1_max_epochs: int   = 50

    # ── Stage 2: unfrozen backbone ─────────────────────────────────────────────
    # Differential learning rates are non-negotiable here.
    # Backbone gets a very small lr to specialize without forgetting pretraining.
    # Fusion + heads get a moderate lr as they still have more to learn.
    stage2_backbone_lr:  float = 1e-5
    stage2_heads_lr:     float = 1e-4
    stage2_max_epochs:   int   = 100

    # L2 weight decay on fusion + heads ONLY — backbone is already regularized
    weight_decay: float = 1e-4

    # ── Early Stopping ─────────────────────────────────────────────────────────
    # Monitors BOTH total loss AND sensitivity simultaneously.
    # Monitoring loss alone would miss silent degradation in the binary head.
    # Weights are restored from the best-SENSITIVITY epoch (clinical optimality
    # > mathematical optimality).
    early_stopping_patience:     int   = 10    # epochs without improvement
    min_sensitivity_threshold:   float = 0.85  # stop if sensitivity drops below

    # ── ReduceLROnPlateau ──────────────────────────────────────────────────────
    # Reacts to actual validation behavior — chosen over step decay (too rigid)
    # and cosine annealing (conflicts with early stopping variable duration).
    lr_scheduler_patience:  int   = 5    # shorter than early stopping patience
    lr_scheduler_factor:    float = 0.1
    lr_scheduler_min_lr:    float = 1e-6


# ─── Inference / Decision Thresholds ──────────────────────────────────────────

@dataclass
class InferenceConfig:
    # The uncertainty flag is the core clinical safety mechanism.
    # Threshold set below 0.5 to prioritize sensitivity over specificity.
    #
    # confidence < benign_threshold    → Benign,    flag OFF
    # benign_threshold ≤ conf ≤ malign → Uncertain, flag ON  (radiologist review)
    # confidence > malign_threshold    → Malignant, flag OFF
    benign_threshold:    float = 0.30
    malignant_threshold: float = 0.70


# ─── Top-level Config ─────────────────────────────────────────────────────────

@dataclass
class Config:
    data:      DataConfig      = field(default_factory=DataConfig)
    model:     ModelConfig     = field(default_factory=ModelConfig)
    training:  TrainingConfig  = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Experiment tracking
    experiment_name: str = "bloc3_baseline"
    output_dir:      str = "outputs"
    log_dir:         str = "logs"

    # 'cuda' for GPU training, 'cpu' for debugging only
    device: str = "cuda"


def get_config() -> Config:
    """Return the default configuration. Modify fields as needed before passing to Trainer."""
    return Config()
