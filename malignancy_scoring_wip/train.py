"""
train.py — Main entry point for Bloc 3 training.

Usage:
    # Train with Models Genesis backbone (default):
    python train.py

    # Train with MedicalNet ResNet18:
    python train.py --backbone medicalnet_resnet18 \\
                    --weights  weights/resnet_18.pth

    # Custom experiment name (for comparison logging):
    python train.py --backbone models_genesis \\
                    --weights  weights/genesis_chest_ct.pt \\
                    --name     bloc3_genesis_run1

Workflow:
  1. Load metadata.json and splits.json (produced by scripts/preprocess_lidc.py).
  2. Build DataLoaders.
  3. Build Bloc3Model with chosen backbone.
  4. Run Stage 1 (frozen backbone) then Stage 2 (unfrozen backbone).
  5. Evaluate on test set and print final metrics.
  6. Save results to outputs/ and training history to logs/.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np

from config              import get_config
from data.lidc_preprocessor import load_metadata
from data.dataset        import build_dataloaders
from models.bloc3        import Bloc3Model
from training.trainer    import Trainer
from training.metrics    import compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Train Bloc 3 — Malignancy Scoring Module")
    p.add_argument("--backbone",    type=str, default="models_genesis",
                   choices=["models_genesis", "medicalnet_resnet18"],
                   help="Pretrained backbone to use.")
    p.add_argument("--weights",     type=str, default="",
                   help="Path to pretrained backbone weights (.pth file).")
    p.add_argument("--data_dir",    type=str, default="data/preprocessed",
                   help="Directory with .npz files and metadata.json.")
    p.add_argument("--name",        type=str, default=None,
                   help="Experiment name (default: bloc3_{backbone}).")
    p.add_argument("--device",      type=str, default="cuda",
                   help="Compute device: 'cuda' or 'cpu'.")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--lambda1",     type=float, default=1.0,
                   help="Weight for regression MSE loss.")
    p.add_argument("--lambda2",     type=float, default=2.0,
                   help="Weight for binary classification loss.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Configuration ──────────────────────────────────────────────────────────
    cfg = get_config()
    cfg.model.backbone              = args.backbone
    cfg.model.backbone_weights_path = args.weights
    cfg.training.lambda_1           = args.lambda1
    cfg.training.lambda_2           = args.lambda2
    cfg.training.batch_size         = args.batch_size
    cfg.device                      = args.device
    cfg.experiment_name             = args.name or f"bloc3_{args.backbone}"

    print(f"\n{'='*60}")
    print(f" Bloc 3 — Malignancy Scoring Module")
    print(f" Backbone  : {cfg.model.backbone}")
    print(f" Weights   : {cfg.model.backbone_weights_path or 'random init'}")
    print(f" Experiment: {cfg.experiment_name}")
    print(f"{'='*60}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    metadata = load_metadata(args.data_dir)

    splits_path = os.path.join(args.data_dir, "splits.json")
    with open(splits_path) as f:
        splits = json.load(f)

    loaders = build_dataloaders(
        metadata    = metadata,
        train_ids   = splits["train_patient_ids"],
        val_ids     = splits["val_patient_ids"],
        test_ids    = splits["test_patient_ids"],
        batch_size  = cfg.training.batch_size,
        num_workers = cfg.data.num_workers,
        random_seed = cfg.data.random_seed,
    )

    # ── Build model ────────────────────────────────────────────────────────────
    model = Bloc3Model(cfg.model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {total_params:,}")

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = Trainer(model, loaders, cfg)
    trainer.run()

    # ── Final test evaluation ──────────────────────────────────────────────────
    print("\n[Evaluation] Running test set evaluation...")
    model.eval()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_true_labels, all_true_scores = [], []
    all_pred_probs,  all_pred_scores = [], []

    with torch.no_grad():
        for patches, radiomics, labels, scores, _ in loaders["test"]:
            patches   = patches.to(device)
            radiomics = radiomics.to(device)
            pred_score, pred_probs = model(patches, radiomics)

            all_true_labels.append(labels.cpu())
            all_true_scores.append(scores.cpu())
            all_pred_probs.append(pred_probs[:, 1].cpu())
            all_pred_scores.append(pred_score.squeeze(1).cpu())

    test_metrics = compute_metrics(
        true_labels  = torch.cat(all_true_labels).numpy(),
        true_scores  = torch.cat(all_true_scores).numpy(),
        pred_probs   = torch.cat(all_pred_probs).numpy(),
        pred_scores  = torch.cat(all_pred_scores).numpy(),
    )

    print("\n" + "="*50)
    print(" TEST SET RESULTS")
    print("="*50)
    for k, v in test_metrics.items():
        if k == "threshold_pass":
            status = "PASS ✓" if v else "FAIL ✗"
            print(f"  All thresholds: {status}")
        else:
            print(f"  {k:<20}: {v:.4f}")
    print("="*50)

    # Save test results
    results_path = os.path.join(
        cfg.output_dir, f"{cfg.experiment_name}_test_results.json"
    )
    with open(results_path, "w") as f:
        json.dump({**test_metrics, "backbone": args.backbone}, f, indent=2)
    print(f"\n[Results] Saved → {results_path}")


if __name__ == "__main__":
    main()
