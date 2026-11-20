"""
trainer.py — Two-stage training loop for Bloc 3.

Stage 1 (backbone FROZEN):
  Only fusion module and output heads are trained.
  Learning rate: 1e-4 for all trainable parameters.
  Runs until validation loss stabilizes (monitored, not fixed epochs).

Stage 2 (backbone UNFROZEN):
  Full model fine-tuning with differential learning rates:
    backbone:       1e-5  (very small — gentle specialization)
    fusion + heads: 1e-4  (moderate — still learning more)
  Same early stopping and LR scheduling as Stage 1.

Early stopping monitors TWO metrics simultaneously:
  (1) Validation total loss  → patience=10 epochs without improvement
  (2) Validation sensitivity → stops if sensitivity drops below 0.85
  Weights restored from BEST SENSITIVITY epoch (clinical > mathematical).

ReduceLROnPlateau:
  patience=5 (shorter than early stopping to give model a second chance
  before stopping) → reduce LR by 0.1 → minimum 1e-6.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional
from torch.utils.data import DataLoader

from models.bloc3       import Bloc3Model
from training.loss      import Bloc3Loss
from training.metrics   import collect_epoch_outputs
from config             import Config


class EarlyStopping:
    """
    Monitors validation total loss (patience-based) AND validation sensitivity
    (minimum threshold-based) simultaneously.

    Stops if EITHER:
      (a) total loss shows no improvement for `patience` consecutive epochs, OR
      (b) sensitivity drops below `min_sensitivity`.

    Saves best weights based on highest sensitivity (not lowest loss).
    """

    def __init__(self, patience: int = 10, min_sensitivity: float = 0.85):
        self.patience        = patience
        self.min_sensitivity = min_sensitivity

        self.best_sensitivity = -1.0
        self.best_loss        = float("inf")
        self.best_state_dict  = None
        self.epochs_no_improve = 0
        self.should_stop      = False

    def step(
        self,
        val_loss:        float,
        val_sensitivity: float,
        model_state:     dict,
    ) -> bool:
        """
        Update state. Returns True if training should stop.

        Args:
            val_loss:        Validation total loss for this epoch.
            val_sensitivity: Validation sensitivity for this epoch.
            model_state:     model.state_dict() snapshot.
        """
        # Save best weights based on sensitivity (clinical priority)
        if val_sensitivity > self.best_sensitivity:
            self.best_sensitivity = val_sensitivity
            self.best_state_dict  = {k: v.cpu().clone() for k, v in model_state.items()}

        # Loss-based patience counter
        if val_loss < self.best_loss - 1e-4:
            self.best_loss         = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        # Stop conditions
        if self.epochs_no_improve >= self.patience:
            print(f"[EarlyStopping] No loss improvement for {self.patience} epochs.")
            self.should_stop = True

        if val_sensitivity < self.min_sensitivity:
            print(f"[EarlyStopping] Sensitivity {val_sensitivity:.4f} "
                  f"< threshold {self.min_sensitivity}.")
            self.should_stop = True

        return self.should_stop

    def restore_best_weights(self, model: torch.nn.Module):
        """Load the best-sensitivity checkpoint back into the model."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            print(f"[EarlyStopping] Restored best weights "
                  f"(sensitivity={self.best_sensitivity:.4f}).")


class Trainer:
    """
    Orchestrates Stage 1 and Stage 2 training for Bloc 3.

    Usage:
        trainer = Trainer(model, loaders, cfg)
        trainer.run()
    """

    def __init__(
        self,
        model:   Bloc3Model,
        loaders: Dict[str, DataLoader],
        cfg:     Config,
    ):
        self.model   = model
        self.loaders = loaders
        self.cfg     = cfg
        self.device  = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Loss function
        self.criterion = Bloc3Loss(
            lambda_1      = cfg.training.lambda_1,
            lambda_2      = cfg.training.lambda_2,
            class_weights = (cfg.training.benign_class_weight,
                             cfg.training.malignant_class_weight),
            device        = str(self.device),
        )

        # Output directories
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.log_dir,    exist_ok=True)

        self.history: Dict[str, list] = {
            "epoch": [], "stage": [],
            "train_loss": [], "val_loss": [],
            "train_sensitivity": [], "val_sensitivity": [],
            "val_auc_roc": [], "val_f1": [], "val_mae": [],
        }

    # ─── Public entry point ───────────────────────────────────────────────────

    def run(self):
        """Run Stage 1 then Stage 2 training."""
        print(f"\n{'='*60}")
        print(f" Bloc 3 Training — {self.cfg.experiment_name}")
        print(f" Device: {self.device}")
        print(f"{'='*60}\n")

        self._run_stage(stage=1)
        self._run_stage(stage=2)

        self._save_history()
        print("\n[Trainer] Training complete.")

    # ─── Stage runner ─────────────────────────────────────────────────────────

    def _run_stage(self, stage: int):
        """Run one training stage with its own optimizer and early stopping."""
        print(f"\n{'─'*50}")
        print(f" Stage {stage} — backbone {'FROZEN' if stage == 1 else 'UNFROZEN'}")
        print(f"{'─'*50}")

        tcfg = self.cfg.training

        # ── Configure backbone freeze / unfreeze ──────────────────────────────
        if stage == 1:
            self.model.freeze_backbone()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr           = tcfg.stage1_lr,
                weight_decay = tcfg.weight_decay,
            )
            max_epochs = tcfg.stage1_max_epochs

        else:
            self.model.unfreeze_backbone()
            optimizer = torch.optim.Adam(
                self.model.get_param_groups(
                    backbone_lr  = tcfg.stage2_backbone_lr,
                    heads_lr     = tcfg.stage2_heads_lr,
                    weight_decay = tcfg.weight_decay,
                ),
            )
            max_epochs = tcfg.stage2_max_epochs

        # ── ReduceLROnPlateau ─────────────────────────────────────────────────
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode     = "min",
            patience = tcfg.lr_scheduler_patience,
            factor   = tcfg.lr_scheduler_factor,
            min_lr   = tcfg.lr_scheduler_min_lr,
            verbose  = True,
        )

        # ── Early stopping ────────────────────────────────────────────────────
        stopper = EarlyStopping(
            patience        = tcfg.early_stopping_patience,
            min_sensitivity = tcfg.min_sensitivity_threshold,
        )

        for epoch in range(1, max_epochs + 1):
            # ── Training epoch ────────────────────────────────────────────────
            train_loss, train_metrics = self._epoch(
                loader    = self.loaders["train"],
                optimizer = optimizer,
                train     = True,
            )

            # ── Validation epoch ──────────────────────────────────────────────
            val_loss, val_metrics = self._epoch(
                loader    = self.loaders["val"],
                optimizer = None,
                train     = False,
            )

            scheduler.step(val_loss)

            # ── Logging ───────────────────────────────────────────────────────
            self._log_epoch(stage, epoch, train_loss, val_loss,
                            train_metrics, val_metrics)

            # ── Early stopping check ──────────────────────────────────────────
            if stopper.step(val_loss, val_metrics["sensitivity"],
                            self.model.state_dict()):
                break

        stopper.restore_best_weights(self.model)
        self._save_checkpoint(stage)

    # ─── Epoch runner ─────────────────────────────────────────────────────────

    def _epoch(
        self,
        loader:    DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        train:     bool,
    ):
        """Run one full pass over a DataLoader. Returns (mean_loss, metrics)."""
        self.model.train(train)
        context = torch.enable_grad() if train else torch.no_grad()

        total_loss = 0.0
        all_true_labels, all_true_scores = [], []
        all_pred_probs,  all_pred_scores = [], []

        with context:
            for batch in tqdm(loader, desc="train" if train else "val ",
                              leave=False, ncols=80):
                patches, radiomics, labels, scores, _ = batch
                patches   = patches.to(self.device)
                radiomics = radiomics.to(self.device)
                labels    = labels.to(self.device)
                scores    = scores.to(self.device)

                pred_score, pred_probs = self.model(patches, radiomics)

                loss, _, _ = self.criterion(
                    pred_score, pred_probs, scores, labels
                )

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping prevents exploding gradients in Stage 2
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    optimizer.step()

                total_loss += loss.item()

                all_true_labels.append(labels.cpu())
                all_true_scores.append(scores.cpu())
                all_pred_probs.append(pred_probs[:, 1].detach().cpu())  # P(malignant)
                all_pred_scores.append(pred_score.squeeze(1).detach().cpu())

        mean_loss = total_loss / len(loader)
        metrics   = collect_epoch_outputs(
            all_true_labels, all_true_scores,
            all_pred_probs,  all_pred_scores,
        )
        return mean_loss, metrics

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _save_checkpoint(self, stage: int):
        """Save model weights after completing a stage."""
        path = os.path.join(
            self.cfg.output_dir,
            f"{self.cfg.experiment_name}_stage{stage}.pt"
        )
        torch.save(self.model.state_dict(), path)
        print(f"[Trainer] Checkpoint saved → {path}")

    def _save_history(self):
        """Save training history as CSV for plotting."""
        path = os.path.join(self.cfg.log_dir,
                            f"{self.cfg.experiment_name}_history.csv")
        pd.DataFrame(self.history).to_csv(path, index=False)
        print(f"[Trainer] History saved → {path}")

    def _log_epoch(self, stage, epoch, train_loss, val_loss,
                   train_m, val_m):
        """Print epoch summary and append to history."""
        print(
            f"[S{stage} E{epoch:03d}] "
            f"loss: {train_loss:.4f}/{val_loss:.4f}  "
            f"sens: {train_m['sensitivity']:.4f}/{val_m['sensitivity']:.4f}  "
            f"auc: {val_m['auc_roc']:.4f}  "
            f"f1: {val_m['f1_score']:.4f}  "
            f"mae: {val_m['mae']:.4f}"
            + ("  ✓" if val_m["threshold_pass"] else "")
        )

        self.history["epoch"].append(epoch)
        self.history["stage"].append(stage)
        self.history["train_loss"].append(round(train_loss, 5))
        self.history["val_loss"].append(round(val_loss, 5))
        self.history["train_sensitivity"].append(train_m["sensitivity"])
        self.history["val_sensitivity"].append(val_m["sensitivity"])
        self.history["val_auc_roc"].append(val_m["auc_roc"])
        self.history["val_f1"].append(val_m["f1_score"])
        self.history["val_mae"].append(val_m["mae"])
