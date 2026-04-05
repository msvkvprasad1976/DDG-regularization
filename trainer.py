"""
ddg/trainer.py
==============
Generic training engine used for both ModelNet40 and QM9 experiments.

Implements the Algorithm 1 forward pass from the paper:
  1. Pre-compute Cotangent Laplacian L and Voronoi areas A  (once)
  2. For each training step:
     a. Forward pass → get latent X^(L)
     b. Compute L_task (cross-entropy or MAE)
     c. Compute E_D and E_W from X^(L), L, A
     d. L_total = L_task + λ1*E_D + λ2*E_W
     e. Backward + optimiser step
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Dict, List, Optional, Callable
import numpy as np

from .regularizer import DDGRegularizer


class Trainer:
    """
    Training engine for DDG-regularised models.

    Args:
        model      : Neural network (DDGPointNet2 or DDGGIN).
        criterion  : Task loss function.
        regularizer: DDGRegularizer instance.
        device     : Compute device.
        lr         : Initial learning rate (default 1e-3, as in paper).
        patience   : Early stopping patience in epochs (default 20).
        max_epochs : Maximum training epochs (default 200).
        lr_factor  : LR reduction factor every 50 epochs (default 0.5).
        lr_step    : Epoch interval for LR reduction (default 50).
        use_ddg    : Whether to apply DDG regularization (default True).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        regularizer: DDGRegularizer,
        device: torch.device,
        lr: float = 1e-3,
        patience: int = 20,
        max_epochs: int = 200,
        lr_factor: float = 0.5,
        lr_step: int = 50,
        use_ddg: bool = True,
    ):
        self.model       = model.to(device)
        self.criterion   = criterion
        self.regularizer = regularizer
        self.device      = device
        self.patience    = patience
        self.max_epochs  = max_epochs
        self.lr_step     = lr_step
        self.use_ddg     = use_ddg

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        # Step LR: halved every lr_step epochs (matches paper Section V-B)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=lr_factor,
            patience=lr_step, min_lr=1e-6,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_metric": [], "val_metric": [],
            "e_dirichlet": [], "e_willmore": [],
        }

    # ─── Dummy Laplacian for no-DDG baselines ─────────────────────────────────
    @staticmethod
    def _null_laplacian(device):
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, torch.zeros(0, device=device)

    def train_epoch(
        self,
        loader,
        metric_fn: Callable,
        laplacian_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        One training epoch.

        Args:
            loader       : DataLoader for training set.
            metric_fn    : Function(pred, target) → scalar metric value.
            laplacian_fn : Optional function(batch) → (L_row, L_col, L_val, areas).
                           If None or use_ddg=False, DDG is skipped.

        Returns:
            Dict with avg loss, avg metric, avg E_D, avg E_W.
        """
        self.model.train()
        total_loss = 0.0
        total_metric = 0.0
        total_ed = 0.0
        total_ew = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="  Train", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # ── Forward pass ──────────────────────────────────────────────────
            pred = self.model(batch)
            task_loss = self.criterion(pred, batch.y)

            # ── DDG regularization ────────────────────────────────────────────
            if self.use_ddg and laplacian_fn is not None:
                X = self.model.get_latent()           # X^(L)
                L_row, L_col, L_val, areas = laplacian_fn(batch)
                N = X.shape[0]
                total_loss_val, e_d, e_w = self.regularizer(
                    X, task_loss, L_row, L_col, L_val, areas, N
                )
                total_ed += e_d.item()
                total_ew += e_w.item()
            else:
                total_loss_val = task_loss

            # ── Backward + step ───────────────────────────────────────────────
            total_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            metric = metric_fn(pred.detach(), batch.y)
            total_loss   += total_loss_val.item()
            total_metric += metric
            n_batches    += 1

        return {
            "loss":        total_loss   / n_batches,
            "metric":      total_metric / n_batches,
            "e_dirichlet": total_ed     / n_batches,
            "e_willmore":  total_ew     / n_batches,
        }

    @torch.no_grad()
    def eval_epoch(
        self,
        loader,
        metric_fn: Callable,
    ) -> Dict[str, float]:
        """One evaluation epoch (no DDG penalty applied during validation)."""
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="  Eval ", leave=False):
            batch = batch.to(self.device)
            pred = self.model(batch)
            loss = self.criterion(pred, batch.y)
            metric = metric_fn(pred, batch.y)
            total_loss   += loss.item()
            total_metric += metric
            n_batches    += 1

        return {
            "loss":   total_loss   / n_batches,
            "metric": total_metric / n_batches,
        }

    def fit(
        self,
        train_loader,
        val_loader,
        metric_fn: Callable,
        metric_name: str = "metric",
        laplacian_fn: Optional[Callable] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.

        Args:
            train_loader  : Training DataLoader.
            val_loader    : Validation DataLoader.
            metric_fn     : Scalar metric function (accuracy or MAE).
            metric_name   : Display name for the metric.
            laplacian_fn  : Pre-computed Laplacian provider (optional).
            save_path     : Path to save the best model checkpoint.

        Returns:
            Training history dictionary.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        convergence_epoch = self.max_epochs

        print(f"\n{'='*60}")
        print(f"  DDG={'ON' if self.use_ddg else 'OFF'}  "
              f"λ1={self.regularizer.lambda1:.3f}  "
              f"λ2={self.regularizer.lambda2:.3f}")
        print(f"{'='*60}")

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()

            train_stats = self.train_epoch(train_loader, metric_fn, laplacian_fn)
            val_stats   = self.eval_epoch(val_loader, metric_fn)
            self.scheduler.step(val_stats["loss"])

            # Record history
            self.history["train_loss"].append(train_stats["loss"])
            self.history["val_loss"].append(val_stats["loss"])
            self.history["train_metric"].append(train_stats["metric"])
            self.history["val_metric"].append(val_stats["metric"])
            self.history["e_dirichlet"].append(train_stats["e_dirichlet"])
            self.history["e_willmore"].append(train_stats["e_willmore"])

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"TrainLoss={train_stats['loss']:.4f} | "
                f"Val{metric_name}={val_stats['metric']:.4f} | "
                f"E_D={train_stats['e_dirichlet']:.4f} | "
                f"E_W={train_stats['e_willmore']:.4f} | "
                f"t={elapsed:.1f}s"
            )

            # Early stopping
            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                patience_counter = 0
                convergence_epoch = epoch
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n  Early stopping at epoch {epoch}. "
                          f"Best val loss: {best_val_loss:.4f} "
                          f"(epoch {convergence_epoch})")
                    break

        self.history["convergence_epoch"] = convergence_epoch
        return self.history
