"""
train_qm9.py
============
Training script for QM9 dipole moment (μ) regression using
GIN with DDG Regularization.

Reproduces Table I results for QM9 (Section VI-B of the paper).

Usage:
    python train_qm9.py --use_ddg --lambda1 0.05 --lambda2 0.01
    python train_qm9.py --baseline l2
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.datasets import QM9
from torch_geometric.transforms import Compose, Distance
from torch_geometric.loader import DataLoader

from ddg.models import DDGGIN
from ddg.regularizer import DDGRegularizer
from ddg.trainer import Trainer


# ─── Reproducibility ─────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Target index for QM9 ─────────────────────────────────────────────────────
# QM9 target index 0 = dipole moment μ (Debye)
DIPOLE_IDX = 0


# ─── Metric: Mean Absolute Error ─────────────────────────────────────────────
def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred.squeeze() - target[:, DIPOLE_IDX]).abs().mean().item()


# ─── Data loading ─────────────────────────────────────────────────────────────
def get_qm9_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """
    Load QM9 dataset and split into train/val/test.
    Molecular graphs: atoms as vertices, bonds as edges.
    """
    transform = Distance(norm=True, cat=True)
    dataset = QM9(root=root, transform=transform)

    # Shuffle and split
    dataset = dataset.shuffle()
    N = len(dataset)
    n_train = int(N * train_ratio)
    n_val   = int(N * val_ratio)

    train_ds = dataset[:n_train]
    val_ds   = dataset[n_train:n_train + n_val]
    test_ds  = dataset[n_train + n_val:]

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    in_dim = dataset[0].x.shape[1]
    print(f"QM9 | N={N} | Train={n_train} | Val={n_val} | "
          f"Test={N - n_train - n_val} | NodeFeatures={in_dim}")

    return train_loader, val_loader, test_loader, in_dim


# ─── Batch-level Laplacian using molecular graph structure ────────────────────
def make_qm9_laplacian_fn(device):
    """
    For QM9, the molecular graph already has defined connectivity (bonds).
    We use the bond graph to construct the Laplacian directly from edge_index,
    rather than k-NN triangulation (which is only needed for unstructured
    point clouds like ModelNet40).
    """
    def laplacian_fn(batch):
        edge_index = batch.edge_index           # (2, E)
        N = batch.num_nodes

        row = edge_index[0]
        col = edge_index[1]

        # Symmetric degree-normalized Laplacian: L = D - A
        # Unweighted: w_ij = 1 for all bonds
        ones = torch.ones(row.shape[0], device=device)

        # Off-diagonal: L[i,j] = -1  (negative adjacency)
        off_val = -ones

        # Diagonal: L[i,i] = degree(i)
        degree = torch.zeros(N, device=device).scatter_add_(0, row, ones)

        diag_row = torch.arange(N, device=device)
        diag_val = degree

        all_row = torch.cat([row, diag_row])
        all_col = torch.cat([col, diag_row])
        all_val = torch.cat([off_val, diag_val])

        # Voronoi areas — use uniform 1.0 for molecular graphs
        areas = torch.ones(N, device=device)

        return all_row, all_col, all_val, areas

    return laplacian_fn


# ─── MAE criterion ───────────────────────────────────────────────────────────
class MAELoss(nn.Module):
    def __init__(self, target_idx: int = 0):
        super().__init__()
        self.target_idx = target_idx

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred.squeeze() - target[:, self.target_idx]).abs().mean()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading QM9...")
    train_loader, val_loader, test_loader, in_dim = get_qm9_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DDGGIN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=1,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    use_ddg = (args.baseline == "ddg") and args.use_ddg

    # ── Loss and regularizer ──────────────────────────────────────────────────
    criterion  = MAELoss(target_idx=DIPOLE_IDX)
    regularizer = DDGRegularizer(lambda1=args.lambda1, lambda2=args.lambda2)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        criterion=criterion,
        regularizer=regularizer,
        device=device,
        lr=args.lr,
        patience=args.patience,
        max_epochs=args.epochs,
        lr_step=50,
        use_ddg=use_ddg,
    )
    if args.baseline == "l2":
        for pg in trainer.optimizer.param_groups:
            pg["weight_decay"] = 1e-4

    # ── Laplacian from molecular graph ────────────────────────────────────────
    laplacian_fn = make_qm9_laplacian_fn(device) if use_ddg else None

    # ── Train ─────────────────────────────────────────────────────────────────
    save_path = os.path.join(args.out_dir, "qm9_best.pth")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        metric_fn=mae,
        metric_name="MAE(D)",
        laplacian_fn=laplacian_fn,
        save_path=save_path,
    )

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_stats = trainer.eval_epoch(test_loader, mae)

    print("\n" + "="*60)
    print(f"  FINAL RESULTS — QM9 Dipole Moment")
    print(f"  Test MAE     : {test_stats['metric']:.4f} Debye")
    print(f"  Convergence  : epoch {history.get('convergence_epoch', '—')}")
    print("="*60)

    # ── Save ──────────────────────────────────────────────────────────────────
    import json
    os.makedirs(args.out_dir, exist_ok=True)
    result = {
        "test_mae_debye": test_stats["metric"],
        "convergence_epoch": history.get("convergence_epoch"),
        "use_ddg": use_ddg,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "baseline": args.baseline,
    }
    with open(os.path.join(args.out_dir, "qm9_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {args.out_dir}/qm9_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DDG-regularized GIN on QM9 dipole moment"
    )
    # Data
    parser.add_argument("--data_root",   default="./data/QM9")
    parser.add_argument("--out_dir",     default="./outputs/qm9")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size",  type=int, default=64)
    # Model
    parser.add_argument("--hidden_dim",  type=int, default=256)
    parser.add_argument("--num_layers",  type=int, default=5)
    parser.add_argument("--dropout",     type=float, default=0.0)
    # Training
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--patience",    type=int,   default=20)
    parser.add_argument("--seed",        type=int,   default=42)
    # DDG
    parser.add_argument("--use_ddg",    action="store_true", default=True)
    parser.add_argument("--lambda1",    type=float, default=0.05)
    parser.add_argument("--lambda2",    type=float, default=0.01)
    # Baseline
    parser.add_argument("--baseline",   default="ddg",
                        choices=["ddg", "l2", "dropout", "none"])

    args = parser.parse_args()
    main(args)
