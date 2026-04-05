"""
ddg/regularizer.py
==================
Implements the DDG Regularization loss described in Section IV-B of the paper.

Composite loss:
    L_total = L_task + λ1 * E_Dirichlet + λ2 * E_Willmore

Dirichlet Energy (Eq. 4):
    E_D(X^(L)) = 0.5 * (X^(L))^T L X^(L)

Willmore Energy (Eq. 5):
    E_W(X^(L)) ≈ Σ_v ||( Δ X^(L) )_v||² * A_v
               = Σ_v ||(L X^(L))_v||² * A_v

Reference: Meyer et al., Springer 2003; Willmore, Oxford Univ. Press 1993.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .mesh_construction import sparse_laplacian_mm


class DDGRegularizer(nn.Module):
    """
    DDG Regularization layer.

    Computes Dirichlet and Willmore energy penalties on the latent
    representation X^(L) using the pre-computed Cotangent Laplacian L
    and Voronoi area vector A.

    Args:
        lambda1 : Weight for Dirichlet energy  (λ₁).
        lambda2 : Weight for Willmore energy   (λ₂).
    """

    def __init__(self, lambda1: float = 0.05, lambda2: float = 0.01):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def dirichlet_energy(
        self,
        X: torch.Tensor,
        L_row: torch.Tensor,
        L_col: torch.Tensor,
        L_val: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """
        Discrete Dirichlet Energy: E_D = 0.5 * Tr(X^T L X)

        Penalises high-frequency oscillations in the latent space.
        Adjacent vertices on the manifold are encouraged to have
        similar representations.

        Args:
            X            : (N, d) latent feature matrix.
            L_row, L_col, L_val : Sparse COO Laplacian.
            N            : Number of vertices.

        Returns:
            Scalar Dirichlet energy.
        """
        LX = sparse_laplacian_mm(L_row, L_col, L_val, N, X)  # (N, d)
        # E_D = 0.5 * Tr(X^T L X) = 0.5 * sum(X * LX)
        energy = 0.5 * (X * LX).sum()
        return energy

    def willmore_energy(
        self,
        X: torch.Tensor,
        L_row: torch.Tensor,
        L_col: torch.Tensor,
        L_val: torch.Tensor,
        areas: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """
        Discrete Willmore Energy: E_W = Σ_v ||(ΔX)_v||² * A_v
                                      = Σ_v ||(LX)_v||² * A_v

        The Laplacian approximates the mean curvature normal H.
        Penalising ||H||² * A prevents the manifold representation
        from developing sharp topological folds during optimisation.

        Args:
            X            : (N, d) latent feature matrix.
            L_row, L_col, L_val : Sparse COO Laplacian.
            areas        : (N,) Voronoi area per vertex.
            N            : Number of vertices.

        Returns:
            Scalar Willmore energy.
        """
        LX = sparse_laplacian_mm(L_row, L_col, L_val, N, X)  # (N, d)
        H_sq = (LX * LX).sum(dim=-1)                           # (N,) = ||LX_v||²
        energy = (H_sq * areas).sum()                          # Σ_v ||LX_v||² * A_v
        return energy

    def forward(
        self,
        X: torch.Tensor,
        task_loss: torch.Tensor,
        L_row: torch.Tensor,
        L_col: torch.Tensor,
        L_val: torch.Tensor,
        areas: torch.Tensor,
        N: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the composite DDG-regularised loss (Eq. 3):
            L_total = L_task + λ₁ E_D + λ₂ E_W

        Args:
            X         : (N, d) latent representation at final hidden layer.
            task_loss : Scalar primary task loss (cross-entropy or MAE).
            L_row, L_col, L_val : Sparse COO Cotangent Laplacian.
            areas     : (N,) Voronoi area per vertex.
            N         : Number of vertices.

        Returns:
            total_loss  : Composite loss scalar.
            e_dirichlet : Dirichlet energy (for logging).
            e_willmore  : Willmore energy  (for logging).
        """
        e_d = self.dirichlet_energy(X, L_row, L_col, L_val, N)
        e_w = self.willmore_energy(X, L_row, L_col, L_val, areas, N)

        total = task_loss + self.lambda1 * e_d + self.lambda2 * e_w
        return total, e_d.detach(), e_w.detach()


class StabilityScore:
    """
    Computes the Stability Score S = 1 / (1 + σ²_ε) as defined in
    Section VI-A of the paper.

    Measures output robustness under controlled Gaussian vertex noise
    ε ~ N(0, σ²_noise). A score near 1.0 indicates low variance.
    """

    def __init__(self, sigma_noise: float = 0.02, n_trials: int = 5):
        self.sigma_noise = sigma_noise
        self.n_trials = n_trials

    @torch.no_grad()
    def compute(
        self,
        model: nn.Module,
        batch,
        device: torch.device,
        criterion,
    ) -> float:
        """
        Compute stability score over n_trials noisy perturbations.

        Args:
            model     : Trained model.
            batch     : A single data batch (PyG Data object).
            device    : Compute device.
            criterion : Loss function.

        Returns:
            Stability score in (0, 1].
        """
        model.eval()
        losses = []
        batch = batch.to(device)

        for _ in range(self.n_trials):
            noise = torch.randn_like(batch.pos) * self.sigma_noise
            noisy_batch = batch.clone()
            noisy_batch.pos = batch.pos + noise

            out = model(noisy_batch)
            loss = criterion(out, batch.y).item()
            losses.append(loss)

        variance = float(torch.tensor(losses).var().item())
        score = 1.0 / (1.0 + variance)
        return score
