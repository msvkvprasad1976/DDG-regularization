"""
ddg/models.py
=============
Neural network architectures used in the paper:
  - DDGPointNet2  : PointNet++ (SSG) with DDG regularization for ModelNet40.
  - DDGGIN        : Graph Isomorphism Network with DDG regularization for QM9.

Both models expose the final hidden layer representation X^(L) so the
DDGRegularizer can compute geometric energy penalties on it.

References:
  [5] Qi et al., "PointNet++," NIPS 2017.
  [8] Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    PointNetConv, fps, radius, global_max_pool,
    GINConv, global_add_pool, BatchNorm,
)
from torch_geometric.nn import MLP as PyGMLP
from torch_geometric.data import Data
from typing import Tuple, Optional


# =============================================================================
#  PointNet++ (SSG variant) with DDG hook
# =============================================================================

class SAModule(nn.Module):
    """
    Set Abstraction module for PointNet++.
    Uses furthest point sampling + ball query + PointNet local aggregation.
    """

    def __init__(self, ratio: float, r: float, nn_channels):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(
            local_nn=PyGMLP(nn_channels, act="relu", norm="batch_norm"),
            global_nn=None,
            add_self_loops=False,
        )

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx],
            max_num_neighbors=64,
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x_out = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos_out = pos[idx]
        batch_out = batch[idx]
        return x_out, pos_out, batch_out


class GlobalSAModule(nn.Module):
    """Global set abstraction — aggregates all points into one feature."""

    def __init__(self, nn_channels):
        super().__init__()
        self.nn = PyGMLP(nn_channels, act="relu", norm="batch_norm")

    def forward(self, x, pos, batch):
        x_in = torch.cat([x, pos], dim=-1)
        x_out = self.nn(x_in)
        x_global = global_max_pool(x_out, batch)  # (B, d)
        pos_out = x_global.new_zeros((x_global.size(0), 3))
        batch_out = torch.arange(x_global.size(0), device=batch.device)
        return x_global, pos_out, batch_out


class DDGPointNet2(nn.Module):
    """
    PointNet++ (SSG) with DDG Regularization hook.

    Architecture follows the SSG variant from Qi et al. (2017).
    The final hidden layer representation is exposed via self.latent
    so the DDGRegularizer can apply geometric energy penalties.

    Args:
        num_classes : Number of output classes (40 for ModelNet40).
        dropout     : Dropout probability in the classification head.
    """

    def __init__(self, num_classes: int = 40, dropout: float = 0.5):
        super().__init__()

        # SA layers (Set Abstraction)
        self.sa1 = SAModule(ratio=0.5, r=0.2,
                            nn_channels=[3 + 3, 64, 64, 128])
        self.sa2 = SAModule(ratio=0.25, r=0.4,
                            nn_channels=[128 + 3, 128, 128, 256])
        self.sa3 = GlobalSAModule(nn_channels=[256 + 3, 256, 512, 1024])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.out_layer = nn.Linear(256, num_classes)

        # Latent buffer (set during forward, used by DDGRegularizer)
        self.latent: Optional[torch.Tensor] = None

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data : PyG Data with data.pos (N*B, 3) and data.batch (N*B,).

        Returns:
            logits : (B, num_classes).
        """
        x0, pos0, batch0 = None, data.pos, data.batch

        # Hierarchical feature extraction
        x1, pos1, batch1 = self.sa1(x0, pos0, batch0)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        x3, pos3, batch3 = self.sa3(x2, pos2, batch2)  # (B, 1024)

        # Classification head — expose latent before final linear
        h = self.classifier(x3)          # (B, 256)
        self.latent = h                   # X^(L) for DDG regularizer
        logits = self.out_layer(h)        # (B, num_classes)
        return logits

    def get_latent(self) -> torch.Tensor:
        """Return the most recently computed latent representation X^(L)."""
        if self.latent is None:
            raise RuntimeError("Call forward() before get_latent().")
        return self.latent


# =============================================================================
#  Graph Isomorphism Network (GIN) with DDG hook
# =============================================================================

class GINLayer(nn.Module):
    """Single GIN convolution layer with residual connection."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.conv = GINConv(mlp, train_eps=True)
        self.bn   = BatchNorm(out_dim)
        self.res  = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.bn(out)
        out = F.relu(out + self.res(x))
        return out


class DDGGIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) with DDG Regularization hook for QM9.

    Architecture: 5 GIN layers → global sum pooling → MLP regression head.
    The final hidden layer representation is exposed via self.latent.

    Args:
        in_dim      : Input node feature dimension.
        hidden_dim  : Hidden channel size (default 256).
        out_dim     : Output dimension (1 for single property regression).
        num_layers  : Number of GIN layers (default 5).
        dropout     : Dropout in the regression head.
    """

    def __init__(
        self,
        in_dim: int = 11,
        hidden_dim: int = 256,
        out_dim: int = 1,
        num_layers: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.atom_encoder = nn.Linear(in_dim, hidden_dim)

        self.gin_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gin_layers.append(GINLayer(hidden_dim, hidden_dim))

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
        )
        self.out_layer = nn.Linear(hidden_dim // 4, out_dim)

        self.latent: Optional[torch.Tensor] = None

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data : PyG Data with data.x (node features), data.edge_index,
                   data.batch.

        Returns:
            pred : (B, out_dim) property predictions.
        """
        x = self.atom_encoder(data.x.float())

        for gin in self.gin_layers:
            x = gin(x, data.edge_index)

        # Global sum pooling → graph-level representation
        x_graph = global_add_pool(x, data.batch)   # (B, hidden_dim)

        # Regression head — expose latent
        h = self.regressor(x_graph)                # (B, hidden_dim//4)
        self.latent = h
        pred = self.out_layer(h)                   # (B, out_dim)
        return pred

    def get_latent(self) -> torch.Tensor:
        if self.latent is None:
            raise RuntimeError("Call forward() before get_latent().")
        return self.latent
