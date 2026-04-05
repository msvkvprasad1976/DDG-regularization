"""
DDG Regularization Package
==========================
Implements Discrete Differential Geometry regularization for deep learning
on manifold-valued data, as described in:

  "Discrete Differential Geometry as a Regularizer for Deep Learning on
   Manifold-Valued Data"
  Dr. Bomma Rama Krishna
  Swarnandhra College of Engineering and Technology (Autonomous)
"""

from .mesh_construction import (
    build_knn_mesh,
    cotangent_laplacian,
    precompute_mesh,
    sparse_laplacian_mm,
)
from .regularizer import DDGRegularizer, StabilityScore
from .models import DDGPointNet2, DDGGIN
from .trainer import Trainer

__all__ = [
    "build_knn_mesh",
    "cotangent_laplacian",
    "precompute_mesh",
    "sparse_laplacian_mm",
    "DDGRegularizer",
    "StabilityScore",
    "DDGPointNet2",
    "DDGGIN",
    "Trainer",
]
