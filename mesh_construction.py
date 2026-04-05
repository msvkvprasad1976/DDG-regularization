"""
ddg/mesh_construction.py
========================
Constructs a triangulated mesh from an unstructured point cloud using
k-nearest-neighbor (k-NN) graph triangulation, then computes the sparse
Cotangent Laplacian matrix and Voronoi area vector.

Reference: Meyer et al., "Discrete Differential-Geometry Operators for
           Triangulated 2-Manifolds," Visualization and Mathematics III,
           Springer, 2003, pp. 35-57.
           Wang et al., "Dynamic Graph CNN for Learning on Point Clouds,"
           ACM Trans. Graph., vol. 38, no. 5, 2019.
"""

import torch
import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from torch_sparse import SparseTensor
from typing import Tuple


def build_knn_mesh(
    points: torch.Tensor,
    k: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a triangulated mesh from a point cloud via k-NN + Delaunay.

    For each point, find its k nearest neighbours, then apply Delaunay
    triangulation within that local neighbourhood to recover face topology.
    This is the preprocessing protocol described in Section IV-A of the paper.

    Args:
        points : (N, 3) float tensor — 3D point cloud coordinates.
        k      : Number of nearest neighbours (default 20).

    Returns:
        vertices  : (N, 3) numpy array of vertex positions.
        edges     : (E, 2) numpy array of undirected edge indices.
        faces     : (F, 3) numpy array of triangle face indices.
    """
    pts = points.cpu().numpy().astype(np.float64)
    N = pts.shape[0]

    # ── Step 1: k-NN graph ────────────────────────────────────────────────────
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(pts)
    _, indices = nbrs.kneighbors(pts)          # (N, k+1); first col is self
    indices = indices[:, 1:]                   # remove self → (N, k)

    # ── Step 2: local Delaunay triangulation per vertex ───────────────────────
    face_set = set()
    edge_set = set()

    for i in range(N):
        local_idx = np.concatenate([[i], indices[i]])  # vertex i + its k neighbours
        local_pts = pts[local_idx]                      # (k+1, 3)

        # Project onto local tangent plane (PCA to 2D) for Delaunay
        centroid = local_pts.mean(axis=0)
        centered = local_pts - centroid

        # SVD for principal components
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        local_2d = centered @ Vt[:2].T                 # (k+1, 2)

        try:
            tri = Delaunay(local_2d)
        except Exception:
            continue

        for simplex in tri.simplices:
            # Map local indices back to global indices
            global_face = tuple(sorted(local_idx[simplex]))
            if len(set(global_face)) == 3:
                face_set.add(global_face)
                for a, b in [(0,1),(1,2),(0,2)]:
                    edge_set.add((min(global_face[a], global_face[b]),
                                  max(global_face[a], global_face[b])))

    faces = np.array(list(face_set), dtype=np.int64)   # (F, 3)
    edges = np.array(list(edge_set), dtype=np.int64)   # (E, 2)

    return pts, edges, faces


def cotangent_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_vertices: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the sparse symmetric Cotangent Laplacian matrix L and the
    Voronoi area vector A from a triangle mesh.

    The cotangent weight for edge (i, j) is:
        w_ij = (cot α_ij + cot β_ij) / 2
    where α_ij and β_ij are the angles opposite to edge (i,j) in the
    two triangles sharing that edge.

    The Laplacian entry is:
        L_ij = -w_ij          for i ≠ j
        L_ii = sum_j w_ij     (positive diagonal)

    Args:
        vertices   : (N, 3) numpy array of 3D vertex positions.
        faces      : (F, 3) numpy array of face vertex indices.
        n_vertices : Number of vertices N.

    Returns:
        row_idx : (M,) long tensor — sparse row indices.
        col_idx : (M,) long tensor — sparse col indices.
        values  : (M,) float tensor — sparse Laplacian values.
        areas   : (N,) float tensor — Voronoi area per vertex.
    """
    N = n_vertices
    F = faces.shape[0]

    row_list, col_list, val_list = [], [], []
    areas = np.zeros(N, dtype=np.float64)

    for f_idx in range(F):
        i, j, k = faces[f_idx]
        vi = vertices[i]
        vj = vertices[j]
        vk = vertices[k]

        # Edge vectors
        eij = vj - vi
        eik = vk - vi
        ejk = vk - vj
        eji = vi - vj
        eki = vi - vk
        ekj = vj - vk

        # Cotangent weights for each angle of the triangle
        # Angle at vertex i → opposite edge jk → contributes to w_jk
        cos_i = np.dot(eij, eik)
        sin_i = np.linalg.norm(np.cross(eij, eik))
        cot_i = cos_i / (sin_i + 1e-8)

        # Angle at vertex j → opposite edge ik → contributes to w_ik
        cos_j = np.dot(eji, ejk)
        sin_j = np.linalg.norm(np.cross(eji, ejk))
        cot_j = cos_j / (sin_j + 1e-8)

        # Angle at vertex k → opposite edge ij → contributes to w_ij
        cos_k = np.dot(eki, ekj)
        sin_k = np.linalg.norm(np.cross(eki, ekj))
        cot_k = cos_k / (sin_k + 1e-8)

        # Triangle area (half cross-product magnitude)
        face_area = 0.5 * np.linalg.norm(np.cross(eij, eik))

        # Voronoi area contribution (barycentric approximation)
        # Use equal 1/3 barycentric area per vertex for obtuse triangles
        areas[i] += face_area / 3.0
        areas[j] += face_area / 3.0
        areas[k] += face_area / 3.0

        # Accumulate off-diagonal cotangent weights
        # w_ij += 0.5 * cot_k  (angle opposite to edge ij is at vertex k)
        for (a, b, cot) in [(i, j, cot_k), (j, k, cot_i), (i, k, cot_j)]:
            w = 0.5 * cot
            # Off-diagonal: L[a,b] += -w, L[b,a] += -w
            row_list += [a, b]
            col_list += [b, a]
            val_list += [-w, -w]
            # Diagonal: L[a,a] += w, L[b,b] += w
            row_list += [a, b]
            col_list += [a, b]
            val_list += [w, w]

    row_idx = torch.tensor(row_list, dtype=torch.long)
    col_idx = torch.tensor(col_list, dtype=torch.long)
    values  = torch.tensor(val_list, dtype=torch.float32)
    area_t  = torch.tensor(areas, dtype=torch.float32).clamp(min=1e-8)

    # Coalesce duplicate indices by summing
    # Use a dense accumulation for correctness
    L_dense = torch.zeros(N, N, dtype=torch.float32)
    L_dense[row_idx, col_idx] += values
    row_s, col_s = L_dense.nonzero(as_tuple=True)
    val_s = L_dense[row_s, col_s]

    return row_s, col_s, val_s, area_t


def precompute_mesh(
    points: torch.Tensor,
    k: int = 20,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full preprocessing pipeline:
      point cloud → k-NN mesh → Cotangent Laplacian + Voronoi areas.

    This is called once per sample before training begins. Results are
    cached and reused across all training epochs (Section IV-A).

    Args:
        points : (N, 3) float tensor.
        k      : k-NN neighbours (default 20).
        device : Target device.

    Returns:
        L_row   : Sparse Laplacian row indices.
        L_col   : Sparse Laplacian col indices.
        L_val   : Sparse Laplacian values.
        areas   : (N,) Voronoi area per vertex.
    """
    vertices, edges, faces = build_knn_mesh(points, k=k)

    if len(faces) == 0:
        N = points.shape[0]
        return (torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.float32, device=device),
                torch.ones(N, dtype=torch.float32, device=device))

    L_row, L_col, L_val, areas = cotangent_laplacian(
        vertices, faces, n_vertices=len(vertices)
    )
    return (L_row.to(device), L_col.to(device),
            L_val.to(device), areas.to(device))


def sparse_laplacian_mm(
    L_row: torch.Tensor,
    L_col: torch.Tensor,
    L_val: torch.Tensor,
    N: int,
    X: torch.Tensor,
) -> torch.Tensor:
    """
    Sparse matrix-matrix product: L @ X.

    Args:
        L_row, L_col, L_val : Sparse COO representation of L (N×N).
        N                   : Number of vertices.
        X                   : (N, d) feature matrix.

    Returns:
        LX : (N, d) result of L @ X.
    """
    # Use torch.sparse for efficient multiplication
    indices = torch.stack([L_row, L_col], dim=0)  # (2, nnz)
    L_sparse = torch.sparse_coo_tensor(indices, L_val, (N, N),
                                        dtype=X.dtype, device=X.device)
    return torch.sparse.mm(L_sparse, X)
