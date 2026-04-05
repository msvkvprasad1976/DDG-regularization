# DDG Regularization — Code Repository

Companion code for the paper:

> **Discrete Differential Geometry as a Regularizer for Deep Learning on Manifold-Valued Data**
> Dr. Bomma Rama Krishna
> Department of Artificial Intelligence and Machine Learning
> Swarnandhra College of Engineering and Technology (Autonomous)

---

## Project Structure

```
ddg_project/
├── ddg/
│   ├── __init__.py           # Package exports
│   ├── mesh_construction.py  # k-NN triangulation + Cotangent Laplacian
│   ├── regularizer.py        # Dirichlet energy, Willmore energy, DDGRegularizer
│   ├── models.py             # DDGPointNet2 (ModelNet40) + DDGGIN (QM9)
│   └── trainer.py            # Training engine (Algorithm 1 from paper)
├── train_modelnet40.py       # ModelNet40 training script (Table I)
├── train_qm9.py              # QM9 training script (Table I)
├── ablation_study.py         # Tables II and III reproduction
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse
pip install -r requirements.txt
```

---

## Quick Demo (no dataset required)

Verify DDG energy computation on a synthetic sphere:

```bash
python ablation_study.py --study demo
```

Expected output:
```
DDG Energy Demo — Synthetic Sphere Point Cloud
  Dirichlet E_D  : 0.xxxx
  Willmore  E_W  : 0.xxxx
  E_D (random X) : 0.xxxx   > E_D (smooth X)   ✓
```

---

## Reproduce Table I — ModelNet40

```bash
# DDG Regularization (ours) — λ1=0.05, λ2=0.01
python train_modelnet40.py --use_ddg --lambda1 0.05 --lambda2 0.01

# Baselines
python train_modelnet40.py --baseline none      # No regularization
python train_modelnet40.py --baseline l2        # L2 weight decay
python train_modelnet40.py --baseline dropout   # Dropout only
```

Data is auto-downloaded to `./data/ModelNet40` on first run.

---

## Reproduce Table I — QM9

```bash
# DDG Regularization (ours)
python train_qm9.py --use_ddg --lambda1 0.05 --lambda2 0.01

# Baselines
python train_qm9.py --baseline none
python train_qm9.py --baseline l2
```

---

## Reproduce Table II — Ablation Study

```bash
python ablation_study.py --task modelnet40 --study ablation --n_runs 5
```

Sweeps (λ1, λ2) ∈ {0, 0.05} × {0, 0.01} with 5 seeds each.

---

## Reproduce Table III — Hyperparameter Sensitivity

```bash
python ablation_study.py --task modelnet40 --study sensitivity --n_runs 5
```

Sweeps λ1 ∈ {0.001, 0.010, 0.050, 0.100, 0.500} with λ2=0.01 fixed.

---

## Core API

```python
import torch
from ddg import precompute_mesh, DDGRegularizer, DDGPointNet2

# 1. Build mesh and Laplacian from point cloud (once, before training)
points = torch.randn(1024, 3)
L_row, L_col, L_val, areas = precompute_mesh(points, k=20)

# 2. Initialise model and regularizer
model = DDGPointNet2(num_classes=40)
reg   = DDGRegularizer(lambda1=0.05, lambda2=0.01)

# 3. Training step (Algorithm 1 from paper)
pred = model(batch)
task_loss = criterion(pred, batch.y)
X = model.get_latent()                            # X^(L)
total_loss, e_d, e_w = reg(
    X, task_loss, L_row, L_col, L_val, areas, N=X.shape[0]
)
total_loss.backward()
optimizer.step()
```

---

## Paper — Section-to-Code Mapping

| Paper Section          | Code File                        |
|------------------------|----------------------------------|
| Sec. III — Math        | `ddg/mesh_construction.py`       |
| Eq. (1)–(2)            | `cotangent_laplacian()`          |
| Sec. IV-A — Mesh       | `build_knn_mesh()`               |
| Eq. (3)–(5)            | `ddg/regularizer.py`             |
| Algorithm 1            | `ddg/trainer.py:train_epoch()`   |
| Sec. V — Setup         | `train_modelnet40.py`            |
| Sec. VI — Results      | Trainer + eval outputs           |
| Sec. VII — Ablation    | `ablation_study.py`              |
| Sec. VIII — Sensitivity| `ablation_study.py`              |
| Eq. (6) — Stability    | `ddg/regularizer.py:StabilityScore` |

---

## Citation

```bibtex
@inproceedings{ramakrishna2025ddg,
  title     = {Discrete Differential Geometry as a Regularizer for Deep
               Learning on Manifold-Valued Data},
  author    = {Rama Krishna, Bomma},
  booktitle = {Proceedings of the IEEE Conference},
  year      = {2025},
  institution = {Swarnandhra College of Engineering and Technology}
}
```
