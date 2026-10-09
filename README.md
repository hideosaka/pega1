# Low-Rank Symmetric-Score Polynomial Self-Attention

Official simulation implementation for the JMLR paper:
**"Metric Entropy and Generalization Upper Bounds for Low-Rank Symmetric-Score Polynomial Self-Attention"**  
*Author:* Hideki Ishiyama (Independent Researcher, Osaka, Japan)

---

# Overview

This repository contains the PyTorch implementation and numerical simulation suite for analyzing the capacity and generalization behavior of a **softmax-free polynomial self-attention surrogate**:

$$f_{S,V}(x) = (x^T S x)(V x), \quad x \in \mathbb{R}^d, \; S \in \mathcal{S}_r, \; V \in \mathbb{R}^{d 	imes d}$$

where $\mathcal{S}_r = \{S \in 	ext{Sym}_d(\mathbb{R}) : 	ext{rank}(S) \le r, \|S\|_F \le R_A\}$ is the bounded symmetric determinantal variety.

### Theoretical Highlights & $S$-Reduction
- **$S$-Reduction:** By exploiting quadratic form symmetry $x^T A x = x^T 	ext{sym}(A) x$, non-functional skew-symmetric components are eliminated.
- **Capacity Exponent $D_r$:** The total parameter-space entropy exponent for rank-$r$ symmetric attention in $\mathbb{R}^d$ is:
  $$D_r = d_S(r) + d^2 = \frac{r(2d - r + 1)}{2} + d^2$$
  For $d=4$, $D_r$ scales from $20$ ($r=1$) to $26$ ($r=4$), strictly tighter than generic matrix capacity bounds ($r(2d-r) + d^2$).
- **Generalization Upper Bound:** Dudley chaining yields an $O(\sqrt{D_r / n})$ uniform generalization error bound.

---

## Repository Structure

```text
.
├── experiment_simulation.py  # PyTorch model, projected training, and visualization script
├── README.md                 # Documentation
└── figures/                  # Generated diagnostic plots (Output)
    ├── fig1_rank_tradeoff.png
    ├── fig2_capacity_curve.png
    ├── fig3_sample_curve.png
    └── fig4_bound_comparison.png
```

---

## Prerequisites & Installation

### Requirements
- **Python:** 3.10+ (tested on Python 3.12)
- **Dependencies:**
  - `torch >= 2.0.0`
  - `numpy >= 1.24.0`
  - `matplotlib >= 3.7.0`

### Quick Setup
Clone the repository and install required packages:

```bash
git clone https://github.com/your-username/polynomial-attention-jmlr.git
cd polynomial-attention-jmlr
pip install torch numpy matplotlib
```

---

## Usage

### 1. Fast Dry-Run (Figure Generation)
To generate all paper figures using preset metrics:

```bash
python experiment_simulation.py --quick
```

### 2. Full Experiment Run
To train student models ($r \in \{1, 2, 3, 4\}$) against teacher rank $r_\star = 2$ across 50 random seeds:

```bash
python experiment_simulation.py
```

---

## Key Output Figures

When executed, the simulation outputs four publication-ready figures (`300 DPI`):

1. **`fig1_rank_tradeoff.png` (Trade-off vs. Student Rank $r$):**
   Illustrates the bias-variance trade-off between approximation error (vanishes at $r \ge r_\star=2$) and estimation error $O(\sqrt{D_r/n})$.
2. **`fig2_capacity_curve.png` (Capacity Exponent $D_r$ Scaling):**
   Plots $D_r$ vs. student rank $r$ for $d=4$, showing reduction from unconstrained dimension $2d^2 = 32$.
3. **`fig3_sample_curve.png` (Generalization Gap Decay vs. $n$):**
   Demonstrates the $O(n^{-1/2})$ decay rate of empirical generalization gaps across sample sizes $n \in [100, 5000]$.
4. **`fig4_bound_comparison.png` (Capacity Exponent Comparison):**
   Bar chart comparing $S$-reduced capacity $D_r$ (Navy) against generic matrix capacity (Crimson).

---

## Code Architecture

The core PyTorch module `PolynomialSelfAttention` implements:
- **Symmetric Forward Pass:** Enforces quadratic score evaluation $x^T 	ext{sym}(S) x$.
- **Projected Low-Rank SGD:**
  1. Top-$r$ Eigenvalue Truncation via `torch.linalg.eigh(S_sym)`.
  2. Frobenius Norm Clipping to $\|S\|_F \le R_A$ and $\|V\|_F \le R_V$.

```python
from experiment_simulation import PolynomialSelfAttention

# Initialize model for d=4, student rank r=2
model = PolynomialSelfAttention(d=4, rank=2, R_A=1.0, R_V=1.0)

# Apply projected constraints post gradient step
model.project_constraints()
