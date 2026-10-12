# Low-Rank Symmetric-Score Polynomial Self-Attention

Official simulation implementation and reproducible experiments for the JMLR paper:
**"Metric Entropy and Generalization Upper Bounds for Low-Rank Symmetric-Score Polynomial Self-Attention"**  
Author: **Hideki Ishiyama** (Independent Researcher, Osaka, Japan)

---

## Overview

This repository contains the PyTorch implementation and numerical simulation suite for analyzing the capacity and generalization behavior of a softmax-free polynomial self-attention surrogate:

$$f_{S,V}(x) = (x^T S x)(V x), \quad x \in \mathbb{R}^d, \; S \in \mathcal{S}_r, \; V \in \mathbb{R}^{d \times d}$$

where $\mathcal{S}_r = \{S \in \text{Sym}_d(\mathbb{R}) : \text{rank}(S) \le r, \|S\|_F \le R_A\}$ is the bounded symmetric determinantal variety.

---

## Theoretical Highlights & Specifications

* **S-Reduction & Quadratic Symmetry**: By exploiting quadratic form invariance $x^T A x = x^T \text{sym}(A) x$, non-functional skew-symmetric components $\text{skew}(A)$ are eliminated, reducing the score parameter dimension to $d_S(r) = \frac{r(2d-r+1)}{2}$.
* **Capacity Exponent $D_r$**: The total rank-dependent parameter-space entropy exponent is:
  $$D_r = d_S(r) + d^2 = \frac{r(2d - r + 1)}{2} + d^2$$
  For $d=4$, $D_r \in \{20, 23, 25, 26\}$ for $r \in \{1, 2, 3, 4\}$, strictly tighter than generic matrix capacity $D_r^{\text{gen}} = r(2d-r) + d^2 \in \{23, 28, 31, 32\}$.
* **Generalization Upper Bound**: Dudley chaining on the centered scalar loss class yields an explicit $O\left(C_R L_\ell B^3 R_A R_V \sqrt{D_r / n}\right)$ uniform generalization error bound.
* **Two-Tier Capacity Framework & Local RLCT**: Near the rank-collapse locus $\det(S) = 0$, the local determinantal normal form $K_{\text{NF}}(x,y,z,w,v) = (xy - zw)^2 + \|v\|_2^2$ with $v \in \mathbb{R}^{d^2}$ exhibits an exact Real Log Canonical Threshold (RLCT):
  $$\lambda_{\text{NF}} = 0.5 + \frac{d^2}{2} = \frac{1 + d^2}{2} \quad (\lambda_{\text{NF}} = 8.5 \text{ for } d=4)$$
  governed by the smooth hypersurface locus $xy - zw = 0$ with $\lambda_{\text{score}} = 0.5$.
* **Rank-1 Functional Approximation Bias**: Parameter truncation error $\|S_\star - S_{\text{rank-1}}\|_F = \lambda_2 = 1/\sqrt{5} \approx 0.4472$, while functional population MSE approximation bias $R_{\text{approx}} \approx 0.051039$ for fixed $V = V_\star$.

---

## Repository Structure

```text
.
├── experiment_simulation.py  # PyTorch model, projected training, and multi-seed simulation suite
├── colab_simulation.py       # Google Colab multi-core parallel execution script
├── README.md                    # Documentation and specification guide
└── figures/                     # Generated publication-ready diagnostic plots (300 DPI)
    ├── fig1_rank_tradeoff.png   # Trade-off vs. Student Rank r (n = 150)
    ├── fig2_capacity_curve.png  # Capacity Exponent D_r Scaling vs. Unconstrained Dim
    ├── fig3_sample_curve.png    # Generalization Gap Decay vs. n (100 to 5000)
    ├── fig4_bound_comparison.png# S-Reduced Capacity vs. Generic Matrix Capacity
    └── fig5_convergence_steps.png # Optimization convergence dynamics
```

---

## Prerequisites & Installation

### Requirements
* **Python**: 3.10+ (tested on Python 3.12)
* **Dependencies**:
  * `torch >= 2.0.0`
  * `numpy >= 1.24.0`
  * `matplotlib >= 3.7.0`

---

## Simulation Setup & Parameters

* **Teacher Model**: True rank $r_\star = 2$ in $d=4$, normalized to $\|S_\star\|_F = R_A = 1.0$ and $\|V_\star\|_F = R_V = 1.0$.
* **Input Sampling**: Exact rejection sampling $X \sim \mathcal{N}(0, I_d)$ truncated to $\|X\|_2 \le B = 2.0$.
* **Noise Floor**: Label noise $\xi \sim \mathcal{N}(0, \sigma^2 I_d)$ with $\sigma = 0.01$. Theoretical irreducible noise floor $\frac{1}{2}d\sigma^2 = 0.000200$.
* **Optimization**: Pure SGD ($\eta = 0.01$), 10,000 gradient steps per run, batch size 32, with symmetric SVD rank projection and Frobenius norm clipping.
* **Evaluation**: 50 independent random seeds (Seeds 1–50) evaluated on an independent test set $N_{\text{test}} = 5,000$.

---

## Usage

### 1. Fast Dry-Run (Verification)
To run a fast multi-seed check across 5 seeds with 2,500 training steps:
```bash
python experiment_simulation.py --steps 2500 --seeds 5
```

### 2. Full Paper Simulation
To run the full 50-seed parallel simulation with 10,000 training steps per run:
```bash
python experiment_simulation.py --steps 10000 --seeds 50
```

---

## Key Output Figures

When executed, the simulation suite outputs five publication-ready figures (300 DPI):

1. **`fig1_rank_tradeoff.png` (Trade-off vs. Student Rank $r$)**: Illustrates empirical train risk, generalization gap, and test risk across student ranks $r \in \{1, 2, 3, 4\}$ ($n=150$), highlighting exact convergence to the noise floor at $r \ge r_\star = 2$.
2. **`fig2_capacity_curve.png` (Capacity Exponent $D_r$ Scaling)**: Plots $D_r$ vs. student rank $r$ for $d=4$, demonstrating reduction from unconstrained dimension $2d^2 = 32$.
3. **`fig3_sample_curve.png` (Generalization Gap Decay vs. $n$)**: Demonstrates $O(n^{-1/2})$ decay rate of empirical generalization gaps across sample sizes $n \in [100, 5000]$ with unclipped $\pm 1$ SD error bands.
4. **`fig4_bound_comparison.png` (Capacity Exponent Comparison)**: Bar chart comparing $S$-reduced capacity $D_r$ against generic matrix capacity $D_r^{\text{gen}}$.
5. **`fig5_convergence_steps.png` (Convergence Dynamics)**: Training loss convergence curves confirming stability across gradient steps.

---

## Citation & References

* **Main Paper**: Ishiyama, H. (2026). "Metric Entropy and Generalization Upper Bounds for Low-Rank Symmetric-Score Polynomial Self-Attention." *Journal of Machine Learning Research* (Submitted Oct 15, 2026).
* **Companion Study**: Ishiyama, H. (2026). "Rank-adaptive local empirical processes in low-rank attention." *Under Review*. Preprint available at [https://doi.org/10.5281/zenodo.23147548](https://doi.org/10.5281/zenodo.23147548).
* **Code Repository**: Anonymous reproducibility repository at [https://anonymous.4open.science/r/pega1-D170/](https://anonymous.4open.science/r/pega1-D170/).
