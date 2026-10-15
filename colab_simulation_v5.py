"""
JMLR Paper Simulation Script (v4 - Google Colab Ready Version)
Paper: Metric Entropy and Generalization Upper Bounds for Low-Rank Symmetric-Score Polynomial Self-Attention
Author: Hideki Ishiyama

Features:
- Stage 1 Bugfixes: True Rejection Sampling (||x||_2 <= B) & Unclipped Gap Error Bands
- Stage 2 Validated Settings: Configurable Seeds (default 20, up to 50) & Steps (default 5000)
- Optimized for Google Colab CPU / GPU execution with tqdm progress tracking
- Automatic high-res export (Figures 1-4) & formatted numerical text summary
"""

import os
import argparse
import time
import concurrent.futures
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Try importing tqdm for friendly Colab progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

CONFIG = {
    'd': 4,
    'r_teacher': 2,
    'R_A': 1.0,
    'R_V': 1.0,
    'B': 2.0,
    'noise_std': 0.01,
    'lr': 0.01,
    'batch_size': 32,
    'N_test': 5000,
    'dpi': 300
}

def paper_loss(z, y):
    """Paper loss function: l(z, y) = 0.5 * ||z - y||_2^2 averaged over batch."""
    return 0.5 * torch.sum((z - y) ** 2, dim=-1).mean()

class PolynomialSelfAttention(nn.Module):
    def __init__(self, d=4, rank=2, R_A=1.0, R_V=1.0):
        super().__init__()
        self.d = d
        self.rank = rank
        self.R_A = R_A
        self.R_V = R_V
        S_init = torch.randn(d, d)
        S_sym = 0.5 * (S_init + S_init.T)
        self.S = nn.Parameter(S_sym * 0.1)
        self.V = nn.Parameter(torch.randn(d, d) * 0.1)

    def forward(self, x):
        S_sym = 0.5 * (self.S + self.S.T)
        quad = torch.bmm(x.unsqueeze(1), torch.matmul(x, S_sym).unsqueeze(2)).squeeze(-1)
        Vx = torch.matmul(x, self.V.T)
        return quad * Vx

    def project_constraints(self):
        with torch.no_grad():
            S_sym = 0.5 * (self.S + self.S.T)
            L, U = torch.linalg.eigh(S_sym)
            abs_L, idx = torch.sort(torch.abs(L), descending=True)
            top_idx = idx[:self.rank]
            L_trunc = torch.zeros_like(L)
            L_trunc[top_idx] = L[top_idx]
            S_proj = U @ torch.diag(L_trunc) @ U.T
            norm_S = torch.norm(S_proj, 'fro')
            if norm_S > self.R_A:
                S_proj = S_proj * (self.R_A / norm_S)
            norm_V = torch.norm(self.V, 'fro')
            if norm_V > self.R_V:
                self.V.copy_(self.V * (self.R_V / norm_V))
            self.S.copy_(S_proj)

def create_teacher_model(seed):
    """
    Teacher score S_star is rank 2 with initial unnormalized singular values [1.0, 0.5].
    Normalized to Frobenius norm ||S_star||_F = R_A = 1.0.
    """
    torch.manual_seed(seed * 10000)
    np.random.seed(seed * 10000)
    d = CONFIG['d']
    
    S_init = torch.randn(d, d)
    S_sym = 0.5 * (S_init + S_init.T)
    U_t, L_t, _ = torch.linalg.svd(S_sym)
    L_t_trunc = torch.zeros_like(L_t)
    L_t_trunc[:CONFIG['r_teacher']] = torch.tensor([1.0, 0.5])
    S_teacher = U_t @ torch.diag(L_t_trunc) @ U_t.T
    
    norm_before = torch.norm(S_teacher, 'fro').item()
    S_teacher = S_teacher / norm_before * CONFIG['R_A']
    
    V_teacher = torch.randn(d, d)
    V_teacher = V_teacher / torch.norm(V_teacher, 'fro') * CONFIG['R_V']
    
    return S_teacher, V_teacher

def generate_dataset(S_teacher, V_teacher, n_samples, seed_offset):
    """
    Generates synthetic dataset using Rejection Sampling.
    Conditioned on ||x||_2 <= B (B=2.0).
    """
    torch.manual_seed(seed_offset)
    np.random.seed(seed_offset)
    d = CONFIG['d']
    B = CONFIG['B']
    
    samples = []
    collected = 0
    while collected < n_samples:
        batch_size_gen = max((n_samples - collected) * 2, 1000)
        x_cand = torch.randn(batch_size_gen, d)
        norms = torch.norm(x_cand, p=2, dim=-1)
        valid_x = x_cand[norms <= B]
        if valid_x.shape[0] > 0:
            samples.append(valid_x)
            collected += valid_x.shape[0]
            
    x = torch.cat(samples, dim=0)[:n_samples]
    
    with torch.no_grad():
        quad = torch.bmm(x.unsqueeze(1), torch.matmul(x, S_teacher).unsqueeze(2)).squeeze(-1)
        Vx = torch.matmul(x, V_teacher.T)
        y_clean = quad * Vx
        noise = torch.randn_like(y_clean) * CONFIG['noise_std']
        y = y_clean + noise
    return x, y

def train_and_evaluate(r_student, x_train, y_train, x_test, y_test, init_seed, num_steps=5000):
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    
    model = PolynomialSelfAttention(d=CONFIG['d'], rank=r_student, R_A=CONFIG['R_A'], R_V=CONFIG['R_V'])
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'])
    
    batch_size = min(CONFIG['batch_size'], len(x_train))
    n_train = len(x_train)
    
    batch_indices = torch.randint(0, n_train, (num_steps, batch_size))
    
    for step in range(num_steps):
        idx = batch_indices[step]
        x_b, y_b = x_train[idx], y_train[idx]
        
        optimizer.zero_grad()
        y_pred = model(x_b)
        loss = paper_loss(y_pred, y_b)
        loss.backward()
        optimizer.step()
        model.project_constraints()
        
    with torch.no_grad():
        final_train_loss = paper_loss(model(x_train), y_train).item()
        final_test_loss = paper_loss(model(x_test), y_test).item()
        
    return final_train_loss, final_test_loss

def run_single_seed(seed, sample_sizes, ranks, num_steps=5000):
    S_teacher, V_teacher = create_teacher_model(seed)
    x_test, y_test = generate_dataset(S_teacher, V_teacher, CONFIG['N_test'], seed * 10000 + 5000)
    
    seed_fig1 = {r: {} for r in ranks}
    seed_fig3 = {r: {n: 0.0 for n in sample_sizes} for r in ranks}
    
    # Fig 1: n = 150
    x_tr_150, y_tr_150 = generate_dataset(S_teacher, V_teacher, 150, seed * 10000 + 1000 + 150)
    for r in ranks:
        init_seed = seed * 10000 + 2000 + r
        tr_l, te_l = train_and_evaluate(r, x_tr_150, y_tr_150, x_test, y_test, init_seed, num_steps=num_steps)
        seed_fig1[r] = {'train': tr_l, 'test': te_l}
        
    # Fig 3: sample size sweep
    for n in sample_sizes:
        x_tr, y_tr = generate_dataset(S_teacher, V_teacher, n, seed * 10000 + 1000 + n)
        for r in ranks:
            init_seed = seed * 10000 + 2000 + r
            tr_l, te_l = train_and_evaluate(r, x_tr, y_tr, x_test, y_test, init_seed, num_steps=num_steps)
            seed_fig3[r][n] = te_l - tr_l
            
    return seed, seed_fig1, seed_fig3

def run_simulation(num_seeds=20, num_steps=5000, num_workers=None):
    sample_sizes = [100, 200, 500, 1000, 2000, 5000]
    ranks = [1, 2, 3, 4]
    
    results_fig1 = {r: {'train': [], 'test': []} for r in ranks}
    results_fig3 = {r: {n: [] for n in sample_sizes} for r in ranks}
    
    print(f"=========================================================")
    print(f"Starting JMLR Multi-Seed Simulation")
    print(f"  Seeds: {num_seeds} | Steps per run: {num_steps}")
    print(f"=========================================================")
    start_time = time.time()
    
    workers = num_workers or min(os.cpu_count() or 4, num_seeds)
    print(f"Parallelizing across {workers} CPU workers...")
    
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_seed, seed, sample_sizes, ranks, num_steps) for seed in range(1, num_seeds + 1)]
        
        pbar = tqdm(total=num_seeds, desc="Simulating Seeds") if HAS_TQDM else None
        
        for future in concurrent.futures.as_completed(futures):
            seed, s_fig1, s_fig3 = future.result()
            for r in ranks:
                results_fig1[r]['train'].append(s_fig1[r]['train'])
                results_fig1[r]['test'].append(s_fig1[r]['test'])
                for n in sample_sizes:
                    results_fig3[r][n].append(s_fig3[r][n])
            completed += 1
            if pbar:
                pbar.update(1)
            else:
                print(f"Progress: {completed}/{num_seeds} seeds completed ({(time.time()-start_time):.1f}s)")
        if pbar:
            pbar.close()
                    
    elapsed = time.time() - start_time
    print(f"\nSimulation finished in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
    return results_fig1, results_fig3, sample_sizes, ranks

def export_figures_and_summary(results_fig1, results_fig3, sample_sizes, ranks, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})
    
    d = CONFIG['d']
    r_arr = np.array(ranks)
    d_S = r_arr * (2*d - r_arr + 1) / 2
    D_r = d_S + d**2
    
    # Figure 1: Empirical Trade-off
    mean_train_risks = np.array([np.mean(results_fig1[r]['train']) for r in ranks])
    mean_test_risks = np.array([np.mean(results_fig1[r]['test']) for r in ranks])
    std_test_risks = np.array([np.std(results_fig1[r]['test']) for r in ranks])
    emp_gaps = mean_test_risks - mean_train_risks
    
    plt.figure(figsize=(6, 4.5), dpi=CONFIG['dpi'])
    plt.plot(r_arr, mean_train_risks, 'o--', label='Empirical Train Risk', color='blue', linewidth=1.8)
    plt.plot(r_arr, emp_gaps, 's--', label=r'Empirical Gen Gap $(R_{\mathrm{test}} - R_{\mathrm{train}})$', color='orange', linewidth=1.8)
    plt.errorbar(r_arr, mean_test_risks, yerr=std_test_risks, fmt='d-', label='Empirical Test Risk', color='green', linewidth=2.0, capsize=4)
    plt.axvline(x=CONFIG['r_teacher'], color='red', linestyle=':', label=r'Teacher Rank $r_star = 2$')
    
    plt.xlabel('Student Rank $r$', fontsize=11)
    plt.ylabel('Paper Loss Magnitude', fontsize=11)
    plt.title('Trade-off vs. Student Rank $r$', fontsize=12, fontweight='bold')
    plt.xticks(r_arr)
    plt.legend(fontsize=9, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, 'fig1_rank_tradeoff.png')
    plt.savefig(fig1_path, dpi=CONFIG['dpi'])
    plt.close()
    print(f"Saved: {fig1_path}")
    
    # Figure 2: Capacity Exponent D_r Scaling
    plt.figure(figsize=(6, 4.5), dpi=CONFIG['dpi'])
    plt.plot(r_arr, D_r, 's-', color='purple', linewidth=2.0, label=r'$S$-Reduced Exponent $D_r$')
    plt.axhline(y=2*d**2, color='gray', linestyle='--', label=r'Unconstrained Dim $2d^2 = 32$', linewidth=1.5)
    for xi, yi in zip(r_arr, D_r):
        plt.annotate(f'{yi:.1f}', (xi, yi), textcoords="offset points", xytext=(0,8), ha='center')
        
    plt.xlabel('Student Rank $r$', fontsize=11)
    plt.ylabel('Capacity Exponent', fontsize=11)
    plt.title('Capacity Exponent $D_r$ Scaling', fontsize=12, fontweight='bold')
    plt.xticks(r_arr)
    plt.ylim(15, 35)
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'fig2_capacity_curve.png')
    plt.savefig(fig2_path, dpi=CONFIG['dpi'])
    plt.close()
    print(f"Saved: {fig2_path}")
    
    # Figure 3: Generalization Gap Decay vs n (2-Panel: Linear & Log-Log Scale)
    n_array = np.array(sample_sizes)
    colors = ['blue', 'green', 'orange', 'purple']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=CONFIG['dpi'])
    
    # Panel (a): Linear Scale (Mean ± SD)
    for idx, r in enumerate(ranks):
        mean_gaps = np.array([np.mean(results_fig3[r][n]) for n in sample_sizes])
        std_gaps = np.array([np.std(results_fig3[r][n]) for n in sample_sizes])
        D_val = D_r[idx]
        ax1.plot(n_array, mean_gaps, 'o-', label=f'Rank $r={r}$ ($D_r={int(D_val)}$)', color=colors[idx], linewidth=1.8)
        ax1.fill_between(n_array, mean_gaps - std_gaps, mean_gaps + std_gaps, color=colors[idx], alpha=0.15)
        
    ax1.set_xlabel('Sample Size $n$', fontsize=10)
    ax1.set_ylabel('Empirical Generalization Gap', fontsize=10)
    ax1.set_title('(a) Linear Scale (Mean \pm SD)', fontsize=11, fontweight='bold')
    ax1.grid(True, which="both", linestyle='--', alpha=0.6)
    ax1.legend(fontsize=8, loc='upper right')
    
    # Panel (b): Log-Log Scale (|R_test - R_train| vs n)
    for idx, r in enumerate(ranks):
        mean_gaps = np.array([np.mean(results_fig3[r][n]) for n in sample_sizes])
        abs_gaps = np.maximum(np.abs(mean_gaps), 1e-6)
        D_val = D_r[idx]
        ax2.plot(n_array, abs_gaps, 'o-', label=f'Rank $r={r}$ ($D_r={int(D_val)}$)', color=colors[idx], linewidth=1.8)
        
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sample Size $n$ (log scale)', fontsize=10)
    ax2.set_ylabel('$|R_{\\mathrm{test}} - R_{\\mathrm{train}}|$ (log scale)', fontsize=10)
    ax2.set_title('(b) Log-Log Scale', fontsize=11, fontweight='bold')
    ax2.grid(True, which="both", linestyle='--', alpha=0.6)
    ax2.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, 'fig3_sample_curve.png')
    plt.savefig(fig3_path, dpi=CONFIG['dpi'])
    plt.close()
    print(f"Saved: {fig3_path}")
    
    # Figure 4: Capacity Exponent Comparison
    labels = ['r=1', 'r=2', 'r=3', 'r=4']
    s_reduced = [20, 23, 25, 26]
    generic_matrix = [23, 28, 31, 32]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(6, 4.5), dpi=CONFIG['dpi'])
    plt.bar(x - width/2, s_reduced, width, label=r'$S$-Reduced Capacity $D_r$', color='navy')
    plt.bar(x + width/2, generic_matrix, width, label='Generic Matrix Capacity', color='crimson')
    plt.title('Capacity Exponent Comparison ($d=4$)', fontsize=12, fontweight='bold')
    plt.xlabel('Rank $r$', fontsize=11)
    plt.ylabel('Capacity Exponent', fontsize=11)
    plt.xticks(x, labels)
    plt.ylim(0, 36)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, 'fig4_bound_comparison.png')
    plt.savefig(fig4_path, dpi=CONFIG['dpi'])
    plt.close()
    print(f"Saved: {fig4_path}")

    # Generate Summary Text Report
    summary_txt = f"""
================================================================================
JMLR SIMULATION NUMERICAL RESULTS SUMMARY (COPY-PASTE TO CHAT)
================================================================================
Theoretical Noise Floor: 0.000200 (0.5 * d * sigma^2)

Figure 1 Data (Sample size n = 150):
Rank r=1: Train={mean_train_risks[0]:.6f}, Test={mean_test_risks[0]:.6f} ± {std_test_risks[0]:.6f}, Gap={emp_gaps[0]:.6f}
Rank r=2: Train={mean_train_risks[1]:.6f}, Test={mean_test_risks[1]:.6f} ± {std_test_risks[1]:.6f}, Gap={emp_gaps[1]:.6f}
Rank r=3: Train={mean_train_risks[2]:.6f}, Test={mean_test_risks[2]:.6f} ± {std_test_risks[2]:.6f}, Gap={emp_gaps[2]:.6f}
Rank r=4: Train={mean_train_risks[3]:.6f}, Test={mean_test_risks[3]:.6f} ± {std_test_risks[3]:.6f}, Gap={emp_gaps[3]:.6f}

Figure 3 Generalization Gap Decay vs n (Mean ± SD):
"""
    for r in ranks:
        summary_txt += f"\n-- Rank r={r} --\n"
        for n in sample_sizes:
            g_mean = np.mean(results_fig3[r][n])
            g_std = np.std(results_fig3[r][n])
            summary_txt += f"  n={n:4d}: Gap = {g_mean:.6f} ± {g_std:.6f}\n"

    summary_path = os.path.join(output_dir, "simulation_results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_txt)
    print(f"Saved summary: {summary_path}")
    print("\n" + summary_txt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="JMLR Simulation (Colab Executable)")
    parser.add_argument('--seeds', type=int, default=20, help='Number of random seeds (e.g., 20 or 50)')
    parser.add_argument('--steps', type=int, default=5000, help='Number of training steps per run (e.g., 5000 or 10000)')
    parser.add_argument('--workers', type=int, default=None, help='Number of CPU workers (default: auto)')
    parser.add_argument('--output_dir', type=str, default='./jmlr_output', help='Output directory')
    args, _ = parser.parse_known_args()
    
    res1, res3, n_list, r_list = run_simulation(num_seeds=args.seeds, num_steps=args.steps, num_workers=args.workers)
    export_figures_and_summary(res1, res3, n_list, r_list, output_dir=args.output_dir)
