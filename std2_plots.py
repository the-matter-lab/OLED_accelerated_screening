#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
std2_plots.py

Generate publication-style plots for std2 results.

- Heatmaps (mean/std) of E_S1 and f_osc across (strength, absorption).
- Violin plots with permutation-based p-values and BH-FDR correction.
- Scatterplot E_S1 vs f_osc with color mapped from wavelength.

Usage:
    python std2_plots.py --csv std2.csv --output_dir results/std2 --n_permutations 5000
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ---------------- wavelength to rgb ----------------
def wavelength_to_rgb(wavelength, gamma=0.8):
    """
    Convert a wavelength in nm to an approximate RGB color.
    Returns realistic RGB color in [0, 1] for visible range (380–750 nm).
    Returns gray (0.5, 0.5, 0.5) for UV and NIR.
    """
    if wavelength < 380 or wavelength > 750:
        return (0.5, 0.5, 0.5)

    if 380 <= wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif 440 < wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif 490 < wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif 510 < wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif 580 < wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif 645 < wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    return (R, G, B)

# ---------------- latex setup ----------------
def try_enable_latex():
    try:
        matplotlib.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 11,
        })
        return True
    except Exception:
        matplotlib.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 11,
        })
        return False

# ---------------- column identification ----------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_and_identify(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    col_Es1 = find_col(df, ["E_S1_eV","E_S1","E_S1 (eV)","Es1_eV"])
    col_f   = find_col(df, ["f_osc_S1","f_S1","fosc_S1","f_osc","fosc"])
    col_str = find_col(df, ["strength","precond_strength","target_strength"])
    col_abs = find_col(df, ["absorption","precond_absorption","target_absorption"])

    for c in [col_Es1, col_f]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, col_Es1, col_f, col_str, col_abs

# ---------------- permutation stats ----------------
def f_statistic(groups):
    k = len(groups)
    n_i = np.array([len(g) for g in groups], dtype=float)
    n = n_i.sum()
    means = np.array([np.mean(g) if len(g)>0 else np.nan for g in groups])
    grand = np.average(means, weights=n_i)
    ssb = np.sum(n_i * (means - grand)**2)
    ssw = sum(np.sum((g - np.mean(g))**2) for g in groups if len(g) > 1)
    msb = ssb / (k - 1) if k > 1 else 0.0
    msw = ssw / (n - k) if (n - k) > 0 else np.nan
    return msb / msw if msw > 0 else np.inf

def permutation_anova(values, labels, n_perm=5000, rng=None):
    if rng is None: rng = np.random.default_rng(123)
    uniq = [u for u in pd.unique(labels) if pd.notna(u)]
    groups = [values[labels == u] for u in uniq]
    f_obs = f_statistic(groups)
    count = 0
    lbl = labels.copy()
    for _ in range(n_perm):
        rng.shuffle(lbl)
        groups_perm = [values[lbl == u] for u in uniq]
        f_perm = f_statistic(groups_perm)
        if f_perm >= f_obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(f_obs), float(p), uniq

def permutation_pairwise_means(x, y, n_perm=5000, rng=None):
    if rng is None: rng = np.random.default_rng(123)
    d_obs = np.mean(x) - np.mean(y)
    pooled = np.concatenate([x, y])
    n_x = len(x)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = np.mean(pooled[:n_x]) - np.mean(pooled[n_x:])
        if abs(d) >= abs(d_obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(d_obs), float(p)

def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n+1)
    q = pvals * n / ranks
    q_sorted = q[order]
    for i in range(n-2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i+1])
    q[order] = q_sorted
    return np.clip(q, 0, 1)

def significance_stars(p):
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"

# ---------------- plotting utils ----------------
def save_heatmap_pdf(tbl, out_path, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    im = ax.imshow(tbl.values, aspect="auto")
    ax.set_xticks(range(len(tbl.columns)))
    ax.set_xticklabels([str(c) for c in tbl.columns])
    ax.set_yticks(range(len(tbl.index)))
    ax.set_yticklabels([str(i) for i in tbl.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(tbl.shape[0]):
        for j in range(tbl.shape[1]):
            v = tbl.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)

def violin_plot(groups_dict, out_path, title, ylabel, n_perm=5000):
    labels = list(groups_dict.keys())
    data = [np.asarray(groups_dict[k], dtype=float) for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    meds = [np.median(d) if len(d)>0 else np.nan for d in data]
    ax.scatter(np.arange(1, len(labels) + 1), meds, zorder=3)

    vals = np.concatenate([d for d in data])
    grps = np.concatenate([np.full(len(d), labels[i]) for i, d in enumerate(data)])
    F, p_global, uniq = permutation_anova(vals, grps, n_perm=n_perm)
    ax.text(0.02, 0.98, f"global p={p_global:.2e}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9)

    if len(labels) >= 2:
        base = data[0]
        pw_rows = []
        pvals = []
        for j in range(1, len(labels)):
            d_obs, p = permutation_pairwise_means(base, data[j], n_perm=n_perm)
            pvals.append(p)
            pw_rows.append({"baseline": labels[0], "other": labels[j],
                            "mean_diff": d_obs, "p": p})
        if pvals:
            qvals = benjamini_hochberg(pvals)
            for r, q in zip(pw_rows, qvals):
                r["q"] = float(q)
                r["stars"] = significance_stars(q)
            ymax = np.nanmax(vals)
            y0 = ymax + 0.05*(np.nanmax(vals)-np.nanmin(vals)+1e-9)
            for j in range(1, len(labels)):
                ax.text(j+1, y0, pw_rows[j-1]["stars"], ha="center", va="bottom", fontsize=11)
            pd.DataFrame(pw_rows).to_csv(str(out_path).replace(".pdf","_pairwise.csv"), index=False)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)

# ---------------- scatter with color ----------------
def scatter_colored(df, col_Es1, col_f, out_path):
    e = df[col_Es1].dropna()
    f = df[col_f].dropna()
    df2 = df.dropna(subset=[col_Es1, col_f])
    wavelengths = 1239.841984 / df2[col_Es1].values  # eV → nm
    colors = [wavelength_to_rgb(w) for w in wavelengths]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.scatter(df2[col_Es1], df2[col_f], c=colors, s=10)
    ax.set_xlabel(r"$E_{\mathrm{S1}}$ (eV)")
    ax.set_ylabel(r"$f_{\mathrm{osc}}(\mathrm{S1})$")
    ax.set_title(r"$E_{\mathrm{S1}}$ vs $f_{\mathrm{osc}}(\mathrm{S1})$ (colored by wavelength)")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="std2.csv")
    parser.add_argument("--output_dir", type=str, default="results/std2")
    parser.add_argument("--n_permutations", type=int, default=5000)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    try_enable_latex()

    df, col_Es1, col_f, col_str, col_abs = load_and_identify(args.csv)

    # heatmaps
    for val_col, nice in [(col_Es1, r"$E_{\mathrm{S1}}$ (eV)"), (col_f, r"$f_{\mathrm{osc}}(\mathrm{S1})$")]:
        g = df[[col_str, col_abs, val_col]].dropna()
        tbl_mean = g.pivot_table(index=col_str, columns=col_abs, values=val_col, aggfunc="mean")
        tbl_std  = g.pivot_table(index=col_str, columns=col_abs, values=val_col, aggfunc="std")
        save_heatmap_pdf(tbl_mean, Path(args.output_dir)/f"{val_col}_mean.pdf",
                         f"Mean of {nice}", "absorption", "strength")
        save_heatmap_pdf(tbl_std, Path(args.output_dir)/f"{val_col}_std.pdf",
                         f"Std of {nice}", "absorption", "strength")
        tbl_mean.to_csv(Path(args.output_dir)/f"{val_col}_mean.csv")
        tbl_std.to_csv(Path(args.output_dir)/f"{val_col}_std.csv")

    # violins
    g = df[[col_abs, col_Es1]].dropna()
    groups = {lvl: sub[col_Es1].to_numpy() for lvl, sub in g.groupby(col_abs)}
    violin_plot(groups, Path(args.output_dir)/"violin_E_S1_by_absorption_all_strengths.pdf",
                r"$E_{\mathrm{S1}}$ by absorption (all strengths)", r"$E_{\mathrm{S1}}$ (eV)", n_perm=args.n_permutations)

    g = df[[col_str, col_f]].dropna()
    groups = {lvl: sub[col_f].to_numpy() for lvl, sub in g.groupby(col_str)}
    violin_plot(groups, Path(args.output_dir)/"violin_fosc_by_strength_all_absorptions.pdf",
                r"$f_{\mathrm{osc}}(\mathrm{S1})$ by strength (all absorptions)", r"$f_{\mathrm{osc}}(\mathrm{S1})$", n_perm=args.n_permutations)

    # scatter colored
    scatter_colored(df, col_Es1, col_f, Path(args.output_dir)/"scatter_E_S1_vs_fosc_colored.pdf")

if __name__ == "__main__":
    main()

