#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
std2_plots.py

Generate publication-style plots for std2 results.

Outputs (PDFs + CSVs) to --output_dir (default: results/std2):
1) Heatmaps (mean/std) of E_S1_eV and f_osc_S1 over (strength, absorption).
2) Violin plots with permutation p-values (BH-FDR corrected):
   a) E_S1_eV by absorption (aggregating all strengths).
   b) E_S1_eV by absorption, panels for strength = 0..4 (1×5 multi-panel + individual files).
   c) f_osc_S1 by strength (aggregating all absorptions).
   d) f_osc_S1 by strength, panels for absorption = 0..4 (1×5 multi-panel + individual files).
3) Scatter E_S1_eV vs f_osc_S1 colored by wavelength (E[eV]→λ[nm]→RGB).

Usage:
    python std2_plots.py --csv std2.csv --output_dir results/std2 --n_permutations 5000
"""

import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ---------------- wavelength -> rgb ----------------
def wavelength_to_rgb(wavelength, gamma=0.8):
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

# ---------------- LaTeX safe setup ----------------
def try_enable_latex():
    has_latex = shutil.which("latex") is not None
    if has_latex:
        matplotlib.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 11,
        })
    else:
        matplotlib.rcParams.update({
            "text.usetex": False,     # fallback (prevents the crash you saw)
            "font.family": "serif",
            "font.size": 11,
        })
    return has_latex

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
    grand = np.average(means, weights=n_i) if np.isfinite(means).all() else np.nan
    ssb = np.sum(n_i * (means - grand)**2)
    ssw = sum(np.sum((g - np.mean(g))**2) for g in groups if len(g) > 1)
    msb = ssb / (k - 1) if k > 1 else 0.0
    msw = ssw / (n - k) if (n - k) > 0 else np.nan
    return msb / msw if (msw is not None and msw > 0) else np.inf

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
    x = np.asarray(x); y = np.asarray(y)
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
    if len(data) == 0: return
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
    _, p_global, _ = permutation_anova(vals, grps, n_perm=n_perm)
    ax.text(0.02, 0.98, f"global p={p_global:.2e}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9)

    if len(labels) >= 2:
        base = data[0]
        pw_rows, pvals = [], []
        for j in range(1, len(labels)):
            d_obs, p = permutation_pairwise_means(base, data[j], n_perm=n_perm)
            pvals.append(p)
            pw_rows.append({"baseline": labels[0], "other": labels[j],
                            "mean_diff": d_obs, "p": p})
        if pvals:
            qvals = benjamini_hochberg(pvals)
            ymax = np.nanmax(vals); ymin = np.nanmin(vals)
            y0 = ymax + 0.05*(ymax - ymin + 1e-9)
            for j, (r, q) in enumerate(zip(pw_rows, qvals), start=1):
                r["q"] = float(q); r["stars"] = significance_stars(q)
                ax.text(j+1, y0, r["stars"], ha="center", va="bottom", fontsize=11)
            pd.DataFrame(pw_rows).to_csv(str(out_path).replace(".pdf","_pairwise.csv"), index=False)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)

def multi_panel_violins(groupby_level, value_col, fixed_col, fixed_levels, out_path, title, ylabel, df, n_perm=5000):
    """Create a 1x5 multi-panel violin figure; also save individual per-panel files."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), dpi=300, squeeze=False)
    axes = axes[0]
    for idx, lvl in enumerate(fixed_levels):
        sub = df[df[fixed_col] == lvl][[groupby_level, value_col]].dropna()
        if sub.empty:
            axes[idx].set_axis_off()
            continue
        labels = sorted(pd.unique(sub[groupby_level]))
        data = [sub.loc[sub[groupby_level]==g, value_col].to_numpy() for g in labels]
        vp = axes[idx].violinplot(data, showmeans=True, showmedians=True, showextrema=False)
        axes[idx].set_xticks(np.arange(1, len(labels)+1))
        axes[idx].set_xticklabels([str(l) for l in labels], rotation=0)
        axes[idx].set_title(f"{fixed_col}={lvl}")
        if idx == 0:
            axes[idx].set_ylabel(ylabel)
        # individual file for this panel
        groups = {l:d for l,d in zip(labels, data)}
        single_path = Path(str(out_path).replace(".pdf", f"_{fixed_col}_{lvl}.pdf"))
        violin_plot(groups, single_path, f"{title} ({fixed_col}={lvl})", ylabel, n_perm=n_perm)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, format="pdf")
    plt.close(fig)

# ---------------- scatter colored by wavelength ----------------
def scatter_colored(df, col_Es1, col_f, out_path):
    df2 = df.dropna(subset=[col_Es1, col_f]).copy()
    wavelengths = 1239.841984 / df2[col_Es1].values  # eV -> nm
    colors = [wavelength_to_rgb(w) for w in wavelengths]
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)
    ax.scatter(df2[col_Es1], df2[col_f], c=colors, s=10)
    ax.set_xlabel(r"$E_{\mathrm{S1}}$ (eV)")
    ax.set_ylabel(r"$f_{\mathrm{osc}}(\mathrm{S1})$")
    ax.set_title(r"$E_{\mathrm{S1}}$ vs $f_{\mathrm{osc}}(\mathrm{S1})$ (colored by $\lambda$)")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)

# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="std2.csv")
    p.add_argument("--output_dir", type=str, default="results/std2")
    p.add_argument("--n_permutations", type=int, default=5000)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try_enable_latex()  # safe fallback if LaTeX not installed

    df, col_Es1, col_f, col_str, col_abs = load_and_identify(args.csv)
    if not all([col_Es1, col_f, col_str, col_abs]):
        raise SystemExit("Required columns not found: need E_S1_eV, f_osc_S1, strength, absorption.")

    # Heatmaps
    for val_col, nice in [(col_Es1, r"$E_{\mathrm{S1}}$ (eV)"), (col_f, r"$f_{\mathrm{osc}}(\mathrm{S1})$")]:
        g = df[[col_str, col_abs, val_col]].dropna()
        tbl_mean = g.pivot_table(index=col_str, columns=col_abs, values=val_col, aggfunc="mean")
        tbl_std  = g.pivot_table(index=col_str, columns=col_abs, values=val_col, aggfunc="std")
        save_heatmap_pdf(tbl_mean, out_dir / f"{val_col}_mean.pdf",
                         f"Mean of {nice} over (strength, absorption)", "absorption", "strength")
        save_heatmap_pdf(tbl_std,  out_dir / f"{val_col}_std.pdf",
                         f"Std of {nice} over (strength, absorption)", "absorption", "strength")
        tbl_mean.to_csv(out_dir / f"{val_col}_mean.csv")
        tbl_std.to_csv(out_dir / f"{val_col}_std.csv")

    # Violins: aggregated
    g = df[[col_abs, col_Es1]].dropna()
    groups = {lvl: sub[col_Es1].to_numpy() for lvl, sub in g.groupby(col_abs)}
    violin_plot(groups, out_dir / "violin_E_S1_by_absorption_all_strengths.pdf",
                r"$E_{\mathrm{S1}}$ by absorption (all strengths)", r"$E_{\mathrm{S1}}$ (eV)",
                n_perm=args.n_permutations)

    g = df[[col_str, col_f]].dropna()
    groups = {lvl: sub[col_f].to_numpy() for lvl, sub in g.groupby(col_str)}
    violin_plot(groups, out_dir / "violin_fosc_by_strength_all_absorptions.pdf",
                r"$f_{\mathrm{osc}}(\mathrm{S1})$ by strength (all absorptions)", r"$f_{\mathrm{osc}}(\mathrm{S1})$",
                n_perm=args.n_permutations)

    # Multi-panel violins: E_S1 by absorption for strength 0..4
    str_levels = sorted(pd.unique(df[col_str]))
    need_levels_s = [lvl for lvl in range(5) if lvl in set(str_levels)]  # strength 0..4 if present
    if need_levels_s:
        multi_panel_violins(col_abs, col_Es1, col_str, need_levels_s,
                            out_dir / "violin_panels_E_S1_by_absorption_per_strength.pdf",
                            r"$E_{\mathrm{S1}}$ by absorption (per strength)", r"$E_{\mathrm{S1}}$ (eV)",
                            df, n_perm=args.n_permutations)

    # Multi-panel violins: f_osc by strength for absorption 0..4
    abs_levels = sorted(pd.unique(df[col_abs]))
    need_levels_a = [lvl for lvl in range(5) if lvl in set(abs_levels)]  # absorption 0..4 if present
    if need_levels_a:
        multi_panel_violins(col_str, col_f, col_abs, need_levels_a,
                            out_dir / "violin_panels_fosc_by_strength_per_absorption.pdf",
                            r"$f_{\mathrm{osc}}(\mathrm{S1})$ by strength (per absorption)", r"$f_{\mathrm{osc}}(\mathrm{S1})$",
                            df, n_perm=args.n_permutations)

    # Scatter colored by wavelength
    scatter_colored(df, col_Es1, col_f, out_dir / "scatter_E_S1_vs_fosc_colored.pdf")

    # README
    with open(out_dir / "README_plots.txt", "w") as f:
        f.write(f"Figures written to: {out_dir}\n")
        f.write("Heatmaps: *_mean.pdf, *_std.pdf for E_S1_eV and f_osc_S1\n")
        f.write("Violins: aggregated and 1x5 multi-panels (+ per-panel PDFs). Pairwise CSVs next to PDFs.\n")
        f.write("Scatter: scatter_E_S1_vs_fosc_colored.pdf (points colored by wavelength).\n")

if __name__ == "__main__":
    main()

