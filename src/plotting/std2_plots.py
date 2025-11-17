#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
std2_plots.py

Create publication-style plots from an std2 summary CSV.

What it makes (PDFs + CSVs) under --output_dir (default: results/std2/plots):
1) Heatmaps (mean, std) of E_S1_eV and f_osc_S1 over (strength, absorption).
2) Violin plots (with seaborn) + permutation p-values (BH-FDR corrected):
   a) E_S1_eV by absorption (aggregating all strengths)
      - labels clarify: preconditioned absorption (token) vs std2 E_S1 (result)
   b) E_S1_eV by absorption within each strength = 0..4 (1×5 panel + per-panel PDFs)
   c) f_osc_S1 by strength (aggregating all absorptions)
      - labels clarify: preconditioned strength (token) vs std2 f_osc (result)
   d) f_osc_S1 by strength within each absorption = 0..4 (1×5 panel + per-panel PDFs)
3) Scatterplots of E_S1_eV vs f_osc_S1 colored by wavelength (E[eV] → λ[nm] → RGB):
   - full range
   - zoomed: E_S1 in [1, 4] eV

Usage:
    python std2_plots.py --csv results/std2/std2.csv --output_dir results/std2/plots --n_permutations 5000
"""

import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- wavelength -> RGB ----------------
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

# ---------------- LaTeX-safe setup ----------------
def setup_matplotlib():
    has_latex = shutil.which("latex") is not None
    if has_latex:
        matplotlib.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 11,
        })
    else:
        matplotlib.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 11,
        })
    sns.set_context("talk")
    sns.set_style("whitegrid")
    return has_latex

# ---------------- column helpers ----------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_std2(csv_path):
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
    if n_i.sum() <= k:
        return np.inf
    means = np.array([np.mean(g) if len(g)>0 else np.nan for g in groups])
    grand = np.average(means, weights=n_i) if np.isfinite(means).all() else np.nan
    ssb = np.sum(n_i * (means - grand)**2)
    ssw = sum(np.sum((g - np.mean(g))**2) for g in groups if len(g) > 1)
    msb = ssb / (k - 1) if k > 1 else 0.0
    msw = ssw / (n_i.sum() - k) if (n_i.sum() - k) > 0 else np.nan
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

# ---------------- plotting helpers ----------------
def save_heatmap(tbl, out_pdf, title, xlabel, ylabel, fmt=".2f"):
    plt.figure(figsize=(9, 6), dpi=300)
    ax = plt.gca()
    sns.heatmap(tbl, annot=True, fmt=fmt, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()

def violin_with_stats(df, group_col, value_col, out_pdf, title, xlab, ylab, n_perm=5000, baseline=None):
    # seaborn violin
    plt.figure(figsize=(9, 5.5), dpi=300)
    ax = sns.violinplot(data=df, x=group_col, y=value_col, inner="quartile", cut=0)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # stats
    clean = df[[group_col, value_col]].dropna()
    labels = clean[group_col].to_numpy()
    vals = clean[value_col].to_numpy()
    _, p_global, _ = permutation_anova(vals, labels, n_perm=n_perm)
    ax.text(0.02, 0.98, f"global p={p_global:.2e}", transform=ax.transAxes,
            ha="left", va="top", fontsize=10)

    if baseline is None:
        try:
            baseline = sorted(clean[group_col].unique())[0]
        except Exception:
            baseline = None

    if baseline is not None:
        groups = []
        keys = []
        for k, sub in clean.groupby(group_col):
            keys.append(k)
            groups.append(sub[value_col].to_numpy())
        # map baseline index
        if baseline in keys:
            b_idx = keys.index(baseline)
            base = groups[b_idx]
            pvals = []
            rows = []
            for j, (k, g) in enumerate(zip(keys, groups)):
                if j == b_idx: 
                    continue
                d_obs, p = permutation_pairwise_means(base, g, n_perm=n_perm)
                rows.append({"baseline": baseline, "other": k, "mean_diff": d_obs, "p": p})
                pvals.append(p)
            if pvals:
                qvals = benjamini_hochberg(pvals)
                for r, q in zip(rows, qvals):
                    r["q"] = float(q)
                    r["stars"] = significance_stars(q)
                # annotate stars above violins (y-limit aware)
                ymax = np.nanmax(vals); ymin = np.nanmin(vals)
                y0 = ymax + 0.05*(ymax - ymin + 1e-9)
                x_positions = {lvl: i for i, lvl in enumerate(sorted(clean[group_col].unique()))}
                for r in rows:
                    x = x_positions[r["other"]]
                    ax.text(x, y0, r["stars"], ha="center", va="bottom", fontsize=11)
                # save pairwise CSV
                pd.DataFrame(rows).to_csv(str(out_pdf).replace(".pdf","_pairwise.csv"), index=False)

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()

def multi_panel_violin(df, varying_col, value_col, fixed_col, fixed_levels, panel_pdf, per_panel_prefix,
                       title, xlab, ylab, n_perm=5000, baseline=None):
    # 5x1 panel
    n_panels = 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(5, 18), dpi=300, )
    axes = axes.flatten()
    lvls_set = set(fixed_levels)
    for i, lvl in enumerate(range(n_panels)):
        ax = axes[i]
        if lvl not in lvls_set:
            ax.set_axis_off()
            continue
        sub = df[df[fixed_col] == lvl][[varying_col, value_col]].dropna()
        if sub.empty:
            ax.set_axis_off()
            continue
        sns.violinplot(data=sub, x=varying_col, y=value_col, inner="quartile", cut=0, ax=ax)
        ax.set_title(f"{fixed_col}={lvl}")
        ax.set_xlabel(xlab)
        if i == 0:
            ax.set_ylabel(ylab)
        else:
            ax.set_ylabel("")
        # also output per-panel violin with stats
        out_pdf = Path(f"{per_panel_prefix}_{fixed_col}_{lvl}.pdf")
        violin_with_stats(sub.rename(columns={varying_col:"group"}), "group", value_col,
                          out_pdf, f"{title} ({fixed_col}={lvl})", xlab, ylab, n_perm=n_perm, baseline=baseline)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(panel_pdf, format="pdf")
    plt.close(fig)

def scatter_colored(df, col_Es1, col_f, out_pdf, title, xlab, ylab, xlim=None):
    d = df.dropna(subset=[col_Es1, col_f]).copy()
    if d.empty:
        return
    lam = 1239.841984 / d[col_Es1].values  # eV -> nm
    colors = [wavelength_to_rgb(w) for w in lam]
    plt.figure(figsize=(6.6, 5.2), dpi=300)
    plt.scatter(d[col_Es1], d[col_f], c=colors, s=10)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to std2.csv")
    ap.add_argument("--output_dir", type=str, default="results/std2/plots",
                    help="Output directory (default: results/std2/plots)")
    ap.add_argument("--n_permutations", type=int, default=5000,
                    help="Number of permutations for p-values (default: 5000)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib()
    df, col_Es1, col_f, col_str, col_abs = load_std2(args.csv)
    if not all([col_Es1, col_f, col_str, col_abs]):
        raise SystemExit("Required columns not found: need E_S1_eV, f_osc_S1, strength, absorption.")

    # ---------------- Heatmaps: mean/std over (strength, absorption) ----------------
    for val_col, pretty, fname in [
        (col_Es1, r"$E_{\mathrm{S1}}$ (eV)", "E_S1_eV"),
        (col_f,   r"$f_{\mathrm{osc}}(\mathrm{S1})$", "f_osc_S1"),
    ]:
        g = df[[col_str, col_abs, val_col]].dropna()
        mean_tbl = g.pivot_table(index=col_str, columns=col_abs, values=val_col, aggfunc="mean")
        std_tbl  = g.pivot_table(index=col_str, columns=col_abs, values=val_col, aggfunc="std")
        # save CSV
        mean_tbl.to_csv(out_dir / f"{fname}_mean.csv")
        std_tbl.to_csv(out_dir / f"{fname}_std.csv")
        # pretty heatmaps
        save_heatmap(mean_tbl, out_dir / f"{fname}_mean.pdf",
                     f"Mean of {pretty} over (strength, absorption)", "absorption (preconditioned token)", "strength (preconditioned token)")
        save_heatmap(std_tbl, out_dir / f"{fname}_std.pdf",
                     f"Std of {pretty} over (strength, absorption)", "absorption (preconditioned token)", "strength (preconditioned token)")

    # ---------------- Aggregated violins ----------------
    # E_S1 vs absorption (aggregate strengths)
    sub = df[[col_abs, col_Es1]].dropna().rename(columns={col_abs:"absorption_token", col_Es1:"E_S1_eV"})
    violin_with_stats(
        sub, "absorption_token", "E_S1_eV",
        out_dir / "violin_E_S1_by_absorption_all_strengths.pdf",
        title=r"$E_{\mathrm{S1}}$ by absorption (aggregating strengths)",
        xlab="absorption (preconditioned token)", 
        ylab="std2 E_S1 (eV)",
        n_perm=args.n_permutations,
        baseline=0
    )

    # f_osc vs strength (aggregate absorptions)
    sub = df[[col_str, col_f]].dropna().rename(columns={col_str:"strength_token", col_f:"f_osc_S1"})
    violin_with_stats(
        sub, "strength_token", "f_osc_S1",
        out_dir / "violin_fosc_by_strength_all_absorptions.pdf",
        title=r"$f_{\mathrm{osc}}(\mathrm{S1})$ by strength (aggregating absorptions)",
        xlab="strength (preconditioned token)",
        ylab="std2 oscillator strength (S1)",
        n_perm=args.n_permutations,
        baseline=0
    )

    # ---------------- Multi-panel violins ----------------
    # E_S1 by absorption within each strength = 0..4
    str_levels = sorted(pd.unique(df[col_str]))
    target_strengths = [lvl for lvl in range(5) if lvl in set(str_levels)]
    sub_df = df[[col_abs, col_Es1, col_str]].dropna()
    sub_df = sub_df.rename(columns={col_abs:"absorption_token", col_Es1:"E_S1_eV", col_str:"strength"})
    if target_strengths:
        multi_panel_violin(
            sub_df, varying_col="absorption_token", value_col="E_S1_eV",
            fixed_col="strength", fixed_levels=target_strengths,
            panel_pdf=out_dir / "violin_panels_E_S1_by_absorption_per_strength.pdf",
            per_panel_prefix=str(out_dir / "violin_E_S1_by_absorption_per_strength"),
            title=r"$E_{\mathrm{S1}}$ by absorption per strength",
            xlab="absorption (preconditioned token)",
            ylab="std2 E_S1 (eV)",
            n_perm=args.n_permutations, baseline=0
        )

    # f_osc by strength within each absorption = 0..4
    abs_levels = sorted(pd.unique(df[col_abs]))
    target_abs = [lvl for lvl in range(5) if lvl in set(abs_levels)]
    sub_df = df[[col_str, col_f, col_abs]].dropna()
    sub_df = sub_df.rename(columns={col_str:"strength_token", col_f:"f_osc_S1", col_abs:"absorption"})
    if target_abs:
        multi_panel_violin(
            sub_df, varying_col="strength_token", value_col="f_osc_S1",
            fixed_col="absorption", fixed_levels=target_abs,
            panel_pdf=out_dir / "violin_panels_fosc_by_strength_per_absorption.pdf",
            per_panel_prefix=str(out_dir / "violin_fosc_by_strength_per_absorption"),
            title=r"$f_{\mathrm{osc}}(\mathrm{S1})$ by strength per absorption",
            xlab="strength (preconditioned token)",
            ylab="std2 oscillator strength (S1)",
            n_perm=args.n_permutations, baseline=0
        )

    # ---------------- Scatterplots colored by wavelength ----------------
    scatter_colored(
        df, col_Es1, col_f,
        out_pdf=out_dir / "scatter_E_S1_vs_fosc_colored.pdf",
        title=r"$E_{\mathrm{S1}}$ vs $f_{\mathrm{osc}}(\mathrm{S1})$ (colored by $\lambda$)",
        xlab=r"$E_{\mathrm{S1}}$ (eV)", ylab=r"$f_{\mathrm{osc}}(\mathrm{S1})$"
    )
    # zoomed 1–4 eV
    scatter_colored(
        df, col_Es1, col_f,
        out_pdf=out_dir / "scatter_E_S1_vs_fosc_colored_zoom_1to4eV.pdf",
        title=r"$E_{\mathrm{S1}}$ vs $f_{\mathrm{osc}}(\mathrm{S1})$ (colored by $\lambda$, zoom 1–4 eV)",
        xlab=r"$E_{\mathrm{S1}}$ (eV)", ylab=r"$f_{\mathrm{osc}}(\mathrm{S1})$",
        xlim=(1.0, 4.0)
    )

    # ---------------- README ----------------
    with open(out_dir / "README_plots.txt", "w") as f:
        f.write(f"Figures written to: {out_dir}\n")
        f.write("Heatmaps: *_mean.pdf, *_std.pdf (+ CSV) for E_S1_eV and f_osc_S1 over (strength, absorption)\n")
        f.write("Violins: aggregated and 1x5 multi-panels (+ per-panel PDFs), pairwise CSVs with BH-FDR\n")
        f.write("Scatter: wavelength-colored, plus zoom (1–4 eV)\n")
        f.write(f"Permutation tests used n={args.n_permutations} shuffles\n")

if __name__ == "__main__":
    """Usage example:
    python src/plotting/std2_plots.py --csv results/std2/std2.csv \
    --output_dir results/std2/plots --n_permutations 5000
    """
    main()

