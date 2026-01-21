#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_property_accuracy.py

Compute and plot "joint-property accuracy": fraction of molecules for which
BOTH preconditioning tokens match the TD-DFT-derived tokens:
    (precond_strength == td_strength_token) AND
    (precond_absorption == td_absorption_token)

Outputs:
  - overall_joint_accuracy.csv
  - joint_accuracy_by_precond_pair.csv
  - joint_accuracy_by_td_pair.csv
  - joint_confusion_counts_precond_vs_td.csv (pair-level confusion table)
  - PDF plots:
      * joint_confusion_precond_vs_td_counts.pdf
      * joint_accuracy_by_precond_pair.pdf
      * joint_accuracy_by_td_pair.pdf

Usage example (validation set):
    python src/plotting/joint_property_accuracy.py \
        --td-dft-csv results/validation/td-dft/VEE_wb97x-D3.csv \
        --std2-csv    results/validation/std2/std2.csv \
        --output-dir  results/validation/joint_accuracy
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Binning: edges for strength, absorption, splitting
# ---------------------------------------------------------------------------

BINS = dict(
    strength=np.array(
        [
            0.00000000e00,
            5.52315881e-04,
            4.80000000e-03,
            3.05000000e-02,
            3.21600000e-01,
            5.79904384e00,
        ],
        dtype=float,
    ),
    absorption=np.array(
        [0.049, 2.7575, 3.0675, 3.3489, 3.671, 12.796],
        dtype=float,
    ),
    splitting=np.array(
        [1.14075568, 6.20877618, 6.56544554, 6.92625786, 7.35424152, 42.863594],
        dtype=float,
    ),
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def setup_matplotlib() -> None:
    """LaTeX-safe Matplotlib + seaborn setup (mirrors std2_plots style)."""
    import shutil

    has_latex = shutil.which("latex") is not None
    if has_latex:
        matplotlib.rcParams.update(
            {"text.usetex": True, "font.family": "serif", "font.size": 11}
        )
    else:
        matplotlib.rcParams.update(
            {"text.usetex": False, "font.family": "serif", "font.size": 11}
        )

    sns.set_context("talk")
    sns.set_style("whitegrid")


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first column name in `candidates` that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def digitize_to_tokens(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """
    Map continuous values to integer tokens 0..(n_bins-1) using bin edges.

    Bins are defined as:
        [edges[0], edges[1]), [edges[1], edges[2]), ..., [edges[-2], edges[-1]]

    Out-of-range values are clipped into the closest bin.
    """
    arr = pd.to_numeric(values, errors="coerce").to_numpy()
    tokens = np.full_like(arr, fill_value=np.nan, dtype=float)

    mask = np.isfinite(arr)
    valid_vals = arr[mask]
    if valid_vals.size == 0:
        return pd.Series(tokens, index=values.index, dtype="float64")

    idx = np.searchsorted(edges, valid_vals, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    tokens[mask] = idx.astype(float)
    return pd.Series(tokens, index=values.index, dtype="float64")


def save_heatmap(
    tbl: pd.DataFrame,
    out_pdf: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    fmt: str = "d",
    figsize: Tuple[float, float] = (7.0, 6.0),
) -> None:
    """Save a simple annotated heatmap to PDF."""
    plt.figure(figsize=figsize, dpi=300)
    ax = plt.gca()
    sns.heatmap(tbl, annot=True, fmt=fmt, cmap="viridis", vmin=None, vmax=None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()


def save_barplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_pdf: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    rotate: int = 90,
    figsize: Tuple[float, float] = (10.0, 4.5),
) -> None:
    plt.figure(figsize=figsize, dpi=300)
    ax = plt.gca()
    sns.barplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    if rotate:
        plt.xticks(rotation=rotate, ha="center")
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()


# ---------------------------------------------------------------------------
# Loading and merging data
# ---------------------------------------------------------------------------

def load_std2(std2_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(std2_csv)
    df.columns = [c.strip() for c in df.columns]
    col_str = find_col(df, ["strength", "precond_strength", "target_strength"])
    col_abs = find_col(df, ["absorption", "precond_absorption", "target_absorption"])
    if col_str is None or col_abs is None:
        raise SystemExit(
            f"Required columns not found in std2 CSV {std2_csv}: "
            "need strength / absorption (or precond_*/target_* variants)."
        )

    for col in ["mol", "smiles"]:
        if col not in df.columns:
            raise SystemExit(f"std2 CSV {std2_csv} missing required column '{col}'")

    df = df[["mol", "smiles", col_str, col_abs]].copy()
    df = df.rename(columns={col_str: "precond_strength", col_abs: "precond_absorption"})
    return df


def load_td_dft(td_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(td_csv)
    df.columns = [c.strip() for c in df.columns]

    for col in ["mol", "smiles"]:
        if col not in df.columns:
            raise SystemExit(f"TD-DFT CSV {td_csv} missing required column '{col}'")

    col_Es1 = "E_S1_SOC(eV)" if "E_S1_SOC(eV)" in df.columns else "E_S1(eV)"
    col_f = "osc_S1_SOC" if "osc_S1_SOC" in df.columns else "osc_S1"

    for c in [col_Es1, col_f]:
        if c not in df.columns:
            raise SystemExit(
                f"TD-DFT CSV {td_csv} missing required column '{c}' "
                "(needed for absorption / strength tokens)."
            )

    df = df[["mol", "smiles", col_Es1, col_f]].copy()
    df = df.rename(columns={col_Es1: "E_S1_td", col_f: "f_osc_td"})
    df["E_S1_td"] = pd.to_numeric(df["E_S1_td"], errors="coerce")
    df["f_osc_td"] = pd.to_numeric(df["f_osc_td"], errors="coerce")
    return df


def merge_std2_td_dft(std2_df: pd.DataFrame, td_df: pd.DataFrame) -> pd.DataFrame:
    merged = td_df.merge(
        std2_df,
        on=["mol", "smiles"],
        how="inner",
        validate="one_to_one",
        suffixes=("_td", "_std2"),
    )
    if merged.empty:
        raise SystemExit("Merged DataFrame is empty; check that mol/smiles overlap between CSVs.")
    return merged


# ---------------------------------------------------------------------------
# Tokens and joint accuracy
# ---------------------------------------------------------------------------

def add_td_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["td_strength_token"] = digitize_to_tokens(df["f_osc_td"], BINS["strength"])
    df["td_absorption_token"] = digitize_to_tokens(df["E_S1_td"], BINS["absorption"])
    return df


def add_pair_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pair encodings for (strength, absorption) tokens to enable a single
    confusion matrix at the joint level.

    We define:
        pair_id = strength_token * n_abs + absorption_token
    """
    df = df.copy()

    n_str = len(BINS["strength"]) - 1
    n_abs = len(BINS["absorption"]) - 1

    for c in ["precond_strength", "precond_absorption", "td_strength_token", "td_absorption_token"]:
        # allow NaNs; cast later after dropna
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["precond_pair_id"] = df["precond_strength"] * n_abs + df["precond_absorption"]
    df["td_pair_id"] = df["td_strength_token"] * n_abs + df["td_absorption_token"]

    df["precond_pair_label"] = df.apply(
        lambda r: f"s{int(r['precond_strength'])}_a{int(r['precond_absorption'])}"
        if np.isfinite(r["precond_strength"]) and np.isfinite(r["precond_absorption"]) else np.nan,
        axis=1,
    )
    df["td_pair_label"] = df.apply(
        lambda r: f"s{int(r['td_strength_token'])}_a{int(r['td_absorption_token'])}"
        if np.isfinite(r["td_strength_token"]) and np.isfinite(r["td_absorption_token"]) else np.nan,
        axis=1,
    )

    return df


def compute_joint_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall joint accuracy + marginal accuracies for each property.
    """
    work = df[
        ["precond_strength", "precond_absorption", "td_strength_token", "td_absorption_token"]
    ].dropna().copy()

    work["precond_strength"] = work["precond_strength"].astype(int)
    work["precond_absorption"] = work["precond_absorption"].astype(int)
    work["td_strength_token"] = work["td_strength_token"].astype(int)
    work["td_absorption_token"] = work["td_absorption_token"].astype(int)

    strength_ok = (work["precond_strength"] == work["td_strength_token"]).to_numpy()
    absorption_ok = (work["precond_absorption"] == work["td_absorption_token"]).to_numpy()
    joint_ok = strength_ok & absorption_ok

    out = pd.DataFrame(
        {
            "n": [len(work)],
            "accuracy_strength": [float(np.mean(strength_ok))],
            "accuracy_absorption": [float(np.mean(absorption_ok))],
            "accuracy_joint": [float(np.mean(joint_ok))],
        }
    )
    return out


def joint_accuracy_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Joint accuracy stratified by either:
      - precond_pair_label
      - td_pair_label
    """
    work = df[
        [group_col, "precond_strength", "precond_absorption", "td_strength_token", "td_absorption_token"]
    ].dropna().copy()

    work["precond_strength"] = work["precond_strength"].astype(int)
    work["precond_absorption"] = work["precond_absorption"].astype(int)
    work["td_strength_token"] = work["td_strength_token"].astype(int)
    work["td_absorption_token"] = work["td_absorption_token"].astype(int)

    work["joint_ok"] = (
        (work["precond_strength"] == work["td_strength_token"])
        & (work["precond_absorption"] == work["td_absorption_token"])
    ).astype(int)

    stats = (
        work.groupby(group_col)
        .agg(n=("joint_ok", "size"), joint_accuracy=("joint_ok", "mean"))
        .reset_index()
        .sort_values(["n", group_col], ascending=[False, True])
        .reset_index(drop=True)
    )
    return stats


def compute_joint_confusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair-level confusion matrix: rows are preconditioning (s,a) pairs,
    columns are TD-DFT (s,a) pairs.
    """
    n_abs = len(BINS["absorption"]) - 1
    n_pairs = (len(BINS["strength"]) - 1) * n_abs

    work = df[["precond_pair_id", "td_pair_id"]].dropna().copy()
    work["precond_pair_id"] = work["precond_pair_id"].astype(int)
    work["td_pair_id"] = work["td_pair_id"].astype(int)

    labels = list(range(n_pairs))
    cm = pd.crosstab(
        work["precond_pair_id"],
        work["td_pair_id"],
        rownames=["preconditioning_pair_id"],
        colnames=["td_dft_pair_id"],
        dropna=False,
    )
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    return cm


def make_pair_ticklabels() -> Sequence[str]:
    n_str = len(BINS["strength"]) - 1
    n_abs = len(BINS["absorption"]) - 1
    labels = []
    for s in range(n_str):
        for a in range(n_abs):
            labels.append(f"s{s}_a{a}")
    return labels


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot joint-property accuracy: fraction where BOTH preconditioning tokens "
            "(strength, absorption) match TD-DFT-derived tokens."
        )
    )
    ap.add_argument(
        "--td-dft-csv",
        type=str,
        default="results/validation/td-dft/VEE_wb97x-D3.csv",
        help="Path to TD-DFT VEE_wb97x-D3.csv",
    )
    ap.add_argument(
        "--std2-csv",
        type=str,
        default="results/validation/std2/std2.csv",
        help="Path to std2 summary CSV",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="results/validation/joint_accuracy",
        help="Output directory for plots and CSVs",
    )
    ap.add_argument(
        "--min-count-barplot",
        type=int,
        default=10,
        help="Only include groups with at least this many molecules in bar plots (default: 10).",
    )
    args = ap.parse_args()

    td_csv = Path(args.td_dft_csv)
    std2_csv = Path(args.std2_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib()

    std2_df = load_std2(std2_csv)
    td_df = load_td_dft(td_csv)
    merged = merge_std2_td_dft(std2_df, td_df)
    merged = add_td_tokens(merged)
    merged = add_pair_tokens(merged)

    # Overall metrics
    overall = compute_joint_accuracy(merged)
    overall.to_csv(out_dir / "overall_joint_accuracy.csv", index=False)

    # Stratified metrics
    by_precond = joint_accuracy_by_group(merged, "precond_pair_label")
    by_td = joint_accuracy_by_group(merged, "td_pair_label")
    by_precond.to_csv(out_dir / "joint_accuracy_by_precond_pair.csv", index=False)
    by_td.to_csv(out_dir / "joint_accuracy_by_td_pair.csv", index=False)

    # Pair-level confusion matrix (counts)
    cm_pairs = compute_joint_confusion(merged)
    cm_pairs.to_csv(out_dir / "joint_confusion_counts_precond_vs_td.csv")

    # Plot pair-level confusion (counts) with (s,a) tick labels
    ticklabels = make_pair_ticklabels()
    cm_pairs_plot = cm_pairs.copy()
    cm_pairs_plot.index = ticklabels
    cm_pairs_plot.columns = ticklabels

    save_heatmap(
        cm_pairs_plot,
        out_dir / "joint_confusion_precond_vs_td_counts.pdf",
        title="Joint (strength, absorption): preconditioning vs TD-DFT (counts)",
        xlabel="TD-DFT (s,a) token pair",
        ylabel="Preconditioning (s,a) token pair",
        fmt="d",
        figsize=(10.5, 9.0),
    )

    # Bar plots for joint accuracy per pair (filtered by count)
    min_n = int(args.min_count_barplot)

    by_precond_f = by_precond[by_precond["n"] >= min_n].copy()
    by_td_f = by_td[by_td["n"] >= min_n].copy()

    if not by_precond_f.empty:
        save_barplot(
            by_precond_f,
            x="precond_pair_label",
            y="joint_accuracy",
            out_pdf=out_dir / "joint_accuracy_by_precond_pair.pdf",
            title=f"Joint accuracy by preconditioning (s,a) pair (n ≥ {min_n})",
            xlabel="Preconditioning token pair",
            ylabel="Joint accuracy",
            rotate=90,
            figsize=(11.5, 4.5),
        )

    if not by_td_f.empty:
        save_barplot(
            by_td_f,
            x="td_pair_label",
            y="joint_accuracy",
            out_pdf=out_dir / "joint_accuracy_by_td_pair.pdf",
            title=f"Joint accuracy by TD-DFT (s,a) pair (n ≥ {min_n})",
            xlabel="TD-DFT token pair",
            ylabel="Joint accuracy",
            rotate=90,
            figsize=(11.5, 4.5),
        )

    # Simple README
    with open(out_dir / "README_joint_accuracy.txt", "w") as f:
        f.write(f"Joint-property accuracy outputs written to: {out_dir}\n")
        f.write("Files:\n")
        f.write("  overall_joint_accuracy.csv\n")
        f.write("  joint_accuracy_by_precond_pair.csv\n")
        f.write("  joint_accuracy_by_td_pair.csv\n")
        f.write("  joint_confusion_counts_precond_vs_td.csv\n")
        f.write("  joint_confusion_precond_vs_td_counts.pdf\n")
        f.write("  joint_accuracy_by_precond_pair.pdf (if any groups meet min count)\n")
        f.write("  joint_accuracy_by_td_pair.pdf (if any groups meet min count)\n")


if __name__ == "__main__":
    main()
