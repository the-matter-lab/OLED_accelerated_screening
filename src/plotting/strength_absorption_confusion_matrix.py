#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strength_absorption_confusion_matrix.py

Compute confusion matrices between:
  - preconditioning tokens (from std2: strength, absorption)
  - final TD-DFT tokens (from VEE_wb97x-D3: binned osc_S1_SOC, E_S1_SOC(eV))

It also optionally computes scaffold statistics (Murcko scaffolds) per
TD-DFT token to help identify common structural motifs associated with
each token.

Usage example (validation set):
    python src/plotting/strength_absorption_confusion_matrix.py \
        --td-dft-csv results/validation/td-dft/VEE_wb97x-D3.csv \
        --std2-csv    results/validation/std2/std2.csv \
        --output-dir  results/validation/confusion \
        --do-scaffold-analysis
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

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
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.size": 11,
            }
        )
    else:
        matplotlib.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.size": 11,
            }
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

    # searchsorted with side="right" gives index of first edge > v;
    # subtract 1 → largest edge <= v
    idx = np.searchsorted(edges, valid_vals, side="right") - 1
    # clip into valid bin indices 0..len(edges)-2
    idx = np.clip(idx, 0, len(edges) - 2)
    tokens[mask] = idx.astype(float)
    return pd.Series(tokens, index=values.index, dtype="float64")


def save_heatmap(tbl: pd.DataFrame, out_pdf: Path, title: str, xlabel: str, ylabel: str, fmt: str = "d") -> None:
    """Save a simple annotated heatmap to PDF."""
    plt.figure(figsize=(6.0, 5.0), dpi=300)
    ax = plt.gca()
    sns.heatmap(tbl, annot=True, fmt=fmt, cmap="viridis", vmin=None, vmax=None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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

    keep_cols = ["mol", "smiles", col_str, col_abs]
    for col in ["mol", "smiles"]:
        if col not in df.columns:
            raise SystemExit(f"std2 CSV {std2_csv} missing required column '{col}'")

    df = df[keep_cols].copy()
    df = df.rename(
        columns={
            col_str: "precond_strength",
            col_abs: "precond_absorption",
        }
    )
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

    keep_cols = ["mol", "smiles", col_Es1, col_f]
    df = df[keep_cols].copy()
    df = df.rename(
        columns={
            col_Es1: "E_S1_td",
            col_f: "f_osc_td",
        }
    )

    df["E_S1_td"] = pd.to_numeric(df["E_S1_td"], errors="coerce")
    df["f_osc_td"] = pd.to_numeric(df["f_osc_td"], errors="coerce")
    return df


def merge_std2_td_dft(std2_df: pd.DataFrame, td_df: pd.DataFrame) -> pd.DataFrame:
    """Inner join on mol + smiles to keep only molecules present in both."""
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
# Confusion matrix computation
# ---------------------------------------------------------------------------

def compute_confusion(
    df: pd.DataFrame,
    precond_col: str,
    td_token_col: str,
    n_bins: int,
) -> pd.DataFrame:
    """
    Compute a confusion matrix (counts) between:
        preconditioning token   (rows)
        TD-DFT-derived token    (columns)
    """
    sub = df[[precond_col, td_token_col]].dropna().copy()
    sub[precond_col] = sub[precond_col].astype(int)
    sub[td_token_col] = sub[td_token_col].astype(int)

    labels = list(range(n_bins))
    cm = pd.crosstab(
        sub[precond_col],
        sub[td_token_col],
        rownames=["preconditioning_token"],
        colnames=["td_dft_token"],
        dropna=False,
    )
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    return cm


def add_td_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Add TD-DFT-based strength/absorption token columns using BINS."""
    df = df.copy()
    df["td_strength_token"] = digitize_to_tokens(df["f_osc_td"], BINS["strength"])
    df["td_absorption_token"] = digitize_to_tokens(df["E_S1_td"], BINS["absorption"])
    return df


# ---------------------------------------------------------------------------
# Scaffold / functional group analysis (optional)
# ---------------------------------------------------------------------------

def try_import_rdkit():
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
    except Exception as e:  # pragma: no cover - environment-dependent
        print(f"[WARN] RDKit not available ({e}); skipping scaffold analysis.")
        return None, None
    return Chem, MurckoScaffold


def smiles_to_scaffold(smiles: str, Chem, MurckoScaffold) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaff = MurckoScaffold.GetScaffoldForMol(mol)
    if scaff is None:
        return None
    return Chem.MolToSmiles(scaff)


def scaffold_stats(
    df: pd.DataFrame,
    token_col: str,
    out_prefix: Path,
) -> None:
    """
    For each TD-DFT token value in `token_col`, compute how often each
    Murcko scaffold appears, and write CSVs with counts and fractions.
    """
    Chem, MurckoScaffold = try_import_rdkit()
    if Chem is None or MurckoScaffold is None:
        return

    work = df[["smiles", token_col]].dropna().copy()
    if work.empty:
        print(f"[INFO] No data for scaffold analysis on column '{token_col}'.")
        return

    print(f"[INFO] Computing Murcko scaffolds for {len(work)} molecules...")
    work["scaffold"] = work["smiles"].apply(
        lambda s: smiles_to_scaffold(s, Chem, MurckoScaffold)
    )
    work = work.dropna(subset=["scaffold"])
    if work.empty:
        print(f"[WARN] All scaffolds failed to generate for '{token_col}'.")
        return

    counts = (
        work.groupby([token_col, "scaffold"])
        .size()
        .reset_index(name="count")
    )
    # per-token totals and fractions
    counts["total_for_token"] = counts.groupby(token_col)["count"].transform("sum")
    counts["fraction"] = counts["count"] / counts["total_for_token"]
    counts = counts.sort_values([token_col, "count"], ascending=[True, False])

    counts.to_csv(out_prefix.with_suffix(".csv"), index=False)

    # also write a "top-N per token" CSV for quicker inspection
    top_n = counts.groupby(token_col).head(15).reset_index(drop=True)
    top_n.to_csv(out_prefix.with_name(out_prefix.stem + "_top15.csv"), index=False)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compute confusion matrices between preconditioning tokens "
            "(std2 strength/absorption) and TD-DFT tokens (binned osc_S1, E_S1)."
        )
    )
    ap.add_argument(
        "--td-dft-csv",
        type=str,
        default="results/validation/td-dft/VEE_wb97x-D3.csv",
        help="Path to TD-DFT VEE_wb97x-D3.csv (default: results/validation/td-dft/VEE_wb97x-D3.csv)",
    )
    ap.add_argument(
        "--std2-csv",
        type=str,
        default="results/validation/std2/std2.csv",
        help="Path to std2 summary CSV (default: results/validation/std2/std2.csv)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="results/validation/confusion",
        help="Output directory for confusion matrices and plots.",
    )
    ap.add_argument(
        "--do-scaffold-analysis",
        action="store_true",
        help="If set, compute Murcko-scaffold statistics per TD-DFT token.",
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

    n_bins = len(BINS["strength"]) - 1

    # Strength confusion: preconditioning token vs TD-DFT strength token
    cm_strength = compute_confusion(
        merged,
        precond_col="precond_strength",
        td_token_col="td_strength_token",
        n_bins=n_bins,
    )
    cm_strength.to_csv(out_dir / "confusion_strength_counts.csv")

    cm_strength_row = cm_strength.div(
        cm_strength.sum(axis=1).replace(0, np.nan),
        axis=0,
    )
    cm_strength_row.to_csv(out_dir / "confusion_strength_row_normalized.csv")

    save_heatmap(
        cm_strength,
        out_dir / "confusion_strength_counts.pdf",
        title="Strength: preconditioning vs TD-DFT tokens (counts)",
        xlabel="TD-DFT strength token",
        ylabel="Preconditioning strength token",
        fmt="d",
    )
    save_heatmap(
        cm_strength_row,
        out_dir / "confusion_strength_row_normalized.pdf",
        title="Strength: preconditioning vs TD-DFT tokens (row-normalized)",
        xlabel="TD-DFT strength token",
        ylabel="Preconditioning strength token",
        fmt=".2f",
    )

    # Absorption confusion: preconditioning token vs TD-DFT absorption token
    cm_abs = compute_confusion(
        merged,
        precond_col="precond_absorption",
        td_token_col="td_absorption_token",
        n_bins=len(BINS["absorption"]) - 1,
    )
    cm_abs.to_csv(out_dir / "confusion_absorption_counts.csv")

    cm_abs_row = cm_abs.div(
        cm_abs.sum(axis=1).replace(0, np.nan),
        axis=0,
    )
    cm_abs_row.to_csv(out_dir / "confusion_absorption_row_normalized.csv")

    save_heatmap(
        cm_abs,
        out_dir / "confusion_absorption_counts.pdf",
        title="Absorption: preconditioning vs TD-DFT tokens (counts)",
        xlabel="TD-DFT absorption token",
        ylabel="Preconditioning absorption token",
        fmt="d",
    )
    save_heatmap(
        cm_abs_row,
        out_dir / "confusion_absorption_row_normalized.pdf",
        title="Absorption: preconditioning vs TD-DFT tokens (row-normalized)",
        xlabel="TD-DFT absorption token",
        ylabel="Preconditioning absorption token",
        fmt=".2f",
    )

    # Optional scaffold analysis on TD-DFT tokens
    if args.do_scaffold_analysis:
        scaffold_stats(
            merged,
            token_col="td_strength_token",
            out_prefix=out_dir / "scaffolds_by_td_strength_token",
        )
        scaffold_stats(
            merged,
            token_col="td_absorption_token",
            out_prefix=out_dir / "scaffolds_by_td_absorption_token",
        )

    # README to document outputs
    with open(out_dir / "README_confusion.txt", "w") as f:
        f.write(f"Confusion matrices and scaffold stats written to: {out_dir}\n")
        f.write("Files:\n")
        f.write("  confusion_strength_counts.csv / _row_normalized.csv / *.pdf\n")
        f.write("  confusion_absorption_counts.csv / _row_normalized.csv / *.pdf\n")
        f.write("  scaffolds_by_td_strength_token*.csv (if --do-scaffold-analysis)\n")
        f.write("  scaffolds_by_td_absorption_token*.csv (if --do-scaffold-analysis)\n")


if __name__ == "__main__":
    """
    Usage:
        python src/plotting/strength_absorption_confusion_matrix.py \
            --td-dft-csv results/validation/td-dft/VEE_wb97x-D3.csv \
            --std2-csv    results/validation/std2/std2.csv \
            --output-dir  results/validation/confusion \
            --do-scaffold-analysis
    """
    main()

