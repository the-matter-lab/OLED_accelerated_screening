#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import re

MOL_DIR_RE = re.compile(r"^mol\d+$")  # e.g. mol0001


def parse_out_file(out_path: Path) -> bool:
    """
    Return True if the ORCA TD-DFT calculation terminated correctly, False otherwise.

    We require:
      - general ORCA normal termination banner
      - TD-DFT-specific 'FINISHED WITHOUT ERROR' banner
    """
    try:
        text = out_path.read_text(errors="ignore")
    except Exception:
        return False

    ok_orca = "ORCA TERMINATED NORMALLY" in text
    ok_tddft = "*** ORCA-CIS/TD-DFT FINISHED WITHOUT ERROR ***" in text

    return ok_orca and ok_tddft


def main():
    ap = argparse.ArgumentParser(
        description="Check ORCA TD-DFT jobs under downselected/molXXXX/td-dft_wb97x-D3/"
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Directory containing molXXXX/td-dft_wb97x-D3 (e.g. downselected)",
    )
    ap.add_argument(
        "--outdir-name",
        default="td-dft_wb97x-D3",
        help="Subdirectory name used for ORCA TD-DFT runs (default: td-dft_wb97x-D3)",
    )

    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    missing_out = []
    failed = []
    ok = []

    for mol_dir in sorted(root.iterdir()):
        if not mol_dir.is_dir() or not MOL_DIR_RE.match(mol_dir.name):
            continue

        run_dir = mol_dir / args.outdir_name
        if not run_dir.is_dir():
            missing_out.append((mol_dir.name, "TD-DFT directory missing"))
            continue

        # Look for a *.out inside run_dir (each folder should contain exactly one job)
        outs = [p for p in run_dir.glob("*.out") if not p.name.startswith("slurm-")]
        if not outs:
            missing_out.append((mol_dir.name, "no .out file found"))
            continue

        # If multiple .out files exist, take the largest (most complete)
        out_file = max(outs, key=lambda p: p.stat().st_size)

        if parse_out_file(out_file):
            ok.append(mol_dir.name)
        else:
            failed.append(mol_dir.name)

    print("\n===== ORCA TD-DFT CHECK SUMMARY =====\n")

    total = len(ok) + len(failed) + len(missing_out)
    print(f"Total molecules found: {total}")
    print(f"Successful TD-DFT:      {len(ok)}")
    print(f"Failed TD-DFT:          {len(failed)}")
    print(f"Missing outputs:        {len(missing_out)}")

    if missing_out:
        print("\n--- Missing .out file / directory ---")
        for m, reason in missing_out:
            print(f"{m}: {reason}")

    if failed:
        print("\n--- Failed ORCA TD-DFT runs ---")
        for m in failed:
            print(m)

    print("\n=====================================\n")


if __name__ == "__main__":
    """
    Example:
    python src/checking/check_td-dft.py --root downselected
    """
    main()
