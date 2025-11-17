#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import re

MOL_DIR_RE = re.compile(r"^mol\d+$")  # e.g. mol0001


def parse_out_file(out_path: Path) -> bool:
    """Return True if ORCA terminated normally, False otherwise."""
    try:
        text = out_path.read_text(errors="ignore")
    except Exception:
        return False
    return "ORCA TERMINATED NORMALLY" in text


def main():
    ap = argparse.ArgumentParser(
        description="Check ORCA opt_b97_3c jobs under downselected/molXXXX/opt_b97_3c/"
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Directory containing molXXXX/opt_b97_3c (e.g. downselected)",
    )
    ap.add_argument(
        "--outdir-name",
        default="opt_b97_3c",
        help="Subdirectory name used for ORCA runs (default: opt_b97_3c)",
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
            missing_out.append((mol_dir.name, "opt directory missing"))
            continue

        # Look for a *.out inside run_dir (each folder contains exactly one job)
        outs = list(run_dir.glob("*.out"))
        if not outs:
            missing_out.append((mol_dir.name, "no .out file found"))
            continue

        # If multiple .out files exist, take the largest (most complete)
        out_file = max(outs, key=lambda p: p.stat().st_size)

        if parse_out_file(out_file):
            ok.append(mol_dir.name)
        else:
            failed.append(mol_dir.name)

    print("\n===== ORCA opt_b97_3c CHECK SUMMARY =====\n")

    print(f"Total molecules found: {len(ok) + len(failed) + len(missing_out)}")
    print(f"Successful: {len(ok)}")
    print(f"Failed: {len(failed)}")
    print(f"Missing outputs: {len(missing_out)}")

    if missing_out:
        print("\n--- Missing .out file ---")
        for m, reason in missing_out:
            print(f"{m}: {reason}")

    if failed:
        print("\n--- Failed ORCA runs ---")
        for m in failed:
            print(m)

    print("\n==========================================\n")


if __name__ == "__main__":
    """
    Example:
    python src/checking/check_opt_b97_3c.py --root downselected
    """
    main()
