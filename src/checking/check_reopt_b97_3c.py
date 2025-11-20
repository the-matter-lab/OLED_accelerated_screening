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
        description="Check ORCA reopt_b97_3c jobs under downselected/molXXXX/reopt_b97_3c/"
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Directory containing molXXXX/reopt_b97_3c (e.g. downselected)",
    )
    ap.add_argument(
        "--outdir-name",
        default="reopt_b97_3c",
        help="Subdirectory name used for ORCA reopt runs (default: reopt_b97_3c)",
    )

    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    missing_dir = []
    missing_out = []
    failed = []
    ok = []

    for mol_dir in sorted(root.iterdir()):
        if not mol_dir.is_dir() or not MOL_DIR_RE.match(mol_dir.name):
            continue

        run_dir = mol_dir / args.outdir_name
        if not run_dir.is_dir():
            missing_dir.append(mol_dir.name)
            continue

        outs = list(run_dir.glob("*.out"))
        if not outs:
            missing_out.append(mol_dir.name)
            continue

        out_file = max(outs, key=lambda p: p.stat().st_size)

        if parse_out_file(out_file):
            ok.append(mol_dir.name)
        else:
            failed.append(mol_dir.name)

    print("\n===== ORCA reopt_b97_3c CHECK SUMMARY =====\n")

    total = len(ok) + len(failed) + len(missing_dir) + len(missing_out)

    print(f"Total molecules scanned:   {total}")
    print(f"Successful reopts:         {len(ok)}")
    print(f"Failed reopts:             {len(failed)}")
    print(f"No reopt directory:        {len(missing_dir)}")
    print(f"Missing .out file:         {len(missing_out)}")

    if failed:
        print("\n--- Failed ORCA reopt runs ---")
        for m in failed:
            print(m)

    if missing_out:
        print("\n--- reopt directory present but .out missing ---")
        for m in missing_out:
            print(m)

    if missing_dir:
        print("\n--- Molecules without reopt_b97_3c directory (EXPECTED for many) ---")
        print(" ".join(missing_dir))

    print("\n=============================================\n")


if __name__ == "__main__":
    """
    Example:
    python src/checking/check_reopt_b97_3c.py --root downselected
    """
    main()
