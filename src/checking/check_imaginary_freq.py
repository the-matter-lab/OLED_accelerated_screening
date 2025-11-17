#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import re
import json

MOL_DIR_RE = re.compile(r"^mol\d+$")  # e.g. mol0001


def parse_out_file(out_path: Path) -> bool:
    """Return True if ORCA terminated normally, False otherwise."""
    try:
        text = out_path.read_text(errors="ignore")
    except Exception:
        return False
    return "ORCA TERMINATED NORMALLY" in text


def find_imaginary_frequencies(out_path: Path):
    """
    Scan an ORCA frequency .out file for imaginary modes.

    Returns
    -------
    list of floats
        List of imaginary frequencies (cm**-1). Empty if none found.
    """
    try:
        text = out_path.read_text(errors="ignore")
    except Exception:
        return []

    imag_freqs = []
    # ORCA prints e.g.  -45.67 cm**-1  and marks line with ***imaginary mode***
    pattern = re.compile(r"(-?\d+\.\d+)\s+cm\*\*-1")

    for line in text.splitlines():
        if "***imaginary mode***" in line:
            m = pattern.search(line)
            if m:
                try:
                    imag_freqs.append(float(m.group(1)))
                except ValueError:
                    # If parse fails, just skip this one
                    continue

    return imag_freqs


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Check ORCA frequency runs (freq_b97_3c) under downselected/molXXXX "
            "for imaginary frequencies."
        )
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Directory containing molXXXX/freq_b97_3c (e.g. downselected)",
    )
    ap.add_argument(
        "--outdir-name",
        default="freq_b97_3c",
        help="Subdirectory name used for ORCA freq runs (default: freq_b97_3c)",
    )

    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    total = 0
    successful = 0
    with_imag = 0

    mols_with_imag = []

    for mol_dir in sorted(root.iterdir()):
        if not mol_dir.is_dir() or not MOL_DIR_RE.match(mol_dir.name):
            continue

        run_dir = mol_dir / args.outdir_name
        if not run_dir.is_dir():
            print(f"[WARN] {mol_dir.name}: freq directory '{args.outdir_name}' missing")
            continue

        outs = [p for p in run_dir.glob("*.out") if not p.name.startswith("slurm-")]
        if not outs:
            print(f"[WARN] {mol_dir.name}: no .out file found in {run_dir}")
            continue

        # If multiple .out files exist, take the largest (most complete)
        out_file = max(outs, key=lambda p: p.stat().st_size)

        total += 1

        if not parse_out_file(out_file):
            print(f"[WARN] {mol_dir.name}: ORCA did not terminate normally ({out_file.name})")
            # For failed runs we do not create imaginary_freq.json
            continue

        successful += 1
        imag_freqs = find_imaginary_frequencies(out_file)

        contain_imag = bool(imag_freqs)
        # "largest" imaginary: these are negative; we take the most negative value
        largest_imag = min(imag_freqs) if imag_freqs else None

        # Write per-molecule JSON inside freq directory
        meta_path = run_dir / "imaginary_freq.json"
        meta = {
            "contain_imaginary_freq": contain_imag,
            "largest_imag": largest_imag,
            "all_imag": imag_freqs,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        if contain_imag:
            with_imag += 1
            mols_with_imag.append(mol_dir.name)
            freqs_str = ", ".join(f"{f:.2f} cm**-1" for f in imag_freqs)
            print(f"[IMAG] {mol_dir.name}: imaginary frequencies detected: {freqs_str}")

    print("\n===== ORCA freq_b97_3c IMAGINARY FREQUENCY CHECK =====\n")
    print(f"Total freq .out processed (ORCA normal termination): {successful}")
    print(f"Molecules with imaginary frequencies: {with_imag}")
    if mols_with_imag:
        print("\nList of molecules with imaginary modes:")
        for m in mols_with_imag:
            print(f"  {m}")
    print("\n=======================================================\n")


if __name__ == "__main__":
    """
    Example:
    python src/checking/check_imaginary_freq.py --root downselected
    """
    main()
