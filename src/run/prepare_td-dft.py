#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import re
import json
import shutil
from typing import Optional, Tuple, List

MOL_DIR_RE = re.compile(r"^mol\d+$")  # e.g. mol0001


def load_imaginary_info(freq_dir: Path) -> Optional[Tuple[List[float], bool]]:
    """
    Load imaginary frequency information from imaginary_freq.json
    inside freq_dir.

    Returns
    -------
    (all_imag, ok) or None
      all_imag : list of floats (imaginary frequencies, possibly empty)
      ok       : True if freq_dir exists and the JSON could be parsed,
                 False if freq_dir exists but JSON missing/corrupt.

    None is returned only if freq_dir does not exist at all.
    """
    if not freq_dir.is_dir():
        return None  # freq dir does not exist

    meta_path = freq_dir / "imaginary_freq.json"
    if not meta_path.is_file():
        print(f"[WARN] {freq_dir}: imaginary_freq.json not found")
        return ([], False)

    try:
        data = json.loads(meta_path.read_text())
    except Exception as e:
        print(f"[WARN] {freq_dir}: cannot parse imaginary_freq.json: {e}")
        return ([], False)

    all_imag = data.get("all_imag", [])
    if not isinstance(all_imag, list):
        all_imag = []

    # ensure floats
    cleaned = []
    for x in all_imag:
        try:
            cleaned.append(float(x))
        except Exception:
            continue

    return (cleaned, True)


def geometry_ok_for_td(all_imag: List[float], threshold: float) -> bool:
    """
    Decide if geometry is acceptable for TD-DFT, based on imaginary frequencies.

    Rules:
      - If no imaginary frequencies (len == 0) -> OK.
      - If exactly one imaginary frequency and |freq| < threshold -> OK.
      - Otherwise -> NOT OK.
    """
    if not all_imag:
        return True

    if len(all_imag) == 1 and abs(all_imag[0]) < threshold:
        return True

    return False


def choose_final_xyz_from_opt_or_reopt(dir_path: Path, mol_name: str) -> Path:
    """
    Choose the final optimised geometry .xyz from opt_b97_3c or reopt_b97_3c.

    Strategy:
      1. Prefer a file named {mol_name}.xyz if present.
      2. Otherwise, if there is exactly one .xyz file, use it.
      3. Otherwise, raise a RuntimeError.
    """
    cand_named = dir_path / f"{mol_name}.xyz"
    if cand_named.is_file():
        return cand_named

    xyzs = sorted(dir_path.glob("*.xyz"))
    if len(xyzs) == 1:
        return xyzs[0]

    raise RuntimeError(
        f"Cannot uniquely determine final geometry .xyz in {dir_path} "
        f"(found {[p.name for p in xyzs]})"
    )


def prepare_td_folder(
    mol_dir: Path,
    src_xyz: Path,
    geometry_source: str,
    freq_source: str,
    all_imag: List[float],
    td_dir_name: str = "td-dft_wb97x-D3",
) -> None:
    """
    Create td-dft_wb97x-D3 folder, copy geometry, and write metadata.

    Geometry is copied as {mol_name}.xyz.
    """
    mol_name = mol_dir.name
    td_dir = mol_dir / td_dir_name

    if td_dir.exists():
        if td_dir.is_dir():
            print(f"[SKIP] {mol_name}: {td_dir_name} already exists")
            return
        else:
            raise RuntimeError(f"{td_dir} exists and is not a directory")

    td_dir.mkdir(parents=True, exist_ok=False)

    td_xyz = td_dir / f"{mol_name}.xyz"
    shutil.copyfile(src_xyz, td_xyz)

    meta = {
        "molecule": mol_name,
        "geometry_source": geometry_source,   # "reopt_b97_3c" or "opt_b97_3c"
        "freq_source": freq_source,           # "freq_b97_3c_for_reopt" or "freq_b97_3c"
        "source_xyz": str(src_xyz),
        "td_xyz": str(td_xyz),
        "imaginary_frequencies": all_imag,
    }
    (td_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(
        f"[TD-DFT] {mol_name}: geometry from {geometry_source} "
        f"(freq: {freq_source}) → {td_dir_name}/{mol_name}.xyz"
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Prepare td-dft_wb97x-D3 folders selecting final geometries from "
            "reoptimised or original B97-3c optimisations based on imaginary "
            "frequency analysis."
        )
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Directory containing molXXXX subdirs (e.g. downselected)",
    )
    ap.add_argument(
        "--freq-reopt-dir",
        default="freq_b97_3c_for_reopt",
        help="Directory name for freq on reoptimised geometries (default: freq_b97_3c_for_reopt)",
    )
    ap.add_argument(
        "--reopt-dir",
        default="reopt_b97_3c",
        help="Directory name with reoptimised geometries (default: reopt_b97_3c)",
    )
    ap.add_argument(
        "--freq-dir",
        default="freq_b97_3c",
        help="Directory name for original freq calculations (default: freq_b97_3c)",
    )
    ap.add_argument(
        "--opt-dir",
        default="opt_b97_3c",
        help="Directory name with original optimisations (default: opt_b97_3c)",
    )
    ap.add_argument(
        "--td-dir-name",
        default="td-dft_wb97x-D3",
        help="Name of the folder to create for TD-DFT input (default: td-dft_wb97x-D3)",
    )
    ap.add_argument(
        "--imag-threshold",
        type=float,
        default=20.0,
        help="Threshold (cm^-1) for accepting a single small imaginary mode (default: 20.0)",
    )

    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    n_total = 0
    n_prepared = 0
    n_skipped_imag = 0
    n_skipped_missing = 0

    for mol_dir in sorted(root.iterdir()):
        if not mol_dir.is_dir() or not MOL_DIR_RE.match(mol_dir.name):
            continue
        n_total += 1
        mol_name = mol_dir.name

        # If TD-DFT folder already exists, skip immediately.
        td_dir = mol_dir / args.td_dir_name
        if td_dir.is_dir():
            print(f"[SKIP] {mol_name}: {args.td_dir_name} already exists")
            continue

        # 1) Try reoptimised branch first: freq_b97_3c_for_reopt + reopt_b97_3c
        freq_reopt_dir = mol_dir / args.freq_reopt_dir
        reopt_dir = mol_dir / args.reopt_dir

        chose_geometry = False

        info_reopt = load_imaginary_info(freq_reopt_dir)
        if info_reopt is not None:
            all_imag_reopt, ok_json = info_reopt
            if not ok_json:
                print(f"[WARN] {mol_name}: reopt freq imaginary info incomplete; trying original freq")
            else:
                if geometry_ok_for_td(all_imag_reopt, args.imag_threshold):
                    try:
                        src_xyz = choose_final_xyz_from_opt_or_reopt(reopt_dir, mol_name)
                    except Exception as e:
                        print(f"[WARN] {mol_name}: cannot select geometry from {reopt_dir}: {e}")
                    else:
                        prepare_td_folder(
                            mol_dir,
                            src_xyz=src_xyz,
                            geometry_source=args.reopt_dir,
                            freq_source=args.freq_reopt_dir,
                            all_imag=all_imag_reopt,
                            td_dir_name=args.td_dir_name,
                        )
                        n_prepared += 1
                        chose_geometry = True
                else:
                    print(
                        f"[SKIP] {mol_name}: reopt freq has unacceptable imaginary frequencies "
                        f"{all_imag_reopt} (will try original freq)"
                    )

        if chose_geometry:
            continue  # already prepared TD folder from reopt branch

        # 2) Fallback to original branch: freq_b97_3c + opt_b97_3c
        freq_dir = mol_dir / args.freq_dir
        opt_dir = mol_dir / args.opt_dir

        info_orig = load_imaginary_info(freq_dir)
        if info_orig is None:
            print(
                f"[SKIP] {mol_name}: no usable freq directory "
                f"('{args.freq_reopt_dir}' or '{args.freq_dir}')"
            )
            n_skipped_missing += 1
            continue

        all_imag_orig, ok_json_orig = info_orig
        if not ok_json_orig:
            print(f"[SKIP] {mol_name}: original freq imaginary info missing/corrupt")
            n_skipped_missing += 1
            continue

        if not geometry_ok_for_td(all_imag_orig, args.imag_threshold):
            print(
                f"[SKIP] {mol_name}: original freq has unacceptable imaginary frequencies "
                f"{all_imag_orig}"
            )
            n_skipped_imag += 1
            continue

        # Imaginary frequencies OK; use opt_b97_3c geometry
        try:
            src_xyz = choose_final_xyz_from_opt_or_reopt(opt_dir, mol_name)
        except Exception as e:
            print(f"[WARN] {mol_name}: cannot select geometry from {opt_dir}: {e}")
            n_skipped_missing += 1
            continue

        prepare_td_folder(
            mol_dir,
            src_xyz=src_xyz,
            geometry_source=args.opt_dir,
            freq_source=args.freq_dir,
            all_imag=all_imag_orig,
            td_dir_name=args.td_dir_name,
        )
        n_prepared += 1

    print("\n===== TD-DFT PREPARATION SUMMARY =====")
    print(f"Total molXXXX directories scanned:   {n_total}")
    print(f"TD-DFT folders created:             {n_prepared}")
    print(f"Skipped (bad/multiple imaginaries): {n_skipped_imag}")
    print(f"Skipped (missing/corrupt info):     {n_skipped_missing}")
    print("======================================\n")


if __name__ == "__main__":
    """
    Example:
    python src/run/prepare_td-dft.py --root downselected

    Logic per molecule:
      1. If freq_b97_3c_for_reopt exists and its imaginary_freq.json indicates:
           - no imaginary frequencies, or
           - exactly one imaginary with |freq| < imag_threshold
         then use final geometry from reopt_b97_3c as source.

      2. Otherwise, fall back to freq_b97_3c + opt_b97_3c with the same criteria.

      3. For accepted geometries, create td-dft_wb97x-D3, copy the chosen
         final geometry as molXXXX.xyz, and write metadata.json including
         where the geometry came from and the imaginary frequencies.
    """
    main()
