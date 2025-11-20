#!/usr/bin/env python3
import os
import sys
import argparse
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime
from shutil import which

MOL_DIR_RE = re.compile(r"^mol\d+$")  # e.g. mol0001


def find_orca_pltvib() -> str:
    """
    Locate the orca_pltvib executable.

    Priority:
      1) whatever is first in PATH
      2) $EBROOTORCA/orca_pltvib

    Raises RuntimeError if not found, with a hint about loading the ORCA module.
    """
    exe = which("orca_pltvib")
    if exe:
        return exe

    ebroot = os.environ.get("EBROOTORCA")
    if ebroot:
        cand = Path(ebroot) / "orca_pltvib"
        if cand.is_file():
            return str(cand)

    raise RuntimeError(
        "orca_pltvib not found in PATH or under $EBROOTORCA.\n"
        "Please load the ORCA module before running this script, e.g.:\n\n"
        "  module purge\n"
        "  module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/6.1.0\n"
    )



def load_imag_info(freq_dir: Path, json_name: str = "imaginary_freq.json"):
    """
    Load imaginary frequency information from imaginary_freq.json.

    Expected structure:
      {
        "contain_imaginary_freq": bool,
        "largest_imag": float or null,
        "all_imag": [float, ...]
      }
    """
    path = freq_dir / json_name
    if not path.is_file():
        return None

    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    return data


def needs_reopt(imag_data: dict, threshold: float) -> bool:
    """
    Decide whether this molecule requires reoptimization based on imaginary frequencies.

    Criteria:
      - contain_imaginary_freq must be True
      - and (|largest_imag| > threshold  or  number of imaginary modes > 1)
    """
    if not imag_data:
        return False

    if not imag_data.get("contain_imaginary_freq", False):
        return False

    largest = imag_data.get("largest_imag", None)
    all_imag = imag_data.get("all_imag", [])

    try:
        n_imag = len(all_imag)
    except TypeError:
        n_imag = 0

    cond_large = False
    if largest is not None:
        try:
            cond_large = abs(float(largest)) > float(threshold)
        except Exception:
            cond_large = False

    cond_multi = n_imag > 1

    return cond_large or cond_multi


def run_orca_pltvib(pltvib_exe: str, hess_path: Path, mode_index: int = 1) -> Path:
    """
    Run orca_pltvib on a .hess file to obtain a multi-frame XYZ for a given mode.

    On this system ORCA writes files as:
        <hess_filename>.vXXX.xyz
    e.g. mol1650.hess.v001.xyz

    Returns the path to the generated .vXXX.xyz file.
    """
    freq_dir = hess_path.parent

    # Mode index as three-digit
    mode_tag = f"v{mode_index:03d}.xyz"

    # ORCA naming: mol1650.hess.v001.xyz
    vib_xyz = freq_dir / f"{hess_path.name}.{mode_tag}"

    # If file already exists, do not re-run
    if not vib_xyz.is_file():
        cmd = [pltvib_exe, hess_path.name, str(mode_index)]
        res = subprocess.run(
            cmd,
            cwd=freq_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"orca_pltvib failed for {hess_path} (mode {mode_index}):\n"
                f"stdout:\n{res.stdout}\n\nstderr:\n{res.stderr}"
            )

    if not vib_xyz.is_file():
        raise RuntimeError(
            f"Expected vib XYZ file not found after orca_pltvib: {vib_xyz}"
        )

    return vib_xyz



def extract_frame_from_multixyz(vib_xyz: Path, frame_index: int = 8) -> str:
    """
    Extract one frame from a multi-frame XYZ file as a full XYZ string.

    The file is assumed to be a concatenation of standard XYZ blocks:
      N
      comment
      N lines with "Elem x y z"

    frame_index is zero-based: 0 is the first frame.
    """
    lines = vib_xyz.read_text().splitlines()
    if not lines:
        raise RuntimeError(f"{vib_xyz} is empty")

    try:
        natoms = int(lines[0].strip())
    except Exception as e:
        raise RuntimeError(f"Cannot read natoms from {vib_xyz}: {e}")

    block_size = natoms + 2
    if len(lines) < block_size:
        raise RuntimeError(f"{vib_xyz} does not contain a full XYZ block")

    nframes = len(lines) // block_size
    if nframes < 1:
        raise RuntimeError(f"{vib_xyz} does not contain any complete frame")

    # Clamp frame index if fewer frames than 9
    fi = min(max(frame_index, 0), nframes - 1)

    start = fi * block_size
    # first two lines are natoms and comment
    nat = lines[start : start + 1]
    comment = [lines[start + 1]]
    coords = lines[start + 2 : start + 2 + natoms]

    # normalise formatting: keep first 4 columns
    cleaned_coords = []
    for line in coords:
        parts = line.split()
        if len(parts) < 4:
            continue
        cleaned_coords.append(
            f"{parts[0]:2s}  {parts[1]}  {parts[2]}  {parts[3]}"
        )

    out_lines = []
    out_lines.append(str(natoms))
    out_lines.append(
        f"Displaced along imaginary normal mode; frame {fi} from {vib_xyz.name}"
    )
    out_lines.extend(cleaned_coords)

    return "\n".join(out_lines) + "\n"


def write_reopt_geometry(
    reopt_dir: Path,
    mol_name: str,
    xyz_text: str,
    imag_data: dict,
) -> None:
    """
    Write the displaced geometry and a small metadata file in reopt_dir.
    """
    reopt_dir.mkdir(parents=True, exist_ok=True)

    xyz_path = reopt_dir / f"{mol_name}_reopt.xyz"
    xyz_path.write_text(xyz_text)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "freq_b97_3c",
        "imaginary_info": imag_data,
        "note": "Geometry displaced along imaginary normal mode for reoptimization.",
        "xyz_file": xyz_path.name,
    }

    (reopt_dir / "reopt_metadata.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Create reopt_b97_3c folders for molecules with problematic "
            "imaginary frequencies, displacing geometries along the main "
            "imaginary normal mode using orca_pltvib."
        )
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Directory containing molXXXX subdirectories (e.g. downselected)",
    )
    ap.add_argument(
        "--freq-dir-name",
        default="freq_b97_3c",
        help="Frequency calculation directory name (default: freq_b97_3c)",
    )
    ap.add_argument(
        "--reopt-dir-name",
        default="reopt_b97_3c",
        help="Directory name where displaced geometries will be written (default: reopt_b97_3c)",
    )
    ap.add_argument(
        "--imag-threshold",
        type=float,
        default=20.0,
        help="Absolute imaginary frequency threshold (cm^-1) for reopt (default: 20.0)",
    )
    ap.add_argument(
        "--frame-index",
        type=int,
        default=8,
        help=(
            "Frame index (0-based) to extract from the multi-frame "
            "XYZ produced by orca_pltvib (default: 8)."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write or modify any files; only print what would be done.",
    )

    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    try:
        pltvib_exe = find_orca_pltvib()
    except RuntimeError as e:
        sys.exit(str(e))

    print(f"Using orca_pltvib: {pltvib_exe}")
    print(
        f"Root = {root}\n"
        f"freq dir name  = {args.freq_dir_name}\n"
        f"reopt dir name = {args.reopt_dir_name}\n"
        f"imag threshold = {args.imag_threshold} cm^-1\n"
        f"frame index    = {args.frame_index}"
    )

    n_mols = 0
    n_candidates = 0
    n_written = 0

    for mol_dir in sorted(root.iterdir()):
        if not mol_dir.is_dir() or not MOL_DIR_RE.match(mol_dir.name):
            continue

        n_mols += 1
        mol_name = mol_dir.name

        freq_dir = mol_dir / args.freq_dir_name
        if not freq_dir.is_dir():
            print(f"[SKIP] {mol_name}: missing freq directory {args.freq_dir_name}")
            continue

        imag_data = load_imag_info(freq_dir)
        if not needs_reopt(imag_data, args.imag_threshold):
            continue

        n_candidates += 1

        hess_path = freq_dir / f"{mol_name}.hess"
        if not hess_path.is_file():
            print(f"[WARN] {mol_name}: imaginary_freq.json suggests reopt, but {hess_path.name} not found")
            continue

        print(f"[CANDIDATE] {mol_name}: preparing displaced geometry for reoptimization")

        if args.dry_run:
            # In dry-run mode, do not generate or write anything
            continue

        try:
            vib_xyz = run_orca_pltvib(pltvib_exe, hess_path, mode_index=1)
            xyz_text = extract_frame_from_multixyz(vib_xyz, frame_index=args.frame_index)
        except Exception as e:
            print(f"[ERROR] {mol_name}: failed to generate displaced geometry: {e}")
            continue

        reopt_dir = mol_dir / args.reopt_dir_name
        try:
            write_reopt_geometry(reopt_dir, mol_name, xyz_text, imag_data)
        except Exception as e:
            print(f"[ERROR] {mol_name}: failed to write reopt geometry: {e}")
            continue

        n_written += 1
        print(f"[OK] {mol_name}: displaced geometry written to {reopt_dir}")

    print("\n===== set_reopt_imag_freq summary =====")
    print(f"Total molXXXX directories scanned: {n_mols}")
    print(f"Candidates requiring reopt:        {n_candidates}")
    print(f"Reopt geometries written:         {n_written}")
    print("======================================\n")


if __name__ == "__main__":
    """
    Example usage:

    # Dry run: see which molecules would be prepared for reoptimization
    python src/run/set_reopt_imag_freq.py --root downselected --dry-run

    # Actually create reopt_b97_3c geometries for |imag| > 20 cm^-1
    python src/run/set_reopt_imag_freq.py --root downselected

    # More conservative threshold (e.g. > 30 cm^-1) and different frame
    python src/run/set_reopt_imag_freq.py --root downselected \
        --imag-threshold 30.0 \
        --frame-index 6
    """
    main()
