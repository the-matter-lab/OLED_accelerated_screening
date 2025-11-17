#!/usr/bin/env python3
import argparse
import math
import shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(
        description="Partition compounds/ into compounds/partition_* subfolders."
    )
    ap.add_argument("--root", default="compounds",
                    help="Root folder containing molXXXX subfolders")
    ap.add_argument("--partition_size", type=int, required=True,
                    help="Number of molecules per partition")
    ap.add_argument("--symlink", action="store_true",
                    help="Use symlinks instead of moving mol folders (saves disk space)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    mol_dirs = sorted([d for d in root.iterdir()
                       if d.is_dir() and d.name.startswith("mol")])

    n = len(mol_dirs)
    if n == 0:
        raise RuntimeError(f"No molXXXX folders found in {root}")

    n_parts = math.ceil(n / args.partition_size)
    print(f"Found {n} molecules in {root}")
    print(f"Partitioning into {n_parts} partitions of up to {args.partition_size} molecules each")

    for p in range(n_parts):
        start = p * args.partition_size
        end = min((p + 1) * args.partition_size, n)
        part_dirs = mol_dirs[start:end]
        part_folder = root / f"partition_{p}"
        part_folder.mkdir(parents=True, exist_ok=True)
        print(f"Creating {part_folder} with {len(part_dirs)} molecules")

        for d in part_dirs:
            dst = part_folder / d.name
            if dst.exists():
                continue
            if args.symlink:
                dst.symlink_to(d.resolve())
            else:
                shutil.move(str(d), dst)

    print("Done.")

if __name__ == "__main__":
    """Usage example:
    python src/run/partition_compounds.py --root compounds --partition_size 1000 """
    main()

