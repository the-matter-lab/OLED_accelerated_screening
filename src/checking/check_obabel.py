#!/usr/bin/env python3
import json
import shutil
import argparse
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Remove molecules whose obabel/metadata.json shows error=True."
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Path to the compounds directory (e.g. compounds or test_compounds)",
    )
    ap.add_argument(
        "--logs",
        required=True,
        help="Path to the logs directory where failed_obabel.json will be written",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    base_dir = Path(args.root).resolve()
    logs_dir = Path(args.logs).resolve()
    logs_dir.mkdir(exist_ok=True)

    failed_json_path = logs_dir / "failed_obabel.json"

    if not base_dir.is_dir():
        raise SystemExit(f"ERROR: root directory not found → {base_dir}")

    failed = []

    # Loop through molXXXX directories
    for mol_dir in sorted(base_dir.glob("mol*")):
        meta_path = mol_dir / "obabel" / "metadata.json"
        if not meta_path.is_file():
            continue

        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            print(f"[WARN] cannot parse {meta_path}: {e}")
            continue

        if meta.get("error", False):
            print(f"[FAIL] {mol_dir.name} marked as error → removing")
            failed.append(meta)
            shutil.rmtree(mol_dir, ignore_errors=True)

    # Write log summary
    if failed:
        failed_json_path.write_text(json.dumps(failed, indent=2))
        print(f"saved {len(failed)} failed molecules → {failed_json_path}")
    else:
        print("no failed molecules found")


if __name__ == "__main__":
    """
    Usage:
    python src/checking/check_obabel.py --root compounds --logs logs"""
    main()
