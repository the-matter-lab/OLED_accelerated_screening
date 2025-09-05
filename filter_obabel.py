#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

BASE_DIR = Path("compounds")
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

FAILED_JSON = LOGS_DIR / "failed_obabel.json"

def main():
    failed = []

    for mol_dir in sorted(BASE_DIR.glob("mol*")):
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

    if failed:
        FAILED_JSON.write_text(json.dumps(failed, indent=2))
        print(f"saved {len(failed)} failed molecules → {FAILED_JSON}")
    else:
        print("no failed molecules found")

if __name__ == "__main__":
    main()

