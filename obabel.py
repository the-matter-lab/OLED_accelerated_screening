#!/usr/bin/env python3
import os
import json
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

BASE_DIR = Path("compounds")
LOGS_DIR = Path("logs")

def run_obabel(task):
    mol_name, smiles = task
    start_dt = datetime.now().isoformat(timespec="seconds")
    t0 = time.monotonic()

    mol_dir = BASE_DIR / mol_name / "obabel"
    mol_dir.mkdir(parents=True, exist_ok=True)

    xyz_path = mol_dir / f"{mol_name}.xyz"
    cmd = ["obabel", f"-:{smiles}", "-O", str(xyz_path), "--gen3d", "best"]

    rc, error, stderr_tail = None, False, ""
    try:
        # simple call → fewer teardown issues; no shell, no pipe capture
        rc = subprocess.call(cmd)
        success = (rc == 0) and xyz_path.exists() and xyz_path.stat().st_size > 0
        error = not success
    except Exception as e:
        error = True
        rc = None
        stderr_tail = str(e)

    end_dt = datetime.now().isoformat(timespec="seconds")
    runtime_s = round(time.monotonic() - t0, 3)

    meta = {
        "smiles": smiles,
        "started": start_dt,
        "finished": end_dt,
        "runtime_seconds": runtime_s,
        "command": " ".join(cmd),
        "returncode": rc if rc is not None else -999,
        "error": bool(error),
        "stderr_tail": stderr_tail,
        "xyz_path": str(xyz_path),
    }
    (mol_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return mol_name if error else None

def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

if __name__ == "__main__":
    if shutil.which("obabel") is None:
        raise RuntimeError("obabel not found in PATH")

    df = pd.read_csv("smiles.csv")
    if "smiles" not in df.columns:
        raise ValueError("smiles.csv must contain a 'smiles' column")

    n = len(df)
    pad = max(3, len(str(n)))
    names = [f"mol{str(i+1).zfill(pad)}" for i in range(n)]
    tasks = list(zip(names, df["smiles"].astype(str).tolist()))

    BASE_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # conservative worker count to avoid FS contention
    env_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    n_workers = min(env_workers, 32)
    batch_size = 100

    print(f"Using {n_workers} workers (thread pool). Processing in batches of {batch_size}.")

    errors = []
    for i, batch in enumerate(chunks(tasks, batch_size), 1):
        print(f"Batch {i}: processing {len(batch)} molecules")
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for mol_error in tqdm(ex.map(run_obabel, batch),
                                  total=len(batch),
                                  desc=f"batch {i}",
                                  smoothing=0):
                if mol_error:
                    errors.append(mol_error)

    # Global log file
    log_path = LOGS_DIR / "obabel_data.log"
    with log_path.open("w") as logf:
        for mol_name, _ in tasks:
            meta_path = BASE_DIR / mol_name / "obabel" / "metadata.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                    line = (f"{mol_name} | rc={meta['returncode']} "
                            f"| error={meta['error']} "
                            f"| runtime={meta['runtime_seconds']}s "
                            f"| xyz={meta['xyz_path']}\n")
                except Exception as e:
                    line = f"{mol_name} | metadata parse error: {e}\n"
            else:
                line = f"{mol_name} | metadata missing\n"
            logf.write(line)

    if errors:
        print(f"completed with errors in {len(errors)} molecules. see {log_path}")
    else:
        print(f"all {len(tasks)} molecules processed successfully. see {log_path}")

