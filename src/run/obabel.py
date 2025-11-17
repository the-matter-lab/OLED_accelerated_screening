#!/usr/bin/env python3
import os
import json
import time
import shutil
import signal
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_DIR = Path("compounds")  # will be overridden by --root


def run_obabel(task, timeout_s: int = 600):
    mol_name, smiles = task
    start_dt = datetime.now().isoformat(timespec="seconds")
    t0 = time.monotonic()

    mol_dir = BASE_DIR / mol_name / "obabel"
    mol_dir.mkdir(parents=True, exist_ok=True)

    xyz_path = mol_dir / f"{mol_name}.xyz"
    cmd = ["obabel", f"-:{smiles}", "-O", str(xyz_path), "--gen3d", "best"]

    rc, stdout, stderr, error = None, "", "", False
    try:
        # new process group → we can kill all children on timeout
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            # graceful then hard kill of the whole group
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    stdout, stderr = proc.communicate()
            except ProcessLookupError:
                pass
            rc = None
            error = True
            stderr = (stderr or "") + f"\n[TIMEOUT after {timeout_s}s]"
    except Exception as e:
        error = True
        rc = None
        stderr = str(e)

    success = (rc == 0) and xyz_path.exists() and xyz_path.stat().st_size > 0
    if not success:
        error = True

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
        "stdout_tail": (stdout or "")[-2000:],
        "stderr_tail": (stderr or "")[-2000:],
        "xyz_path": str(xyz_path),
    }
    (mol_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return mol_name if error else None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate 3D geometries with Open Babel from a SMILES CSV.\n"
            "Writes results into molXXXX/obabel/ under the root directory."
        )
    )
    parser.add_argument(
        "--smiles",
        required=True,
        help="Path to CSV file with a 'smiles' column.",
    )
    parser.add_argument(
        "--root",
        default="compounds",
        help="Root directory where molXXXX folders will be created (default: compounds)",
    )
    parser.add_argument(
        "--logs",
        default="logs",
        help="Directory where obabel.log will be written (default: logs)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds per molecule (default: 600)",
    )
    args = parser.parse_args()

    if shutil.which("obabel") is None:
        raise RuntimeError("obabel not found in PATH")

    smiles_path = Path(args.smiles).resolve()
    if not smiles_path.is_file():
        raise FileNotFoundError(f"SMILES CSV not found: {smiles_path}")

    df = pd.read_csv(smiles_path)
    if "smiles" not in df.columns:
        raise ValueError(f"{smiles_path} must contain a 'smiles' column")

    n = len(df)
    pad = max(3, len(str(n)))
    names = [f"mol{str(i + 1).zfill(pad)}" for i in range(n)]
    tasks = list(zip(names, df["smiles"].astype(str).tolist()))

    global BASE_DIR
    BASE_DIR = Path(args.root).resolve()
    BASE_DIR.mkdir(exist_ok=True, parents=True)

    logs_dir = Path(args.logs).resolve()
    logs_dir.mkdir(exist_ok=True, parents=True)

    # keep parallelism modest to avoid overloading filesystem/obabel
    default_workers = min(16, os.cpu_count() or 4)
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", default_workers))
    print(f"Using {n_workers} workers.")

    errors = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(run_obabel, t, args.timeout) for t in tasks
        ]  # timeout per molecule
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating XYZ files",
        ):
            mol_error = fut.result()
            if mol_error:
                errors.append(mol_error)

    # Global obabel.log
    log_path = logs_dir / "obabel.log"
    with log_path.open("w") as logf:
        for mol_name, smiles in tasks:
            meta_path = BASE_DIR / mol_name / "obabel" / "metadata.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                    line = (
                        f"{mol_name} | rc={meta['returncode']} "
                        f"| error={meta['error']} "
                        f"| runtime={meta['runtime_seconds']}s "
                        f"| xyz={meta['xyz_path']}\n"
                    )
                except Exception as e:
                    line = f"{mol_name} | metadata parse error: {e}\n"
            else:
                line = f"{mol_name} | metadata missing\n"
            logf.write(line)

    if errors:
        print(
            f"Completed with errors in {len(errors)} molecules. "
            f"See {log_path}"
        )
    else:
        print(
            f"All {len(tasks)} selected molecules processed successfully. "
            f"See {log_path}"
        )


if __name__ == "__main__":
    """Usage example:
    python src/run/obabel.py --smiles smiles.csv --root compounds --logs logs
    """
    main()
