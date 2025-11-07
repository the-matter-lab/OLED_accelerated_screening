#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import re
import subprocess
from datetime import datetime
import json

MOL_RE = re.compile(r"^mol\d+$")  # e.g., mol0001
SUCCESS_BANNER = "CREST terminated normally."

def crest_ok(crest_dir: Path) -> bool:
    out = crest_dir / "crest.out"
    if not out.is_file():
        return False
    try:
        return SUCCESS_BANNER in out.read_text(errors="ignore")
    except Exception:
        return False

def std2_run_ok(std2_dir: Path) -> bool:
    """
    Success only if:
      - xtb4stda had ' 1  SCC done.'
      - std2_singlets.out has 'sTDA done.'
      - std2_triplets.out has 'sTDA done.'
    Prefer metadata.json if present; otherwise scan .out files.
    """
    meta = std2_dir / "metadata.json"
    if meta.is_file():
        try:
            d = json.loads(meta.read_text())
            xtb_ok  = not d.get("xtb4stda_error", True)
            sing_ok = not d.get("std2_singlets_error", True)
            trip_ok = not d.get("std2_triplets_error", True)
            if xtb_ok and sing_ok and trip_ok:
                return True
        except Exception:
            pass

    def contains(p: Path, needle: str) -> bool:
        if not p.is_file():
            return False
        try:
            return (needle in p.read_text(errors="ignore"))
        except Exception:
            return False

    xtb_ok  = contains(std2_dir / "xtb4stda.out",      " 1  SCC done.")
    sing_ok = contains(std2_dir / "std2_singlets.out", "sTDA done.")
    trip_ok = contains(std2_dir / "std2_triplets.out", "sTDA done.")
    return xtb_ok and sing_ok and trip_ok

def load_template(path: Path) -> str:
    return path.read_text()

def write_array_script(template_text: str, out_path: Path, *,
                       cpus_per_task: int, job_name: str, list_file: Path,
                       chunk_size: int, gbsa: str, emax_ev: float,
                       walltime: str):
    walltime_line = f"#SBATCH --time={walltime}" if walltime else ""
    tokens = {
        "[[CPUS_PER_TASK]]": str(cpus_per_task),
        "[[JOB_NAME]]": job_name,
        "[[LIST_FILE]]": str(list_file),
        "[[CHUNK_SIZE]]": str(chunk_size),
        "[[GBSA_SOLVENT]]": gbsa,
        "[[EMAX_EV]]": f"{emax_ev:g}",
        "[[WALLTIME_LINE]]": walltime_line,
    }
    script = template_text
    for k, v in tokens.items():
        script = script.replace(k, v)
    out_path.write_text(script)
    out_path.chmod(0o750)

def main():
    ap = argparse.ArgumentParser(
        description="Submit std2 (singlets+triplets) via a Slurm job array that processes molecules sequentially per task."
    )
    ap.add_argument("--partition_dir", required=True, help="e.g., compounds/partition_0")
    ap.add_argument("--templates", default="sh_templates", help="Folder with std2.sh template")
    ap.add_argument("--cpus-per-task", type=int, default=40, help="cpus-per-task per array task (default 40)")
    ap.add_argument("--time", default="01:00:00", help="Walltime per array task (default 01:00:00)")
    ap.add_argument("--gbsa", default="ch2cl2", help="Solvent for xtb4stda -gbsa (default: ch2cl2)")
    ap.add_argument("--emax", type=float, default=20.0, help="Energy window for std2 -e (eV, default: 20.0)")
    ap.add_argument("--chunk-size", type=int, default=50, help="Molecules per array task (default 50)")
    ap.add_argument("--resubmit", action="store_true",
                    help="Include molecules with successful std2 in todo (force re-run)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without calling sbatch")
    args = ap.parse_args()

    part_dir = Path(args.partition_dir).resolve()
    if not part_dir.is_dir():
        sys.exit(f"Not a directory: {part_dir}")

    tmpl_path = Path(args.templates).resolve() / "std2.sh"
    if not tmpl_path.is_file():
        sys.exit(f"Template not found: {tmpl_path}")
    template = load_template(tmpl_path)

    cpus_env = os.environ.get("SLURM_CPUS_PER_TASK")
    cpus_per_task = args.cpus_per_task or (int(cpus_env) if cpus_env else (os.cpu_count() or 1))

    # Build todo list = mol dirs with CREST OK and missing/failed std2 (unless --resubmit)
    mol_dirs = [d for d in sorted(part_dir.iterdir()) if d.is_dir() and MOL_RE.match(d.name)]
    todo = []
    for mol_dir in mol_dirs:
        crest_dir = mol_dir / "crest"
        if not crest_ok(crest_dir):
            continue
        if not (crest_dir / "crest_best.xyz").is_file():
            continue
        std2_dir = mol_dir / "std2"
        ok = std2_run_ok(std2_dir)
        if args.resubmit or not ok:
            todo.append(mol_dir)

    if not todo:
        print(f"No molecules to process in {part_dir} (todo list empty).")
        return

    # Write list + array sbatch into the partition dir
    list_file = part_dir / "std2_todo.txt"
    list_file.write_text("\n".join(str(p) for p in todo) + "\n")

    job_script = part_dir / "std2_array.sbatch"
    write_array_script(
        template, job_script,
        cpus_per_task=cpus_per_task,
        job_name=f"{part_dir.name}_STD2",
        list_file=list_file,
        chunk_size=args.chunk_size,
        gbsa=args.gbsa,
        emax_ev=args.emax,
        walltime=args.time
    )

    # Compute array size
    n = len(todo)
    tasks = (n + args.chunk_size - 1) // args.chunk_size
    array_range = f"0-{tasks-1}"

    # Submission
    if args.dry_run:
        print(f"[DRY] sbatch --array={array_range} {job_script}")
        print(f"  todo: {n} molecules | chunk-size={args.chunk_size} | tasks={tasks}")
        return

    res = subprocess.run(
        ["sbatch", f"--array={array_range}", str(job_script)],
        cwd=part_dir,
        capture_output=True, text=True
    )
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        sys.exit(res.returncode)

    print(res.stdout.strip())
    # Submission manifest
    manifest = {
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "partition_dir": str(part_dir),
        "list_file": str(list_file),
        "job_script": str(job_script),
        "array_range": array_range,
        "cpus_per_task": cpus_per_task,
        "chunk_size": args.chunk_size,
        "gbsa": args.gbsa,
        "emax_ev": args.emax,
        "walltime": args.time,
        "resubmit": bool(args.resubmit),
        "count_molecules": n,
        "tasks": tasks,
    }
    (part_dir / "std2_array_submit.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    """python submit_std2.py --partition_dir compounds/partition_0 \
        --gbsa ch2cl2 --emax 20.0 --chunk-size 50 \
            --cpus-per-task 40 --time 02:00:00"""

    main()

