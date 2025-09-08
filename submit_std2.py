#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import re
import subprocess
from datetime import datetime
import json
import shutil

MOL_RE = re.compile(r"^mol\d+$")  # e.g., mol0001
SUCCESS_BANNER = "CREST terminated normally."

def load_template(path: Path) -> str:
    return path.read_text()

def write_job_script(template_text: str, out_path: Path, *, cpus_per_task: int,
                     job_name: str, xyz_basename: str, gbsa: str,
                     walltime: str, emax_ev: float):
    """Render [[TOKENS]] in the shell template and write sbatch file."""
    walltime_line = f"#SBATCH --time={walltime}" if walltime else ""
    tokens = {
        "[[CPUS_PER_TASK]]": str(cpus_per_task),
        "[[JOB_NAME]]": job_name,
        "[[XYZ_BASENAME]]": xyz_basename,
        "[[GBSA_SOLVENT]]": gbsa,
        "[[WALLTIME_LINE]]": walltime_line,
        "[[EMAX_EV]]": f"{emax_ev:g}",
    }
    script = template_text
    for k, v in tokens.items():
        script = script.replace(k, v)
    out_path.write_text(script)
    out_path.chmod(0o750)

def submit(job_path: Path, workdir: Path, dry_run: bool) -> int:
    """sbatch from inside the molecule's std2/ folder, so $SLURM_SUBMIT_DIR is correct."""
    if dry_run:
        print(f"[DRY] (cd {workdir} && sbatch {job_path.name})")
        return 0
    res = subprocess.run(
        ["sbatch", job_path.name],
        cwd=workdir,
        capture_output=True,
        text=True
    )
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
    else:
        print(res.stdout.strip())
    return res.returncode

def crest_ok(crest_dir: Path) -> bool:
    out = crest_dir / "crest.out"
    if not out.is_file():
        return False
    try:
        return SUCCESS_BANNER in out.read_text(errors="ignore")
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser(
        description="Submit std2 jobs for all CREST-completed molXXXX in a compounds/partition_* directory."
    )
    ap.add_argument("--partition_dir", required=True,
                    help="e.g., compounds/partition_0")
    ap.add_argument("--templates", default="sh_templates",
                    help="Folder with std2.sh template")
    ap.add_argument("--cpus-per-task", type=int, default=40,
                    help="cpus-per-task; default: SLURM_CPUS_PER_TASK or os.cpu_count()")
    ap.add_argument("--time", default="01:00:00", help="Walltime, default 01:00:00")
    ap.add_argument("--gbsa", default="ch2cl2",
                    help="Solvent for xtb4stda -gbsa (default: ch2cl2)")
    ap.add_argument("--emax", type=float, default=10.0,
                    help="Energy window for std2 -e (eV, default: 10.0)")
    ap.add_argument("--resubmit", action="store_true",
                    help="Submit even if std2.out already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions without calling sbatch")
    args = ap.parse_args()

    part_dir = Path(args.partition_dir).resolve()
    if not part_dir.is_dir():
        sys.exit(f"Not a directory: {part_dir}")

    tmpl_dir = Path(args.templates).resolve()
    tmpl_path = tmpl_dir / "std2.sh"
    if not tmpl_path.is_file():
        sys.exit(f"Template not found: {tmpl_path}")
    template = load_template(tmpl_path)

    # Threads
    cpus_env = os.environ.get("SLURM_CPUS_PER_TASK")
    cpus_per_task = args.cpus_per_task or (int(cpus_env) if cpus_env else (os.cpu_count() or 1))

    # Find molXXXX folders
    mol_dirs = [d for d in sorted(part_dir.iterdir()) if d.is_dir() and MOL_RE.match(d.name)]
    if not mol_dirs:
        print(f"No molXXXX directories inside {part_dir}")
        return

    print(f"Submitting std2 for CREST-complete molecules in {part_dir}")
    print(f"  cpus-per-task={cpus_per_task}  gbsa={args.gbsa}  emax={args.emax} eV")

    for mol_dir in mol_dirs:
        mol_name = mol_dir.name
        crest_dir = mol_dir / "crest"
        if not crest_ok(crest_dir):
            print(f"[SKIP] {mol_name}: CREST not successful or crest.out missing")
            continue

        crest_best = crest_dir / "crest_best.xyz"
        if not crest_best.is_file():
            print(f"[SKIP] {mol_name}: crest_best.xyz not found")
            continue

        std2_dir = mol_dir / "std2"
        std2_dir.mkdir(parents=True, exist_ok=True)

        # Copy crest_best.xyz → std2/crest_mol_<mol>.xyz
        xyz_basename = f"crest_mol_{mol_name}.xyz"
        std2_xyz = std2_dir / xyz_basename
        if not std2_xyz.exists() or args.resubmit:
            shutil.copy2(crest_best, std2_xyz)

        # Skip if std2 already done (unless resubmit)
        std2_out = std2_dir / "std2.out"
        if std2_out.is_file() and std2_out.stat().st_size > 0 and not args.resubmit:
            print(f"[SKIP] {mol_name}: std2.out exists (use --resubmit to force)")
            continue

        # Render sbatch
        job_path = std2_dir / "std2.sbatch"
        job_name = f"{mol_name}_STD2"
        write_job_script(
            template,
            job_path,
            cpus_per_task=cpus_per_task,
            job_name=job_name,
            xyz_basename=xyz_basename,
            gbsa=args.gbsa,
            walltime=args.time or "",
            emax_ev=args.emax
        )

        # Submit metadata
        submit_meta = {
            "mol": mol_name,
            "partition": part_dir.name,
            "xyz_source": str(crest_best),
            "xyz_copied_to": str(std2_xyz),
            "job_script": str(job_path),
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "cpus_per_task": cpus_per_task,
            "gbsa": args.gbsa,
            "emax_ev": args.emax,
            "walltime": args.time or "",
        }
        (std2_dir / "submit_metadata.json").write_text(json.dumps(submit_meta, indent=2))

        # Submit from std2/
        rc = submit(job_path, std2_dir, args.dry_run)
        if rc != 0:
            print(f"[WARN] sbatch returned nonzero for {mol_name}")

if __name__ == "__main__":
    """Usage example:
        python submit_std2.py --partition_dir compounds/partition_0\
              --time 01:00:00 --gbsa ch2cl2 --emax 10.0"""
    main()

