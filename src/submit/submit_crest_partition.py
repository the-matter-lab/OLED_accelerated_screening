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

def load_template(path: Path) -> str:
    return path.read_text()

def write_job_script(template_text: str, out_path: Path, *, cpus_per_task: int,
                     job_name: str, xyz_src_rel: str, xyz_basename: str,
                     gbsa: str, walltime: str):
    """Render [[TOKENS]] in the shell template and write sbatch file."""
    walltime_line = f"#SBATCH --time={walltime}" if walltime else ""
    tokens = {
        "[[CPUS_PER_TASK]]": str(cpus_per_task),
        "[[JOB_NAME]]": job_name,
        "[[XYZ_SRC_REL]]": xyz_src_rel,
        "[[XYZ_BASENAME]]": xyz_basename,
        "[[GBSA_SOLVENT]]": gbsa,
        "[[WALLTIME_LINE]]": walltime_line,
    }
    script = template_text
    for k, v in tokens.items():
        script = script.replace(k, v)
    out_path.write_text(script)
    out_path.chmod(0o750)

def submit(job_path: Path, workdir: Path, dry_run: bool) -> int:
    """sbatch from inside the molecule's crest/ folder, so $SLURM_SUBMIT_DIR is correct."""
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

def main():
    ap = argparse.ArgumentParser(
        description="Submit CREST jobs for all molXXXX in a compounds/partition_* directory."
    )
    ap.add_argument("--partition_dir", required=True,
                    help="e.g., compounds/partition_0")
    ap.add_argument("--templates", default="sh_templates",
                    help="Folder with crest.sh and optional constraints.inp")
    ap.add_argument("--cpus-per-task", type=int, default=None,
                    help="cpus-per-task; default: SLURM_CPUS_PER_TASK or os.cpu_count()")
    ap.add_argument("--time", default=None, help="Walltime, e.g., 24:00:00")
    ap.add_argument("--gbsa", default="chloroform",
                    help="Solvent for --gbsa (string passed to CREST's GBSA)")
    ap.add_argument("--resubmit", action="store_true",
                    help="Submit even if crest.out already shows success")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions without calling sbatch")
    args = ap.parse_args()

    part_dir = Path(args.partition_dir).resolve()
    if not part_dir.is_dir():
        sys.exit(f"Not a directory: {part_dir}")

    tmpl_dir = Path(args.templates).resolve()
    tmpl_path = tmpl_dir / "crest.sh"
    if not tmpl_path.is_file():
        sys.exit(f"Template not found: {tmpl_path}")

    template = load_template(tmpl_path)

    constraints_src = tmpl_dir / "constraints.inp"
    have_constraints = constraints_src.is_file()

    # Threads
    cpus_env = os.environ.get("SLURM_CPUS_PER_TASK")
    cpus_per_task = args.cpus_per_task or (int(cpus_env) if cpus_env else (os.cpu_count() or 1))

    # Find molXXXX folders
    mol_dirs = [d for d in sorted(part_dir.iterdir()) if d.is_dir() and MOL_RE.match(d.name)]
    if not mol_dirs:
        print(f"No molXXXX directories inside {part_dir}")
        return

    print(f"Submitting {len(mol_dirs)} jobs from {part_dir} with cpus-per-task={cpus_per_task}, gbsa={args.gbsa}")
    for mol_dir in mol_dirs:
        mol_name = mol_dir.name
        obabel_xyz = mol_dir / "obabel" / f"{mol_name}.xyz"
        if not obabel_xyz.is_file():
            print(f"[SKIP] {mol_name}: missing {obabel_xyz}")
            continue

        crest_dir = mol_dir / "crest"
        crest_dir.mkdir(parents=True, exist_ok=True)

        # constraints (optional)
        if have_constraints:
            dst = crest_dir / "constraints.inp"
            if not dst.exists():
                dst.write_text(constraints_src.read_text())

        # skip if already successful
        crest_out = crest_dir / "crest.out"
        if crest_out.is_file() and not args.resubmit:
            try:
                if "CREST terminated normally." in crest_out.read_text(errors="ignore"):
                    print(f"[SKIP] {mol_name}: already finished successfully")
                    continue
            except Exception:
                pass

        # render job script
        job_path = crest_dir / "crest.sbatch"
        write_job_script(
            template,
            job_path,
            cpus_per_task=cpus_per_task,
            job_name=f"{mol_name}_CR",
            xyz_src_rel=f"../obabel/{mol_name}.xyz",
            xyz_basename=f"{mol_name}.xyz",
            gbsa=args.gbsa,
            walltime=args.time or "",
        )

        # submit metadata
        submit_meta = {
            "mol": mol_name,
            "partition": part_dir.name,
            "xyz_source": str(obabel_xyz),
            "job_script": str(job_path),
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "cpus_per_task": cpus_per_task,
            "gbsa": args.gbsa,
            "walltime": args.time or "",
        }
        (crest_dir / "submit_metadata.json").write_text(json.dumps(submit_meta, indent=2))

        # submit from crest/
        rc = submit(job_path, crest_dir, args.dry_run)
        if rc != 0:
            print(f"[WARN] sbatch returned nonzero for {mol_name}")

if __name__ == "__main__":
    main()

