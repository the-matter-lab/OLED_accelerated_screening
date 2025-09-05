#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import re
import subprocess
from datetime import datetime
import json

MOL_RE = re.compile(r"^mol\d+$")  # mol0001, mol0096, ...

def load_template(path: Path) -> str:
    return path.read_text()

def write_job_script(template: str, out_path: Path, **vars_):
    # minimal {PLACEHOLDER} templating
    script = template.format_map(vars_)
    out_path.write_text(script)
    out_path.chmod(0o750)

def submit(job_path: Path, dry_run: bool):
    if dry_run:
        print(f"[DRY] sbatch {job_path}")
        return 0
    res = subprocess.run(["sbatch", str(job_path)], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
    else:
        print(res.stdout.strip())
    return res.returncode

def main():
    ap = argparse.ArgumentParser(description="Submit one CREST job per molecule folder.")
    ap.add_argument("--root", default="compounds", help="Root folder with molXXXX subfolders")
    ap.add_argument("--templates", default="sh_templates", help="Folder containing crest.sh and optional constraints.inp")
    ap.add_argument("--job-name-suffix", default="_CR", help="Suffix for the SLURM job name")
    ap.add_argument("--ntasks", type=int, default=None, help="Threads if SLURM_CPUS_PER_TASK unset (e.g., 40)")
    ap.add_argument("--time", default=None, help="Override walltime, e.g., 24:00:00")
    ap.add_argument("--gbsa", default="dichloromethane", help="Solvent for CREST GBSA (default: dichloromethane)")
    ap.add_argument("--resubmit", action="store_true", help="Submit even if crest.out already exists")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without submitting")
    ap.add_argument("--start", type=int, default=None,
                    help="First mol index to process (1-based, inclusive)")
    ap.add_argument("--end", type=int, default=None,
                    help="Last mol index to process (1-based, inclusive)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    templates_dir = Path(args.templates).resolve()
    crest_template_path = templates_dir / "crest.sh"

    if not crest_template_path.is_file():
        sys.exit(f"ERROR: template not found: {crest_template_path}")

    crest_template = load_template(crest_template_path)

    constraints_src = templates_dir / "constraints.inp"
    have_constraints = constraints_src.is_file()

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    threads = int(slurm_cpus) if slurm_cpus else (args.ntasks or os.cpu_count() or 1)

    mol_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and MOL_RE.match(d.name)]
    if not mol_dirs:
        print(f"No molXXXX directories under {root}")
        return
    
    # apply slicing to comply with maximum jobs in cluster
    if args.start or args.end:
        selected = []
        for d in mol_dirs:
            idx = int(d.name.replace("mol", ""))
            if args.start and idx < args.start:
                continue
            if args.end and idx > args.end:
                continue
            selected.append(d)
        mol_dirs = selected
    

    print(f"Found {len(mol_dirs)} molecules under {root}")
    for mol_dir in mol_dirs:
        mol_name = mol_dir.name
        obabel_xyz = mol_dir / "obabel" / f"{mol_name}.xyz"
        if not obabel_xyz.is_file():
            print(f"[SKIP] {mol_name}: missing {obabel_xyz}")
            continue

        crest_dir = mol_dir / "crest"
        crest_dir.mkdir(parents=True, exist_ok=True)

        # copy constraints.inp if available
        if have_constraints:
            dst_constraints = crest_dir / "constraints.inp"
            if not dst_constraints.exists():
                dst_constraints.write_text(constraints_src.read_text())

        best_geom_dir = mol_dir / "best_geom"
        best_geom_dir.mkdir(exist_ok=True)

        crest_out = crest_dir / "crest.out"
        if crest_out.is_file() and not args.resubmit:
            print(f"[SKIP] {mol_name}: crest.out exists (use --resubmit to force)")
            continue

        job_path = crest_dir / "crest.sbatch"
        time_override = args.time or ""  # if empty, template keeps its own --time

        vars_ = {
            "JOB_NAME": f"{mol_name}{args.job_name_suffix}",
            "NTASKS": threads,
            "XYZ_BASENAME": f"{mol_name}.xyz",
            "XYZ_SRC_REL": f"../obabel/{mol_name}.xyz",
            "BEST_GEOM_DST": f"../best_geom/{mol_name}.xyz",
            "WALLTIME_OVERRIDE": time_override,   # template handles empty vs set
            "GBSA_SOLVENT": args.gbsa,
        }

        write_job_script(crest_template, job_path, **vars_)

        # lightweight submit metadata
        meta_submit = {
            "mol": mol_name,
            "xyz_source": str(obabel_xyz),
            "job_script": str(job_path),
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "threads": threads,
            "gbsa": args.gbsa,
        }
        (crest_dir / "submit_metadata.json").write_text(json.dumps(meta_submit, indent=2))

        submit(job_path, args.dry_run)

if __name__ == "__main__":
    """Example usage:
	python run_crest.py --root compounds --templates sh_templates --start 1 --end 999"""
    main()

