#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import re
import subprocess
import json
from datetime import datetime

MOL_DIR_RE = re.compile(r"^mol\d+$")  # e.g. mol0001
DEFAULT_ACCOUNT = "rrg-aspuru"       # metadata only


def render_orca_input_template(
    tmpl: str,
    *,
    xyz_block: str,
    nprocs: int,
    maxcore: int,
    charge: int,
    mult: int,
) -> str:
    rep = {
        "[[NPROCS]]": str(nprocs),
        "[[MAXCORE]]": str(maxcore),
        "[[CHARGE]]": str(charge),
        "[[MULT]]": str(mult),
        "[[XYZ_BLOCK]]": xyz_block,
    }
    out = tmpl
    for k, v in rep.items():
        out = out.replace(k, v)
    return out


def write_job_script(
    sh_template: str,
    out_path: Path,
    *,
    list_file: Path,
    outdir_name: str,
    jobs_per_node: int,
    threads_per_job: int,
    walltime: str,
    job_name: str,
) -> None:
    """Render the node-level ORCA launcher for one chunk of molecules."""
    walltime_line = f"#SBATCH --time={walltime}" if walltime else ""
    rep = {
        "[[LIST_FILE]]": str(list_file),
        "[[OUTDIR_NAME]]": outdir_name,
        "[[JOBS_PER_NODE]]": str(jobs_per_node),
        "[[THREADS_PER_JOB]]": str(threads_per_job),
        "[[WALLTIME_LINE]]": walltime_line,
        "[[JOB_NAME]]": job_name,
    }
    script = sh_template
    for k, v in rep.items():
        script = script.replace(k, v)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script)
    out_path.chmod(0o750)


def submit(job_path: Path, workdir: Path, dry_run: bool) -> int:
    if dry_run:
        print(f"[DRY] (cd {workdir} && sbatch {job_path.name})")
        return 0

    res = subprocess.run(
        ["sbatch", job_path.name],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
    else:
        print(res.stdout.strip())
    return res.returncode


def classify_molecule_for_reopt(
    mol_dir: Path,
    outdir_name: str,
    resubmit_failed: bool,
) -> bool:
    """
    Decide if this molecule should be in the todo list for REOPT.

    Rules:
      - if reopt_b97_3c/molXXXX_reopt.xyz is missing => skip (cannot run).
      - if metadata.json (ORCA reopt metadata) missing/unreadable => include (needs run).
      - if metadata.json has success == False => include only if resubmit_failed=True.
      - if metadata.json has success == True => skip.
    """
    outdir = mol_dir / outdir_name
    reopt_xyz = outdir / f"{mol_dir.name}_reopt.xyz"
    if not reopt_xyz.is_file():
        return False

    meta_path = outdir / "metadata.json"

    # No ORCA metadata => treat as not yet run => include
    if not meta_path.is_file():
        return True

    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        # Corrupt / unreadable metadata => safer to re-run
        return True

    success = bool(meta.get("success", False))
    if success:
        # Already finished successfully
        return False

    # Failed before: only include if user asked to resubmit failed
    return resubmit_failed


def prepare_orca_input_for_mol(
    mol_dir: Path,
    outdir_name: str,
    in_template_text: str,
    *,
    threads_per_job: int,
    maxcore: int,
    charge: int,
    mult: int,
) -> Path:
    """
    Create / overwrite the ORCA .in file for this molecule in reopt_b97_3c.
    Returns the path to the .in file.
    """
    mol_name = mol_dir.name
    outdir = mol_dir / outdir_name
    xyz_path = outdir / f"{mol_name}_reopt.xyz"

    if not xyz_path.is_file():
        raise RuntimeError(f"Reopt geometry not found: {xyz_path}")

    lines = xyz_path.read_text().splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"{xyz_path} too short to be valid .xyz")

    # Standard xyz: first two lines header, remainder coordinates
    xyz_block = "\n".join(lines[2:]) + "\n"

    outdir.mkdir(parents=True, exist_ok=True)

    in_contents = render_orca_input_template(
        in_template_text,
        xyz_block=xyz_block,
        nprocs=threads_per_job,
        maxcore=maxcore,
        charge=charge,
        mult=mult,
    )
    inp = outdir / f"{mol_name}.in"
    inp.write_text(in_contents)
    return inp


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Submit ORCA B97-3c CPCM(dcm) geometry reoptimizations from "
            "downselected/molXXXX/reopt_b97_3c/molXXXX_reopt.xyz using "
            "node-level parallelism."
        )
    )
    ap.add_argument(
        "--downselection-path",
        required=True,
        help="Directory containing molXXXX subdirs (e.g. downselected)",
    )
    ap.add_argument(
        "--templates",
        default="sh_templates",
        help="Folder with orca_cpcm_dcm_reopt.sh template (default: sh_templates)",
    )
    ap.add_argument(
        "--outdir-name",
        default="reopt_b97_3c",
        help="Subdirectory name for ORCA reopt inputs/outputs (default: reopt_b97_3c)",
    )
    ap.add_argument(
        "--in-template",
        default="in_templates/orca_cpcm_dcm_opt_slow.in",
        help="ORCA input template path for reopt (default: in_templates/orca_cpcm_dcm_reopt.in)",
    )
    ap.add_argument(
        "--threads-per-job",
        type=int,
        default=32,
        help="OpenMP threads / %pal nprocs per ORCA reopt job (default: 32)",
    )
    ap.add_argument(
        "--jobs-per-node",
        type=int,
        default=6,
        help="How many ORCA reopt jobs to run in parallel on a 192-core node (default: 6)",
    )
    ap.add_argument("--time", default=None, help="Walltime, e.g. 08:00:00")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument(
        "--maxcore",
        type=int,
        default=2500,
        help="%maxcore in MB per core (default: 2500)",
    )
    ap.add_argument(
        "--resubmit-failed",
        action="store_true",
        help=(
            "Also include molecules whose reopt metadata.json exists and has success=false. "
            "For those, an existing .out is renamed to .out_failed_YYYYMMDD_HHMMSS "
            "and a fresh reopt is submitted."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    root = Path(args.downselection_path).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    in_template_path = Path(args.in_template).resolve()
    if not in_template_path.is_file():
        sys.exit(f"ORCA input template not found: {in_template_path}")
    in_template_text = in_template_path.read_text()

    tmpl_dir = Path(args.templates).resolve()
    sh_template_path = tmpl_dir / "orca_cpcm_dcm_reopt.sh"
    if not sh_template_path.is_file():
        sys.exit(f"Shell template not found: {sh_template_path}")
    sh_template_text = sh_template_path.read_text()

    # Discover molXXXX directories
    mol_dirs_all = [
        p for p in sorted(root.iterdir())
        if p.is_dir() and MOL_DIR_RE.match(p.name)
    ]

    if not mol_dirs_all:
        print(f"No molXXXX directories found under {root}")
        return

    # Build todo list based on reopt_b97_3c contents and metadata.json there
    todo = []
    for mol_dir in mol_dirs_all:
        if classify_molecule_for_reopt(
            mol_dir,
            outdir_name=args.outdir_name,
            resubmit_failed=args.resubmit_failed,
        ):
            todo.append(mol_dir)

    if not todo:
        print(
            f"No molecules to submit for reopt under {root} "
            f"(resubmit_failed={args.resubmit_failed})."
        )
        return

    print(
        f"Preparing ORCA reopt for {len(todo)} molecules under {root}\n"
        f"  reopt dir     = '{args.outdir_name}'\n"
        f"  template      = {in_template_path}\n"
        f"  threads/job   = {args.threads_per_job}\n"
        f"  jobs/node     = {args.jobs_per_node}\n"
        f"  charge/mult   = {args.charge}/{args.mult}\n"
        f"  maxcore       = {args.maxcore} MB\n"
        f"  account       = {DEFAULT_ACCOUNT}\n"
        f"  resubmit_failed = {args.resubmit_failed}"
    )

    # Ensure inputs exist for all todo molecules
    final_todo = []
    for mol_dir in todo:
        outdir = mol_dir / args.outdir_name

        # For resubmission, if there is an old .out, rename it so the shell script
        # does not skip the molecule
        if args.resubmit_failed and outdir.is_dir():
            out_path = outdir / f"{mol_dir.name}.out"
            if out_path.is_file():
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_out = outdir / f"{mol_dir.name}.out_failed_{ts}"
                try:
                    out_path.rename(backup_out)
                    print(f"[INFO] {mol_dir.name}: moved {out_path.name} -> {backup_out.name}")
                except Exception as e:
                    print(f"[WARN] {mol_dir.name}: could not rename {out_path.name}: {e}")

        # Create / overwrite .in from molXXXX_reopt.xyz
        try:
            prepare_orca_input_for_mol(
                mol_dir,
                outdir_name=args.outdir_name,
                in_template_text=in_template_text,
                threads_per_job=args.threads_per_job,
                maxcore=args.maxcore,
                charge=args.charge,
                mult=args.mult,
            )
        except Exception as e:
            print(f"[WARN] Skipping {mol_dir.name} due to input error: {e}")
            continue

        inp = (mol_dir / args.outdir_name) / f"{mol_dir.name}.in"
        if inp.is_file():
            final_todo.append(mol_dir)
        else:
            print(f"[SKIP] {mol_dir.name}: input file missing after preparation")

    if not final_todo:
        print("No valid molecules left to submit (all had input errors).")
        return

    # Chunk into groups of jobs_per_node
    jobs_per_node = max(1, args.jobs_per_node)
    chunks = [
        final_todo[i: i + jobs_per_node]
        for i in range(0, len(final_todo), jobs_per_node)
    ]

    print(
        f"Submitting {len(chunks)} node-jobs, "
        f"each with up to {jobs_per_node} ORCA reopt tasks in parallel."
    )

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{idx:03d}"
        list_file = root / f"orca_reopt_chunk_{chunk_id}.txt"
        list_file.write_text(
            "\n".join(str(m.resolve()) for m in chunk) + "\n"
        )

        job_name = f"orca_reopt_{chunk_id}"
        sbatch_path = root / f"orca_reopt_chunk_{chunk_id}.sbatch"
        write_job_script(
            sh_template_text,
            sbatch_path,
            list_file=list_file,
            outdir_name=args.outdir_name,
            jobs_per_node=jobs_per_node,
            threads_per_job=args.threads_per_job,
            walltime=args.time or "",
            job_name=job_name,
        )

        rc = submit(sbatch_path, root, args.dry_run)
        if rc != 0:
            print(f"[WARN] sbatch returned non-zero for chunk {chunk_id}")

        manifest = {
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "downselection_path": str(root),
            "chunk_id": chunk_id,
            "list_file": str(list_file),
            "job_script": str(sbatch_path),
            "threads_per_job": args.threads_per_job,
            "jobs_per_node": jobs_per_node,
            "walltime": args.time or "",
            "charge": args.charge,
            "mult": args.mult,
            "maxcore_mb": args.maxcore,
            "account": DEFAULT_ACCOUNT,
            "molecules": [m.name for m in chunk],
            "resubmit_failed": bool(args.resubmit_failed),
        }
        (root / f"orca_reopt_chunk_{chunk_id}_submit.json").write_text(
            json.dumps(manifest, indent=2)
        )


if __name__ == "__main__":
    """
    Example:

    python src/submit/submit_orca_reopt.py --downselection-path downselected \
        --threads-per-job 32 \
        --jobs-per-node 6 \
        --time 08:00:00 \
        --charge 0 --mult 1

    To resubmit only failed reopts (success == false in reopt_b97_3c/metadata.json):

    python src/submit/submit_orca_reopt.py --downselection-path downselected \
        --threads-per-job 32 \
        --jobs-per-node 6 \
        --time 08:00:00 \
        --charge 0 --mult 1 \
        --resubmit-failed
    """
    main()
