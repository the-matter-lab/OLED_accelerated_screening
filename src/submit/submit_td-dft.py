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


def classify_molecule_for_td(
    mol_dir: Path,
    outdir_name: str,
    resubmit_failed: bool,
) -> bool:
    """
    Decide if this molecule should be in the todo list for TD-DFT.

    Rules:
      - if td-dft_wb97x-D3/molXXXX.xyz is missing => skip (cannot run).
      - if td_metadata.json (ORCA TD metadata) missing/unreadable => include (needs run).
      - if td_metadata.json has success == False => include only if resubmit_failed=True.
      - if td_metadata.json has success == True => skip.
    """
    td_dir = mol_dir / outdir_name
    td_xyz = td_dir / f"{mol_dir.name}.xyz"
    if not td_xyz.is_file():
        # TD-DFT prep not run / geometry not available
        return False

    meta_path = td_dir / "td_metadata.json"

    # No TD metadata => treat as not yet run => include
    if not meta_path.is_file():
        return True

    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        # Corrupt / unreadable metadata => safer to re-run
        return True

    success = bool(meta.get("success", False))
    if success:
        # TD-DFT already finished successfully
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
    Create / overwrite the ORCA TD-DFT .in file for this molecule in td-dft_wb97x-D3.
    Returns the path to the .in file.
    """
    mol_name = mol_dir.name
    td_dir = mol_dir / outdir_name
    xyz_path = td_dir / f"{mol_name}.xyz"

    if not xyz_path.is_file():
        raise RuntimeError(f"TD-DFT geometry not found: {xyz_path}")

    lines = xyz_path.read_text().splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"{xyz_path} too short to be valid .xyz")

    xyz_block = "\n".join(lines[2:]) + "\n"

    td_dir.mkdir(parents=True, exist_ok=True)

    in_contents = render_orca_input_template(
        in_template_text,
        xyz_block=xyz_block,
        nprocs=threads_per_job,
        maxcore=maxcore,
        charge=charge,
        mult=mult,
    )
    inp = td_dir / f"{mol_name}.in"
    inp.write_text(in_contents)
    return inp


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Submit ORCA TD-DFT wb97x-D3 CPCM(acetonitrile) calculations from "
            "downselected/molXXXX/td-dft_wb97x-D3/molXXXX.xyz using "
            "node-level parallelism."
        )
    )
    ap.add_argument(
        "--downselection-path",
        required=True,
        help="Directory containing molXXXX subdirs (e.g. downselected)",
    )
    ap.add_argument(
        "--sh-template",
        default="sh_templates/orca_cpcm_dcm_td-dft.sh",
        help="ORCA TD-DFT sh template path",
    )
    ap.add_argument(
        "--outdir-name",
        default="td-dft_wb97x-D3",
        help="Subdirectory name for ORCA TD-DFT inputs/outputs (default: td-dft_wb97x-D3)",
    )
    ap.add_argument(
        "--in-template",
        default="in_templates/orca_cpcm_dcm_td-dft.in",
        help="ORCA TD-DFT input template path",
    )
    ap.add_argument(
        "--threads-per-job",
        type=int,
        default=32,
        help="OpenMP threads / %pal nprocs per ORCA TD-DFT job (default: 32)",
    )
    ap.add_argument(
        "--jobs-per-node",
        type=int,
        default=6,
        help="How many ORCA TD-DFT jobs to run in parallel on a 192-core node (default: 6)",
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
            "Also include molecules whose td_metadata.json exists and has success=false. "
            "For those, any existing .out is renamed to .out_failed_YYYYMMDD_HHMMSS "
            "and a fresh TD-DFT run is submitted."
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

    sh_template_path = Path(args.sh_template).resolve()
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

    # Build todo list based on td-dft_wb97x-D3 contents and td_metadata.json there
    todo = []
    for mol_dir in mol_dirs_all:
        if classify_molecule_for_td(
            mol_dir,
            outdir_name=args.outdir_name,
            resubmit_failed=args.resubmit_failed,
        ):
            todo.append(mol_dir)

    if not todo:
        print(
            f"No molecules to submit TD-DFT under {root} "
            f"(resubmit_failed={args.resubmit_failed})."
        )
        return

    print(
        f"Preparing ORCA TD-DFT for {len(todo)} molecules under {root}\n"
        f"  td dir        = '{args.outdir_name}'\n"
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
        td_dir = mol_dir / args.outdir_name
        mol_name = mol_dir.name

        # For resubmission, if there is an old .out, rename it so the shell script
        # does not skip the molecule
        if args.resubmit_failed and td_dir.is_dir():
            out_path = td_dir / f"{mol_name}.out"
            if out_path.is_file():
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_out = td_dir / f"{mol_name}.out_failed_{ts}"
                try:
                    out_path.rename(backup_out)
                    print(f"[INFO] {mol_name}: moved {out_path.name} -> {backup_out.name}")
                except Exception as e:
                    print(f"[WARN] {mol_name}: could not rename {out_path.name}: {e}")

        # Create / overwrite .in from td-dft_wb97x-D3/molXXXX.xyz
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
        f"each with up to {jobs_per_node} ORCA TD-DFT tasks in parallel."
    )

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{idx:03d}"
        list_file = root / f"orca_td_chunk_{chunk_id}.txt"
        list_file.write_text(
            "\n".join(str(m.resolve()) for m in chunk) + "\n"
        )

        job_name = f"orca_td_{chunk_id}"
        sbatch_path = root / f"orca_td_chunk_{chunk_id}.sbatch"
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
        (root / f"orca_td_chunk_{chunk_id}_submit.json").write_text(
            json.dumps(manifest, indent=2)
        )


if __name__ == "__main__":
    """
    Example:

    python src/submit/submit_td-dft.py --downselection-path downselected \
        --threads-per-job 32 \
        --sh-template sh_templates/orca_cpcm_dcm_td-dft_safe.sh \
        --jobs-per-node 6 \
        --time 08:00:00 \
        --charge 0 --mult 1

    To resubmit only failed TD-DFT runs (success == false in td_metadata.json):


    python src/submit/submit_td-dft.py --downselection-path downselected \
        --threads-per-job 32 \
        --sh-template sh_templates/orca_cpcm_dcm_td-dft_safe.sh \
        --jobs-per-node 6 \
        --time 08:00:00 \
        --charge 0 --mult 1 \
        --resubmit-failed
    
    sh_templates/orca_cpcm_dcm_td-dft_safe.sh includes export UCX_VFS_ENABLE=n
    which may help avoid segmentation faults in UCX VFS (MPI runtime).
    """
    main()
