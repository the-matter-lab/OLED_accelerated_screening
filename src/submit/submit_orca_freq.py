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


def opt_succeeded(opt_dir: Path) -> bool:
    """
    Check if the geometry optimisation in opt_dir succeeded,
    based on opt_dir/metadata.json written by the optimisation step.
    """
    meta_path = opt_dir / "metadata.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    return bool(meta.get("success", False))


def classify_molecule_for_freq(
    mol_dir: Path,
    opt_outdir_name: str,
    freq_outdir_name: str,
    resubmit_failed: bool,
) -> bool:
    """
    Decide if this molecule should be in the todo list for frequency.

    Rules:
      - If opt metadata missing or opt NOT successful => skip.
      - If freq/metadata.json missing => include (needs first run).
      - If freq/metadata.json success == True => skip (already done).
      - If freq/metadata.json success == False => include only if resubmit_failed=True.
    """
    opt_dir = mol_dir / opt_outdir_name
    if not opt_succeeded(opt_dir):
        # Either no metadata, or opt failed.
        return False

    freq_dir = mol_dir / freq_outdir_name
    meta_path = freq_dir / "metadata.json"

    # No metadata => treat as not yet run => include
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


def choose_opt_xyz(opt_dir: Path, mol_name: str) -> Path:
    """
    Choose the geometry .xyz file for the frequency calculation from the opt directory.

    Strategy:
      1. Prefer a file named {mol_name}.xyz if present.
      2. Otherwise, if there is exactly one .xyz file, use it.
      3. Otherwise, raise a RuntimeError.
    """
    cand_named = opt_dir / f"{mol_name}.xyz"
    if cand_named.is_file():
        return cand_named

    xyzs = sorted(opt_dir.glob("*.xyz"))
    if len(xyzs) == 1:
        return xyzs[0]

    raise RuntimeError(
        f"Cannot uniquely determine optimised geometry .xyz in {opt_dir} "
        f"(found {[p.name for p in xyzs]})"
    )


def prepare_orca_input_for_mol(
    mol_dir: Path,
    opt_outdir_name: str,
    freq_outdir_name: str,
    in_template_text: str,
    *,
    threads_per_job: int,
    maxcore: int,
    charge: int,
    mult: int,
) -> Path:
    """
    Create/overwrite the ORCA frequency .in file for this molecule.
    Returns the path to the .in file.
    """
    mol_name = mol_dir.name
    opt_dir = mol_dir / opt_outdir_name
    freq_dir = mol_dir / freq_outdir_name

    # Choose geometry from opt directory
    xyz_path = choose_opt_xyz(opt_dir, mol_name)

    lines = xyz_path.read_text().splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"{xyz_path} too short to be valid .xyz")

    xyz_block = "\n".join(lines[2:]) + "\n"

    freq_dir.mkdir(parents=True, exist_ok=True)

    in_contents = render_orca_input_template(
        in_template_text,
        xyz_block=xyz_block,
        nprocs=threads_per_job,
        maxcore=maxcore,
        charge=charge,
        mult=mult,
    )
    inp = freq_dir / f"{mol_name}.in"
    inp.write_text(in_contents)
    return inp


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Submit ORCA B97-3c CPCM(dcm) frequency calculations from "
            "downselected/molXXXX/opt_b97_3c/*.xyz using node-level parallelism."
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
        help="Folder with orca_cpcm_dcm_freq.sh template (default: sh_templates)",
    )
    ap.add_argument(
        "--opt-outdir-name",
        default="opt_b97_3c",
        help="Directory name with ORCA optimisation results (default: opt_b97_3c)",
    )
    ap.add_argument(
        "--outdir-name",
        default="freq_b97_3c",
        help="Subdirectory name for ORCA freq inputs/outputs (default: freq_b97_3c)",
    )
    ap.add_argument(
        "--in-template",
        default="in_templates/orca_cpcm_dcm_freq.in",
        help="ORCA input template path",
    )
    ap.add_argument(
        "--threads-per-job",
        type=int,
        default=32,
        help="MPI ranks / threads per ORCA freq job (default: 32)",
    )
    ap.add_argument(
        "--jobs-per-node",
        type=int,
        default=6,
        help="How many ORCA freq jobs to run in parallel on a 192-core node (default: 6)",
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
            "Also include molecules whose freq metadata.json exists and has success=false. "
            "For those, the existing freq_b97_3c directory is renamed to "
            "freq_b97_3c_failed_YYYYMMDD_HHMMSS and a fresh freq_b97_3c is created."
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
    sh_template_path = tmpl_dir / "orca_cpcm_dcm_freq.sh"
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

    # Build todo list based on opt metadata and freq metadata
    todo = []
    for mol_dir in mol_dirs_all:
        if classify_molecule_for_freq(
            mol_dir,
            opt_outdir_name=args.opt_outdir_name,
            freq_outdir_name=args.outdir_name,
            resubmit_failed=args.resubmit_failed,
        ):
            todo.append(mol_dir)

    if not todo:
        print(
            f"No molecules to submit freq under {root} "
            f"(resubmit_failed={args.resubmit_failed})."
        )
        return

    print(
        f"Preparing ORCA freq for {len(todo)} molecules under {root}\n"
        f"  opt dir       = '{args.opt_outdir_name}'\n"
        f"  freq dir      = '{args.outdir_name}'\n"
        f"  template      = {in_template_path}\n"
        f"  threads/job   = {args.threads_per_job}\n"
        f"  jobs/node     = {args.jobs_per_node}\n"
        f"  charge/mult   = {args.charge}/{args.mult}\n"
        f"  maxcore       = {args.maxcore} MB\n"
        f"  account       = {DEFAULT_ACCOUNT}\n"
        f"  resubmit_failed = {args.resubmit_failed}"
    )

    # Ensure inputs exist for all todo molecules; handle resubmit_failed renaming
    prepared = []
    for mol_dir in todo:
        mol_name = mol_dir.name
        freq_dir = mol_dir / args.outdir_name
        failed_run = False

        if args.resubmit_failed and freq_dir.is_dir():
            # Look at metadata to decide if this is really a previously failed run
            meta_path = freq_dir / "metadata.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                    success = bool(meta.get("success", False))
                    failed_run = not success
                except Exception:
                    # unreadable metadata: treat as failed to be safe
                    failed_run = True

            # If previously failed, rename the old directory
            if failed_run:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = mol_dir / f"{args.outdir_name}_failed_{ts}"
                try:
                    freq_dir.rename(backup)
                    print(f"[INFO] {mol_name}: moved {args.outdir_name} -> {backup.name}")
                except Exception as e:
                    print(f"[WARN] {mol_name}: could not rename {freq_dir.name}: {e}")
                    # Skip this molecule to avoid mixing half-renamed state
                    continue

        # Now (re)create a clean freq dir and .in file
        try:
            prepare_orca_input_for_mol(
                mol_dir,
                opt_outdir_name=args.opt_outdir_name,
                freq_outdir_name=args.outdir_name,
                in_template_text=in_template_text,
                threads_per_job=args.threads_per_job,
                maxcore=args.maxcore,
                charge=args.charge,
                mult=args.mult,
            )
            # Only keep if .in exists
            inp = mol_dir / args.outdir_name / f"{mol_name}.in"
            if inp.is_file():
                prepared.append(mol_dir)
            else:
                print(f"[SKIP] {mol_name}: input file missing after preparation")
        except Exception as e:
            print(f"[WARN] Skipping {mol_name} due to input error: {e}")
            continue

    if not prepared:
        print("No valid molecules left to submit (all had input errors or rename issues).")
        return

    # Chunk into groups of jobs_per_node
    jobs_per_node = max(1, args.jobs_per_node)
    chunks = [
        prepared[i: i + jobs_per_node]
        for i in range(0, len(prepared), jobs_per_node)
    ]

    print(
        f"Submitting {len(chunks)} node-jobs, "
        f"each with up to {jobs_per_node} ORCA freq tasks in parallel."
    )

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{idx:03d}"
        list_file = root / f"orca_freq_chunk_{chunk_id}.txt"
        list_file.write_text(
            "\n".join(str(m.resolve()) for m in chunk) + "\n"
        )

        job_name = f"orca_freq_{chunk_id}"
        sbatch_path = root / f"orca_freq_chunk_{chunk_id}.sbatch"
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

        # Submission
        rc = submit(sbatch_path, root, args.dry_run)
        if rc != 0:
            print(f"[WARN] sbatch returned non-zero for chunk {chunk_id}")

        # Submission manifest for this chunk
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
        (root / f"orca_freq_chunk_{chunk_id}_submit.json").write_text(
            json.dumps(manifest, indent=2)
        )


if __name__ == "__main__":
    """
    Example:

    python src/submit/submit_orca_freq.py --downselection-path downselected \
        --threads-per-job 32 \
        --jobs-per-node 6 \
        --time 08:00:00 \
        --charge 0 --mult 1

    To resubmit only failed freq molecules (success == false in metadata.json),
    renaming old freq_b97_3c -> freq_b97_3c_failed_YYYYMMDD_HHMMSS:

    python src/submit/submit_orca_freq.py --downselection-path downselected \
        --threads-per-job 48 \
        --jobs-per-node 4 \
        --time 08:00:00 \
        --charge 0 --mult 1 \
        --resubmit-failed
    """
    main()
