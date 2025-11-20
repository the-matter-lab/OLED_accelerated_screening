#!/bin/bash
#SBATCH --job-name=[[JOB_NAME]]
#SBATCH --account=rrg-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
[[WALLTIME_LINE]]
#SBATCH -p compute

set -euo pipefail

echo "Nodes:"
cat "$SLURM_JOB_NODELIST" || true
cd "$SLURM_SUBMIT_DIR"

LIST_FILE="[[LIST_FILE]]"
OUTDIR_NAME="[[OUTDIR_NAME]]"
JOBS_PER_NODE=[[JOBS_PER_NODE]]

if [ ! -s "$LIST_FILE" ]; then
  echo "ERROR: List file not found or empty: $LIST_FILE" >&2
  exit 2
fi

# Threads per ORCA calculation (must match %pal nprocs)
export OMP_NUM_THREADS=[[THREADS_PER_JOB]]
export MKL_NUM_THREADS=[[THREADS_PER_JOB]]
ulimit -s unlimited
export OMP_STACKSIZE=512M

# Trillium ORCA environment
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5
module load orca/6.1.0

if [ -z "${EBROOTORCA:-}" ] || [ ! -x "$EBROOTORCA/orca" ]; then
  echo "ERROR: EBROOTORCA/orca not found; ORCA module not loaded correctly?" >&2
  exit 1
fi

set +e
# LIST_FILE contains absolute molXXXX directories
cat "$LIST_FILE" | parallel -j "$JOBS_PER_NODE" '
  MOL_DIR="{}"
  MOL_NAME=$(basename "$MOL_DIR")
  OUTDIR="$MOL_DIR/'"$OUTDIR_NAME"'"
  INFILE="$OUTDIR/${MOL_NAME}.in"
  OUTFILE="$OUTDIR/${MOL_NAME}.out"

  if [ ! -d "$OUTDIR" ]; then
    echo "[SKIP] $MOL_NAME: missing outdir $OUTDIR"
    exit 0
  fi

  if [ ! -e "$INFILE" ]; then
    echo "[SKIP] $MOL_NAME: missing input $INFILE"
    exit 0
  fi

  if [ -e "$OUTFILE" ]; then
    echo "[SKIP] $MOL_NAME: $OUTFILE already exists"
    exit 0
  fi

  echo "Running ORCA TD-DFT on $INFILE in $OUTDIR (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
  cd "$OUTDIR" || exit 1

  # Run ORCA
  "$EBROOTORCA/orca" "$INFILE" > "${MOL_NAME}.log" 2>&1
  rc=$?
  mv -f "${MOL_NAME}.log" "$OUTFILE"
  echo "${MOL_NAME}.out rc=$rc" >> calculated_file
  echo "Finished $MOL_NAME with rc=$rc"

  # Per-calculation metadata (TD-DFT)
  date_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  python3 - <<PY
import json, re, pathlib

out_path = pathlib.Path(r"$OUTFILE")
date = "$date_iso"
calculation_type = "td-dft"
solvent_model = "cpcm"
solvent = "acetonitrile"

success = False
calculation_time = 0

if out_path.is_file():
    txt = out_path.read_text(errors="ignore")

    # success if banner present
    if "ORCA TERMINATED NORMALLY" in txt:
        success = True

    # TOTAL RUN TIME: 0 days 0 hours 5 minutes 5 seconds 801 msec
    m = re.search(
        r"TOTAL\\s+RUN\\s+TIME\\s*:\\s*(\\d+)\\s+days?\\s+(\\d+)\\s+hours?\\s+"
        r"(\\d+)\\s+minutes?\\s+(\\d+)\\s+seconds?",
        txt,
        re.I,
    )
    if m:
        d, h, mi, s = map(int, m.groups())
        calculation_time = d*86400 + h*3600 + mi*60 + s

meta = {
    "date": date,
    "calculation_type": calculation_type,
    "solvent_model": solvent_model,
    "solvent": solvent,
    "success": success,
    "calculation_time": calculation_time,
}
# Keep geometry metadata.json intact; ORCA TD-DFT metadata goes here:
(out_path.parent / "td_metadata.json").write_text(json.dumps(meta, indent=2))
PY

'

RC=$?
set -e

echo "ORCA TD-DFT batch finished with RC=$RC."
exit $RC
