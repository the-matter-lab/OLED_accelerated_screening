#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=[[CPUS_PER_TASK]]
[[WALLTIME_LINE]]
#SBATCH --job-name=[[JOB_NAME]]

set -euo pipefail

echo "Nodes:"
cat "$SLURM_JOB_NODELIST" || true
cd "$SLURM_SUBMIT_DIR"

# Threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-[[CPUS_PER_TASK]]}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-[[CPUS_PER_TASK]]}
ulimit -s unlimited

# Modules (adapt if needed)
module --force purge
module load CCEnv
module load StdEnv/2020
module load crest/2.12
# activate your Python env for the inline parsing step
source "$HOME/jupyter_oled/bin/activate"

# Inputs
cp "[[XYZ_SRC_REL]]" "[[XYZ_BASENAME]]"

# Optional constraints
CINP=""
if [ -f constraints.inp ]; then
  CINP="--cinp constraints.inp"
fi

# Fallback timing anchors (in case crest.out doesn't have wall-time line)
START_EPOCH=$(date +%s)

# Run CREST (don't hard-exit on nonzero so we can always write metadata)
set +e
echo "Starting CREST on [[XYZ_BASENAME]] with OMP_NUM_THREADS=${OMP_NUM_THREADS}, GBSA=[[GBSA_SOLVENT]]"
crest "[[XYZ_BASENAME]]" --gfn2 $CINP --gbsa "[[GBSA_SOLVENT]]" --mquick --noreftopo -T "${OMP_NUM_THREADS}" > crest.log 2>&1
RC=$?
set -e
mv -f crest.log crest.out

END_EPOCH=$(date +%s)

# --- Parse success & wall time from crest.out; fallback to epoch delta ---
python3 - <<'PY'
import re, json, pathlib, os

# Read crest.out
out = pathlib.Path("crest.out").read_text(errors="ignore")

# Success if banner present
success = "CREST terminated normally." in out

# Try to parse "Overall wall time  : 0h : 0m :53s"
sec = None
m = re.search(r"Overall\s+wall\s+time\s*:\s*(\d+)h\s*:\s*(\d+)m\s*:\s*(\d+)s", out)
if m:
    h, mi, s = map(int, m.groups())
    sec = h*3600 + mi*60 + s

# Fallback to wallclock delta captured around the run
if sec is None:
    try:
        start = int(os.environ.get("START_EPOCH","0"))
        end   = int(os.environ.get("END_EPOCH","0"))
        if end > start > 0:
            sec = end - start
    except Exception:
        sec = None

meta = {
    "runtime_seconds": int(sec) if sec is not None else 0,
    "success": bool(success),
}
pathlib.Path("metadata.json").write_text(json.dumps(meta, indent=2))
PY

echo "CREST done. success banner: $(grep -q 'CREST terminated normally\.' crest.out && echo yes || echo no) rc=$RC"

