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
source $HOME/jupyter_oled/bin/activate

# Inputs
cp "[[XYZ_SRC_REL]]" "[[XYZ_BASENAME]]"

# Optional constraints
CINP=""
if [ -f constraints.inp ]; then
  CINP="--cinp constraints.inp"
fi

# Timing
START_ISO=$(date -Iseconds)
START_EPOCH=$(date +%s)

# Run CREST (don't hard-exit on nonzero so we can write metadata)
set +e
echo "Starting CREST on [[XYZ_BASENAME]] with OMP_NUM_THREADS=${OMP_NUM_THREADS}, GBSA=[[GBSA_SOLVENT]]"
crest "[[XYZ_BASENAME]]" --gfn2 $CINP --gbsa "[[GBSA_SOLVENT]]" --mquick --noreftopo -T "${OMP_NUM_THREADS}" > crest.log 2>&1
RC=$?
set -e
mv -f crest.log crest.out

# Best structure (kept locally as best.xyz)
BEST_LOCAL=""
if [ -f crest_best.xyz ]; then
  cp -f crest_best.xyz best.xyz
  BEST_LOCAL="best.xyz"
fi

# Success detection
SUCCESS="false"
if grep -q "CREST terminated normally\." crest.out; then
  SUCCESS="true"
fi

END_ISO=$(date -Iseconds)
END_EPOCH=$(date +%s)
RUNTIME=$((END_EPOCH - START_EPOCH))

# Metadata
python3 - <<'PY'
import json, os, pathlib
meta = {
  "started": os.environ.get("START_ISO",""),
  "finished": os.environ.get("END_ISO",""),
  "runtime_seconds": int(os.environ.get("RUNTIME","0")),
  "returncode": int(os.environ.get("RC","0")),
  "success": os.environ.get("SUCCESS","false") == "true",
  "gbsa": "[[GBSA_SOLVENT]]",
  "threads": int(os.environ.get("OMP_NUM_THREADS","0")),
  "xyz_input": "[[XYZ_BASENAME]]",
  "best_local": os.environ.get("BEST_LOCAL","")
}
pathlib.Path("metadata.json").write_text(json.dumps(meta, indent=2))
PY

echo "CREST done. success=$SUCCESS runtime=${RUNTIME}s rc=$RC"

