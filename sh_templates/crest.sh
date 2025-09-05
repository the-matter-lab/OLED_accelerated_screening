BATCH --account=rrg-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks={NTASKS}
#SBATCH --time=12:00:00
{WALLTIME_OVERRIDE:+#SBATCH --time={WALLTIME_OVERRIDE}}
#SBATCH --job-name={JOB_NAME}

set -euo pipefail

echo "Nodes:"
cat "$SLURM_JOB_NODELIST" || true
cd "$SLURM_SUBMIT_DIR"

# Threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-{NTASKS}}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-{NTASKS}}
ulimit -s unlimited

# Modules (adapt to your cluster)
module --force purge
module load CCEnv
module load StdEnv/2020
module load crest/2.12

# Inputs
cp "{XYZ_SRC_REL}" "{XYZ_BASENAME}"

# Optional constraints
CINP=""
if [ -f constraints.inp ]; then
  CINP="--cinp constraints.inp"
fi

# Timing
START_ISO=$(date -Iseconds)
START_EPOCH=$(date +%s)

# Run CREST (do not hard-fail if crest nonzero; we want to write metadata)
set +e
echo "Starting CREST on {XYZ_BASENAME} with OMP_NUM_THREADS=$OMP_NUM_THREADS, GBSA={GBSA_SOLVENT}"
crest "{XYZ_BASENAME}" --gfn2 $CINP --gbsa "{GBSA_SOLVENT}" --mquick --noreftopo -T "$OMP_NUM_THREADS" > crest.log 2>&1
RC=$?
set -e

mv -f crest.log crest.out

# Collect best structure(s)
BEST_LOCAL=""
if [ -f crest_best.xyz ]; then
  cp -f crest_best.xyz best.xyz
  BEST_LOCAL="best.xyz"
  # also store canonical best geom one level up
  cp -f crest_best.xyz "{BEST_GEOM_DST}" || true
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
  "gbsa": "{GBSA_SOLVENT}",
  "threads": int(os.environ.get("OMP_NUM_THREADS","0")),
  "xyz_input": "{XYZ_BASENAME}",
  "best_local": "{BL}",
  "best_geom_dst": "{BEST_DST}",
}
pathlib.Path("metadata.json").write_text(json.dumps(meta, indent=2))
PY
# substitute placeholders for best paths (bash can't expand inside the heredoc easily)
python3 - <<'PY'
import json
from pathlib import Path
p = Path("metadata.json")
d = json.loads(p.read_text())
d["best_local"] = "{BL}".format(BL=os.environ.get("BEST_LOCAL",""))
d["best_geom_dst"] = "{BEST_DST}"
p.write_text(json.dumps(d, indent=2))
PY

echo "CREST done. success=$SUCCESS runtime=${RUNTIME}s rc=$RC"

