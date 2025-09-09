#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=[[CPUS_PER_TASK]]
[[WALLTIME_LINE]]
#SBATCH --job-name=[[JOB_NAME]]
#SBATCH --array=0-0

set -euo pipefail

echo "Nodes:"
cat "$SLURM_JOB_NODELIST" || true
cd "$SLURM_SUBMIT_DIR"

# Threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-[[CPUS_PER_TASK]]}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-[[CPUS_PER_TASK]]}
ulimit -s unlimited
export OMP_STACKSIZE=4G

# --- toolchain / modules ---
module --force purge
module load NiaEnv/2019b
module load intel/2019u4

# Activate Python for inline parsing
source "$HOME/jupyter_oled/bin/activate"

# --- std2/xTB4sTDA paths ---
export STD2HOME=${STD2HOME:-$HOME/apps/std2/1.6.1}
export PATH="$STD2HOME/bin:$PATH"
export LD_LIBRARY_PATH="$STD2HOME/lib:$STD2HOME/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/scinet/intel/2019u4/compilers_and_libraries_2019.4.243/linux/compiler/lib/intel64_lin:${LD_LIBRARY_PATH}"

export XTB4STDAHOME=${XTB4STDAHOME:-$HOME/apps/xtb4stda/1.0}
export PATH="$XTB4STDAHOME/bin:$PATH"

# --- workload partitioning ---
LIST_FILE="[[LIST_FILE]]"
CHUNK_SIZE=[[CHUNK_SIZE]]
GBSA="[[GBSA_SOLVENT]]"
EMAX="[[EMAX_EV]]"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2
  exit 2
fi

if [[ ! -s "$LIST_FILE" ]]; then
  echo "ERROR: List file not found or empty: $LIST_FILE" >&2
  exit 2
fi

# Read all molecule dirs
mapfile -t ALL < "$LIST_FILE"
N=${#ALL[@]}
START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE - 1 ))
if (( START >= N )); then
  echo "Nothing to do for task $SLURM_ARRAY_TASK_ID (START=$START >= N=$N)."
  exit 0
fi
if (( END >= N )); then END=$(( N - 1 )); fi

echo "Task $SLURM_ARRAY_TASK_ID processing lines [$START .. $END] (of N=$N), sequentially."
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  GBSA=$GBSA  EMAX=$EMAX eV"

# --- per-molecule sequential loop ---
for IDX in $(seq "$START" "$END"); do
  MOL_DIR="${ALL[$IDX]}"
  [[ -z "$MOL_DIR" ]] && continue
  if [[ ! -d "$MOL_DIR" ]]; then
    echo "[SKIP] Not a directory: $MOL_DIR"
    continue
  fi

  MOL_NAME="$(basename "$MOL_DIR")"
  CREST_DIR="$MOL_DIR/crest"
  STD2_DIR="$MOL_DIR/std2"
  XYZ_SRC="$CREST_DIR/crest_best.xyz"

  if [[ ! -s "$XYZ_SRC" ]]; then
    echo "[SKIP] $MOL_NAME: crest_best.xyz missing"
    continue
  fi

  mkdir -p "$STD2_DIR"
  XYZ_DST="$STD2_DIR/crest_mol_${MOL_NAME}.xyz"
  cp -f "$XYZ_SRC" "$XYZ_DST"

  echo "== [$MOL_NAME] =="
  echo "  working in: $STD2_DIR"

  pushd "$STD2_DIR" >/dev/null

  # Clean stale files
  rm -f wfn.xtb xtbtopo.mol tda.dat wbo xtbrestart xtb4stda.out \
        std2.out std2_singlets.out std2_triplets.out \
        tda_singlets.dat tda_triplets.dat

  if [[ ! -s "$(basename "$XYZ_DST")" ]]; then
    echo "ERROR: Missing input $(basename "$XYZ_DST")" >&2
    python3 - <<'PY'
import json
from pathlib import Path
Path("metadata.json").write_text(json.dumps({
  "xtb4stda_error": True,
  "std2_singlets_error": True,
  "std2_triplets_error": True,
  "runtime_seconds_total": 0,
  "gbsa": "[[GBSA_SOLVENT]]",
  "emax_ev": float("[[EMAX_EV]]"),
  "threads": int("[[CPUS_PER_TASK]]"),
  "xyz_input": Path(".").resolve().name  # folder name if no xyz
}, indent=2))
PY
    popd >/dev/null
    continue
  fi

  T0=$(date +%s)

  # 1) xtb4stda (pre-step)
  echo "  xtb4stda: $(basename "$XYZ_DST")  -gbsa $GBSA"
  TXTB0=$(date +%s)
  set +e
  xtb4stda "$(basename "$XYZ_DST")" --gfn2 -gbsa "$GBSA" > xtb4stda.out 2>&1
  set -e
  TXTB1=$(date +%s)

  # 2) std2 singlets
  echo "  std2 singlets up to $EMAX eV"
  TSING0=$(date +%s)
  set +e
  std2 -xtb -e "$EMAX" > std2_singlets.out 2>&1
  set -e
  # Move tda.dat if created
  if [[ -s "tda.dat" ]]; then mv -f tda.dat tda_singlets.dat; fi
  TSING1=$(date +%s)

  # 3) std2 triplets (-t)
  echo "  std2 triplets up to $EMAX eV"
  TTRIP0=$(date +%s)
  set +e
  std2 -xtb -t -e "$EMAX" > std2_triplets.out 2>&1
  set -e
  if [[ -s "tda.dat" ]]; then mv -f tda.dat tda_triplets.dat; fi
  TTRIP1=$(date +%s)

  T1=$(date +%s)
  export T0 TXTB0 TXTB1 TSING0 TSING1 TTRIP0 TTRIP1 T1

  # --- Parse outputs and write metadata.json ---
  python3 - <<'PY'
import json, os
from pathlib import Path

def ok_contains(path: str, needle: str) -> bool:
    p = Path(path)
    if not p.is_file(): return False
    try:
        return (needle in p.read_text(errors="ignore"))
    except Exception:
        return False

xtb_ok    = ok_contains("xtb4stda.out",       " 1  SCC done.")
sing_ok   = ok_contains("std2_singlets.out",  "sTDA done.")
trip_ok   = ok_contains("std2_triplets.out",  "sTDA done.")

to_i = lambda name: int(os.environ.get(name,"0") or "0")
t_total = to_i("T1") - to_i("T0")
t_xtb   = to_i("TXTB1") - to_i("TXTB0")
t_sing  = to_i("TSING1") - to_i("TSING0")
t_trip  = to_i("TTRIP1") - to_i("TTRIP0")

meta = {
  "xtb4stda_error": (not xtb_ok),
  "std2_singlets_error": (not sing_ok),
  "std2_triplets_error": (not trip_ok),
  "runtime_seconds_total": max(0, t_total),
  "runtime_seconds_xtb4stda": max(0, t_xtb),
  "runtime_seconds_singlets": max(0, t_sing),
  "runtime_seconds_triplets": max(0, t_trip),
  "gbsa": "[[GBSA_SOLVENT]]",
  "emax_ev": float("[[EMAX_EV]]"),
  "threads": int(os.environ.get("OMP_NUM_THREADS","0") or "0"),
  "xyz_input": next((p.name for p in Path('.').glob('crest_mol_*.xyz')), ""),
  "outputs": {
    "xtb4stda_out": "xtb4stda.out",
    "std2_singlets_out": "std2_singlets.out",
    "std2_triplets_out": "std2_triplets.out",
    "tda_singlets": "tda_singlets.dat" if Path("tda_singlets.dat").is_file() else "",
    "tda_triplets": "tda_triplets.dat" if Path("tda_triplets.dat").is_file() else ""
  }
}
Path("metadata.json").write_text(json.dumps(meta, indent=2))
PY

  echo "  done [$MOL_NAME]  total=$((T1-T0))s"
  popd >/dev/null
done

echo "Task $SLURM_ARRAY_TASK_ID finished."

