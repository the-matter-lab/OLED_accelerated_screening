#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=[[CPUS_PER_TASK]]
[[WALLTIME_LINE]]
#SBATCH --job-name=[[JOB_NAME]]

echo "Nodes:"
cat "$SLURM_JOB_NODELIST" || true
cd "$SLURM_SUBMIT_DIR"

set -euo pipefail

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

# Clean stale files
rm -f wfn.xtb xtbtopo.mol tda.dat wbo xtbrestart xtb4stda.out std2.out std2_singlets.out std2_triplets.out tda_singlets.dat tda_triplets.dat

echo "GBSA=[[GBSA_SOLVENT]]  EMAX=[[EMAX_EV]] eV  OMP=$OMP_NUM_THREADS"
echo "Input XYZ: [[XYZ_BASENAME]]"
if [ ! -s "[[XYZ_BASENAME]]" ]; then
  echo "ERROR: Missing input [[XYZ_BASENAME]]" >&2
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
  "xyz_input": "[[XYZ_BASENAME]]"
}, indent=2))
PY
  exit 2
fi

T0=$(date +%s)

# 1) xtb4stda (pre-step)
echo "Running xtb4stda on [[XYZ_BASENAME]] with -gbsa [[GBSA_SOLVENT]]"
TXTB0=$(date +%s)
xtb4stda "[[XYZ_BASENAME]]" --gfn2 -gbsa [[GBSA_SOLVENT]] > xtb4stda.out 2>&1
TXTB1=$(date +%s)

# 2) std2 singlets
echo "Running std2 (singlets) up to [[EMAX_EV]] eV"
TSING0=$(date +%s)
std2 -xtb -e [[EMAX_EV]] > std2_singlets.out 2>&1 || true
# Move tda.dat if created
if [ -s "tda.dat" ]; then mv -f tda.dat tda_singlets.dat; fi
TSING1=$(date +%s)

# 3) std2 triplets (-t)
echo "Running std2 (triplets) up to [[EMAX_EV]] eV"
TTRIP0=$(date +%s)
std2 -xtb -t -e [[EMAX_EV]] > std2_triplets.out 2>&1 || true
# Move tda.dat if created
if [ -s "tda.dat" ]; then mv -f tda.dat tda_triplets.dat; fi
TTRIP1=$(date +%s)

T1=$(date +%s)

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

# time accounting
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
  "xyz_input": "[[XYZ_BASENAME]]",
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

echo "std2 singlets+triplets finished. Total runtime $((T1-T0)) s"

