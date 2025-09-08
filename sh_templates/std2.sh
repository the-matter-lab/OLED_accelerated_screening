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

# Threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-[[CPUS_PER_TASK]]}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-[[CPUS_PER_TASK]]}
ulimit -s unlimited
export OMP_STACKSIZE=4G

# --- toolchain / modules ---
module --force purge
module load NiaEnv/2019b
module load intel/2019u4

# Activate your Python env for the inline parsing step
source "$HOME/jupyter_oled/bin/activate"

# --- std2/xTB4sTDA paths (you can override via env) ---
export STD2HOME=${STD2HOME:-$HOME/apps/std2/1.6.1}
export PATH="$STD2HOME/bin:$PATH"
export LD_LIBRARY_PATH="$STD2HOME/lib:$STD2HOME/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/scinet/intel/2019u4/compilers_and_libraries_2019.4.243/linux/compiler/lib/intel64_lin:${LD_LIBRARY_PATH}"

export XTB4STDAHOME=${XTB4STDAHOME:-$HOME/apps/xtb4stda/1.0}
export PATH="$XTB4STDAHOME/bin:$PATH"


# Clean stale wavefunction/logs (prevents confusion on resubmits)
rm -f wfn.xtb xtbtopo.mol tda.dat wbo xtbrestart xtb4stda.out std2.out

echo "GBSA=[[GBSA_SOLVENT]]  EMAX=[[EMAX_EV]] eV  OMP=$OMP_NUM_THREADS"
echo "Input XYZ: [[XYZ_BASENAME]]"

if [ ! -s "[[XYZ_BASENAME]]" ]; then
  echo "ERROR: Missing input [[XYZ_BASENAME]]" >&2
  # still write metadata with errors
  python3 - <<'PY'
import json
from pathlib import Path
Path("metadata.json").write_text(json.dumps({
  "xtb4stda_error": True,
  "std2_error": True,
  "runtime_seconds": 0,
  "gbsa": "[[GBSA_SOLVENT]]",
  "emax_ev": float("[[EMAX_EV]]"),
  "threads": int("[[CPUS_PER_TASK]]"),
  "xyz_input": "[[XYZ_BASENAME]]"
}, indent=2))
PY
  exit 2
fi

START_EPOCH=$(date +%s)

# Run xTB pre-step (line-buffer so logs update live)
echo "Running xtb4stda on [[XYZ_BASENAME]] with -gbsa [[GBSA_SOLVENT]]"
xtb4stda "[[XYZ_BASENAME]]" --gfn2 -gbsa [[GBSA_SOLVENT]] > xtb4stda.out

# Run std2 up to [[EMAX_EV]] eV (line-buffered)
echo "Running std2 (-xtb) up to [[EMAX_EV]] eV"
std2 -xtb -e [[EMAX_EV]] > std2.out 

END_EPOCH=$(date +%s)
RUNTIME=$((END_EPOCH - START_EPOCH))
export RUNTIME 

# --- Parse outputs and write metadata.json ---
python3 - <<'PY'
import re, json, os
from pathlib import Path

def tail_err(txt, maxlen=800):
    if not txt: return ""
    lines = txt.splitlines()
    tail = "\n".join(lines[-80:]).strip()
    if len(tail) > maxlen: tail = tail[:maxlen] + "\n...[truncated]..."
    return tail

xtb_txt = Path("xtb4stda.out").read_text(errors="ignore") if Path("xtb4stda.out").exists() else ""
std2_txt = Path("std2.out").read_text(errors="ignore") if Path("std2.out").exists() else ""

# xtb4stda success = contains "1  SCC done."
xtb_ok = (" 1  SCC done." in xtb_txt)

# std2 success = contains "sTDA done."
std2_ok = ("sTDA done." in std2_txt)

meta = {
  "xtb4stda_error": (not xtb_ok),
  "std2_error": (not std2_ok),
  "runtime_seconds": int(os.environ.get("RUNTIME","0")),
  "gbsa": "[[GBSA_SOLVENT]]",
  "emax_ev": float("[[EMAX_EV]]"),
  "threads": int(os.environ.get("OMP_NUM_THREADS","0") or "0"),
  "xyz_input": "[[XYZ_BASENAME]]",
}

if not xtb_ok:
    meta["xtb4stda_error_excerpt"] = tail_err(xtb_txt)
if not std2_ok:
    meta["std2_error_excerpt"] = tail_err(std2_txt)

Path("metadata.json").write_text(json.dumps(meta, indent=2))
PY

echo "std2 finished. Runtime ${RUNTIME}s"

