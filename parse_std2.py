#!/usr/bin/env python3
"""
Parse std2 tda_{singlets,triplets}.dat files and build a summary CSV.

What it does
------------
- Scans <root>/partition_*/mol*/std2 for:
    - tda_singlets.dat  -> S1 energy (eV) and f_osc(S1)
    - tda_triplets.dat  -> T1 energy (eV)
- Reads SMILES from <root>/partition_*/mol*/obabel/metadata.json
- Optional join of `strength` and `absorption` from --smiles-csv (default: smiles.csv)
- Writes results to --output-dir (default: results/std2)/std2.csv

Assumptions
-----------
- In DATXY block, each data row is:
      idx  E_eV  <col2>  f_osc  <col4>  <col5>
  We take:
      E_S1  = first row's energy (idx==1) from singlets
      f_S1  = the 4th token (0-based index 3) on that row
      E_T1  = first row's energy (idx==1) from triplets
- Triplet file has zero oscillator strengths (as expected).

Usage
-----
  python parse_std2.py --root compounds --output-dir results/std2 --smiles-csv smiles.csv
"""
import argparse
import csv
import json
from pathlib import Path
import re
from typing import Optional, Tuple, Dict

MOL_RE = re.compile(r"^mol\d+$")

def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def parse_tda_first_state(tda_path: Path) -> Optional[Tuple[float, float]]:
    """
    Return (E_eV, f_osc) for the first excited state (idx==1) from a tda_*.dat.
    For triplets, f_osc will just be 0.0 (file prints zeros).
    """
    if not tda_path.is_file():
        return None
    try:
        lines = tda_path.read_text().splitlines()
    except Exception:
        return None

    # find "DATXY" line, then read the next numeric lines
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip().upper().startswith("DATXY"))
    except StopIteration:
        return None

    for ln in lines[start+1:]:
        ln = ln.strip()
        if not ln or not ln[0].isdigit():
            continue
        toks = ln.split()
        # expect: idx, E_eV, ..., f_osc, ...
        if len(toks) < 4:
            continue
        try:
            idx = int(toks[0])
            if idx != 1:
                continue
            e_ev = float(toks[1])
            # In std2 .dat output, f_osc is the 3th token (index 2)
            f_osc = float(toks[2]) if len(toks) >= 3 else 0.0
            return (e_ev, f_osc)
        except Exception:
            continue
    return None

def load_smiles_map(smiles_csv: Path) -> Dict[str, Dict[str, str]]:
    """
    Load a CSV with at least a 'smiles' column, and ideally 'strength' and 'absorption'.
    Returns dict[smiles] -> {'strength': str or '', 'absorption': str or ''}.
    """
    if not smiles_csv.is_file():
        return {}
    out = {}
    with smiles_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if not smi:
                continue
            out[smi] = {
                "strength": (row.get("strength") or "").strip(),
                "absorption": (row.get("absorption") or "").strip(),
            }
    return out

def main():
    ap = argparse.ArgumentParser(description="Parse std2 outputs into a CSV.")
    ap.add_argument("--root", default="compounds",
                    help="Root folder that contains partition_* (default: compounds)")
    ap.add_argument("--output-dir", default="results/std2",
                    help="Output directory (default: results/std2)")
    ap.add_argument("--smiles-csv", default="smiles.csv",
                    help="CSV with columns: smiles,strength,absorption (default: smiles.csv)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "std2.csv"

    # Optional smiles→(strength, absorption) map
    smi_map = load_smiles_map(Path(args.smiles_csv).resolve())

    rows = []
    partitions = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("partition_")])

    for part in partitions:
        for mol_dir in sorted(p for p in part.iterdir() if p.is_dir() and MOL_RE.match(p.name)):
            mol = mol_dir.name

            # Get SMILES from obabel/metadata.json (fallback to empty)
            smi = ""
            ob_meta = read_json(mol_dir / "obabel" / "metadata.json")
            if ob_meta:
                # Try common keys
                for k in ("smiles", "SMILES", "canonical_smiles", "canonical"):
                    if k in ob_meta and isinstance(ob_meta[k], str) and ob_meta[k].strip():
                        smi = ob_meta[k].strip()
                        break

            # Parse tda files
            std2_dir = mol_dir / "std2"
            sing = parse_tda_first_state(std2_dir / "tda_singlets.dat")
            trip = parse_tda_first_state(std2_dir / "tda_triplets.dat")

            # Require both to record a row
            if not sing or not trip:
                continue

            e_s1, f_s1 = sing
            e_t1, _ = trip
            e_st = e_s1 - e_t1

            strength = ""
            absorption = ""
            if smi and smi in smi_map:
                strength = smi_map[smi].get("strength", "")
                absorption = smi_map[smi].get("absorption", "")

            rows.append({
                "partition": part.name,
                "mol": mol,
                "smiles": smi,
                "strength": strength,
                "absorption": absorption,
                "E_S1_eV": f"{e_s1:.6f}",
                "E_T1_eV": f"{e_t1:.6f}",
                "E_ST_eV": f"{e_st:.6f}",
                "f_osc_S1": f"{f_s1:.6f}",
            })

    # Write CSV
    rows.sort(key=lambda r: (r["partition"], r["mol"]))
    fieldnames = ["partition", "mol", "smiles", "strength", "absorption",
                  "E_S1_eV", "E_T1_eV", "E_ST_eV", "f_osc_S1"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {out_csv} with {len(rows)} rows.")

if __name__ == "__main__":
    """Usage example:
    python parse_std2.py --root compounds --output-dir results/std2\
          --smiles-csv smiles.csv
    """
    main()

