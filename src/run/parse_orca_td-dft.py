#!/usr/bin/env python3
"""
Parse ORCA TD-DFT outputs from td-dft_wb97x-D3 folders into a CSV.

For each molXXXX under --root it expects (if present):

  molXXXX/td-dft_wb97x-D3/
      molXXXX.out        # ORCA TD-DFT output
      molXXXX.xyz        # geometry (not used here)
      metadata.json      # from prepare_td-dft.py (not required)

What is extracted per molecule
------------------------------
From the uncorrected "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
block (electric dipole, NOT SOC-corrected), we take the first `num_states` singlet
transitions (i.e. transitions whose final state multiplicity is 1). These are
stored as:

  - E_Sk(eV)        : float
        Excitation energy of the k-th *uncorrected* singlet-like state in eV.
        k = 1..num_states. Energies are taken directly from the "Energy (eV)"
        column of the absorption table.

  - wl_Sk(nm)       : float
        Wavelength (nm) of the same transition.

  - osc_Sk          : float
        Oscillator strength fosc(D2) of the same transition.

From the "TD-DFT/TDA EXCITED STATES (TRIPLETS)" section we read the first
(num_states - 1) triplet roots that are printed and store:

  - E_Tk(eV)        : float
        Excitation energy of the k-th triplet root in eV, as printed in the
        "E= ... eV" field of the triplet state lines. k = 1..(num_states-1).

Using these, we also define:

  - E(S1-T1)(eV)    : float
        Energy gap between the first uncorrected singlet and triplet roots:
            E(S1-T1) = E_S1(eV) - E_T1(eV)

From the "SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE
MOMENTS" block we again select the first `num_states` singlet-like transitions
(final state multiplicity = 1, i.e. labels like "10-1.0A", "11-1.0A", ...). For
these we store:

  - E_Sk_SOC(eV)    : float
        Excitation energy of the k-th SOC-corrected singlet-like transition in eV.

  - wl_Sk_SOC(nm)   : float
        Wavelength (nm) of the same SOC-corrected transition.

  - osc_Sk_SOC      : float
        Oscillator strength fosc(D2) of the same SOC-corrected transition.

If the SOC-corrected block is absent (e.g. no SOC calculation was requested),
no SOC-related keys are added for that molecule.

From the "ORBITAL ENERGIES" section we extract HOMO/LUMO information:

  - HOMO(eV)        : float
        Energy of the highest occupied molecular orbital in eV.

  - LUMO(eV)        : float
        Energy of the lowest unoccupied molecular orbital in eV.

  - Gap(eV)         : float
        HOMO–LUMO gap in eV: Gap = LUMO(eV) - HOMO(eV), rounded to 2 decimals.

If the orbital-energy section cannot be parsed, an "error" key is returned
instead and HOMO/LUMO/Gap are omitted.

Per-molecule identifiers:

  - mol             : str
        Molecule directory name, e.g. "mol0123".

  - smiles          : str
        SMILES string, primarily taken from an STD2 CSV (see --std2-csv) that
        must have at least columns 'mol' and 'smiles'. If the SMILES is not
        found there, we fall back to molXXXX/obabel/metadata.json under any of
        the keys: "smiles", "SMILES", "canonical_smiles", "canonical".
"""

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Dict, List, Optional

# Try to import pandas; if it fails, attempt to load scipy-stack and retry
try:
    import pandas as pd
except ImportError:
    import subprocess
    # This relies on the environment modules system being available in the shell
    subprocess.run("module load scipy-stack/2025a", shell=True, check=False)
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit(
            "ERROR: pandas is required but could not be imported, "
            "even after 'module load scipy-stack/2025a'."
        ) from e

MOL_RE = re.compile(r"^mol\d+$")  # e.g. mol0001


# ---------- Low-level helpers ----------

def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_smiles_from_std2(path: Path) -> Dict[str, str]:
    """
    Load SMILES from an STD2 CSV file.

    The CSV is expected to contain at least columns:
      - 'mol'    : molecule identifier like 'mol0001'
      - 'smiles' : SMILES string

    Parameters
    ----------
    path : Path
        Path to the CSV file (e.g. results/std2/std2.csv).

    Returns
    -------
    dict
        Mapping mol_name -> smiles (both stripped strings). If the file is not
        found or is malformed, an empty dict is returned.
    """
    if not path.is_file():
        print(f"[WARN] STD2 CSV not found at {path}. SMILES will be empty.")
        return {}

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read STD2 CSV {path}: {e}")
        return {}

    if "mol" not in df.columns or "smiles" not in df.columns:
        print(f"[WARN] STD2 CSV missing 'mol'/'smiles' columns: {path}")
        return {}

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        mol = str(row["mol"]).strip()
        smi = str(row["smiles"]).strip()
        if mol and smi:
            mapping[mol] = smi
    return mapping


def extract_homo_lumo_gap(lines: List[str]) -> Dict[str, float]:
    """
    Extract HOMO, LUMO, and HOMO-HUMO gap from the ORBITAL ENERGIES section.

    Returns
    -------
    dict
        Keys:
          - "HOMO(eV)", "LUMO(eV)", "Gap(eV)"  (floats)
        or
          - {"error": "..."} if parsing fails.
    """
    orbital_energies: List[tuple] = []
    capture = False
    start_index: Optional[int] = None

    for i, line in enumerate(lines):
        if "ORBITAL ENERGIES" in line:
            for j in range(i + 1, len(lines)):
                if "E(eV)" in lines[j]:
                    start_index = j + 1
                    capture = True
                    break
            break

    if not capture or start_index is None:
        return {"error": "Could not locate ORBITAL ENERGIES section."}

    for line in lines[start_index:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        parts = stripped.split()
        if len(parts) >= 4:
            try:
                occ = float(parts[1])
                energy_ev = float(parts[3])
                orbital_energies.append((occ, energy_ev))
            except ValueError:
                continue

    if not orbital_energies:
        return {"error": "Orbital section was empty or malformed."}

    occupied = [e for occ, e in orbital_energies if occ > 0]
    virtual = [e for occ, e in orbital_energies if occ == 0]

    if not occupied or not virtual:
        return {"error": "Could not determine HOMO and/or LUMO."}

    homo = max(occupied)
    lumo = min(virtual)
    gap = round(lumo - homo, 2)

    return {
        "HOMO(eV)": homo,
        "LUMO(eV)": lumo,
        "Gap(eV)": gap,
    }


def _parse_final_state_multiplicity(label: str) -> Optional[int]:
    """
    Parse multiplicity from a state label like '1-3A' or '10-1.0A'.

    Expected formats:
      - '<idx>-<mult><sym>'     e.g. '1-3A'
      - '<idx>-<mult>.<x><sym>' e.g. '1-3.0A', '10-1.0A'

    Returns multiplicity as int, or None if not parseable.
    """
    label = label.strip().strip(",;")
    if "-" not in label:
        return None
    try:
        _, right = label.split("-", 1)
    except ValueError:
        return None

    num_str = ""
    for ch in right:
        if ch.isdigit() or ch == ".":
            num_str += ch
        else:
            break
    if not num_str:
        return None
    try:
        return int(round(float(num_str)))
    except ValueError:
        return None


def _parse_absorption_block_singlets(
    lines: List[str],
    header_index: int,
    num_states: int,
    soc_corrected: bool = False,
) -> List[Dict[str, float]]:
    """
    Parse singlet(-like) transitions from an ABSORPTION SPECTRUM block.

    Parameters
    ----------
    lines : list of str
        Full ORCA output split into lines.
    header_index : int
        Index of the line containing either:
          - 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS'
        or
          - 'SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS'
    num_states : int
        Maximum number of singlet-like transitions to return.
    soc_corrected : bool
        Only used for clarity; same parsing logic, but the caller will assign
        keys with '_SOC' suffix.

    Returns
    -------
    list of dict
        Each dict has:
          - 'E_eV' : float
          - 'wl_nm': float
          - 'fosc' : float
    """
    start = None
    for i in range(header_index + 1, len(lines)):
        if lines[i].strip().startswith("---"):
            start = i + 1
            break
    if start is None:
        return []

    results: List[Dict[str, float]] = []
    for ln in lines[start:]:
        stripped = ln.strip()
        if not stripped:
            if results:
                break
            else:
                continue
        if stripped.startswith("---"):
            if results:
                break
            else:
                continue

        parts = stripped.split()
        if len(parts) < 7:
            continue
        trans_i = parts[0]
        arrow = parts[1]
        trans_f = parts[2]
        if arrow != "->" and "->" not in stripped:
            continue

        mult = _parse_final_state_multiplicity(trans_f)
        if mult is None or mult != 1:
            continue

        try:
            e_ev = float(parts[3])
            wl_nm = float(parts[5])
            fosc = float(parts[6])
        except ValueError:
            continue

        results.append({
            "E_eV": e_ev,
            "wl_nm": wl_nm,
            "fosc": fosc,
        })
        if len(results) >= num_states:
            break

    return results


def extract_excitation_data(filename: Path, num_states: int = 3) -> Dict[str, float]:
    """
    Extract TD-DFT excitation data (uncorrected and SOC-corrected) and
    HOMO/LUMO information from an ORCA .out file.

    Parameters
    ----------
    filename : Path
        Path to the ORCA TD-DFT output file (molXXXX.out).
    num_states : int, optional
        Number of singlet(-like) excited states to extract (default: 3).

    Returns
    -------
    dict
        Keys present when successfully parsed may include:

        Uncorrected singlet absorption (electric dipole, NO SOC):
          - E_Sk(eV)      : float
              Energy of the k-th singlet-like transition (k = 1..num_states), in eV.
          - wl_Sk(nm)     : float
              Wavelength (nm) of the same transition.
          - osc_Sk        : float
              Oscillator strength fosc(D2) of that transition.

        Triplet TD-DFT/TDA energies (from 'TD-DFT/TDA EXCITED STATES (TRIPLETS)'):
          - E_Tk(eV)      : float
              Energy of the k-th triplet root (k = 1..num_states-1), in eV.

        Singlet–triplet splitting (uncorrected):
          - E(S1-T1)(eV)  : float
              E_S1(eV) - E_T1(eV) when both are available.

        SOC-corrected absorption (electric dipole):
          - E_Sk_SOC(eV)  : float
              Energy of the k-th SOC-corrected singlet-like transition.
          - wl_Sk_SOC(nm) : float
              Wavelength (nm) of the same SOC-corrected transition.
          - osc_Sk_SOC    : float
              Oscillator strength fosc(D2) of that SOC-corrected transition.

        Orbital energies:
          - HOMO(eV)      : float
          - LUMO(eV)      : float
          - Gap(eV)       : float

        If parsing of a given block fails, the corresponding keys are simply
        omitted. If the orbital-energies section cannot be parsed at all, an
        "error" key describing the issue is added instead of HOMO/LUMO/Gap.
    """
    data: Dict[str, float] = {}
    start_line_singlets: Optional[int] = None
    start_line_triplet_section: Optional[int] = None
    start_line_soc_abs: Optional[int] = None
    count_triplets = 0

    try:
        raw_text = filename.read_text(errors="ignore")
    except Exception as e:
        print(f"*** Error reading {filename}: {e} ***")
        return data

    lines = raw_text.splitlines()

    if "*** ORCA-CIS/TD-DFT FINISHED WITHOUT ERROR ***" not in raw_text:
        print(f"*** {filename} is incomplete (TD-DFT did not finish) ***")
        return data

    # Uncorrected singlet absorption
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i]
        if (
            "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in ln
            and "SPIN ORBIT CORRECTED" not in ln
            and "SOC CORRECTED" not in ln
        ):
            start_line_singlets = i
            break

    if start_line_singlets is not None:
        singlets_uncorr = _parse_absorption_block_singlets(
            lines, start_line_singlets, num_states, soc_corrected=False
        )
        for k, entry in enumerate(singlets_uncorr, start=1):
            data[f"E_S{k}(eV)"] = entry["E_eV"]
            data[f"wl_S{k}(nm)"] = entry["wl_nm"]
            data[f"osc_S{k}"] = entry["fosc"]
    else:
        print(f"*** {filename}: No uncorrected singlet absorption data found ***")

    # Triplets
    for i in range(len(lines) - 1, -1, -1):
        if "TD-DFT/TDA EXCITED STATES (TRIPLETS)" in lines[i]:
            start_line_triplet_section = i
            break

    if start_line_triplet_section is not None:
        for ln in lines[start_line_triplet_section:]:
            stripped = ln.strip()
            if not stripped.startswith("STATE"):
                continue
            parts = stripped.replace(":", " ").split()
            if "eV" not in parts:
                continue
            try:
                ev_index = parts.index("eV") - 1
                e_t = float(parts[ev_index])
            except (ValueError, IndexError):
                continue
            count_triplets += 1
            data[f"E_T{count_triplets}(eV)"] = e_t
            if count_triplets == (num_states - 1):
                break
    else:
        print(f"*** {filename}: No triplet TD-DFT section found ***")

    if "E_S1(eV)" in data and "E_T1(eV)" in data:
        try:
            data["E(S1-T1)(eV)"] = round(data["E_S1(eV)"] - data["E_T1(eV)"], 2)
        except Exception as e:
            print(f"*** Error calculating E(S1-T1) in {filename}: {e} ***")

    # SOC-corrected absorption
    for i in range(len(lines) - 1, -1, -1):
        if "SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in lines[i]:
            start_line_soc_abs = i
            break

    if start_line_soc_abs is not None:
        singlets_soc = _parse_absorption_block_singlets(
            lines, start_line_soc_abs, num_states, soc_corrected=True
        )
        for k, entry in enumerate(singlets_soc, start=1):
            data[f"E_S{k}_SOC(eV)"] = entry["E_eV"]
            data[f"wl_S{k}_SOC(nm)"] = entry["wl_nm"]
            data[f"osc_S{k}_SOC"] = entry["fosc"]

    # HOMO/LUMO
    homo_lumo = extract_homo_lumo_gap(lines)
    data.update(homo_lumo)

    return data


# ---------- Main script ----------

def main():
    ap = argparse.ArgumentParser(
        description="Parse ORCA TD-DFT outputs (td-dft_wb97x-D3) into a CSV."
    )
    ap.add_argument(
        "--root",
        default="downselected",
        help="Root directory containing molXXXX subdirs (default: downselected)",
    )
    ap.add_argument(
        "--td-dir-name",
        default="td-dft_wb97x-D3",
        help="Subdirectory name with TD-DFT runs (default: td-dft_wb97x-D3)",
    )
    ap.add_argument(
        "--output-dir",
        default="results/downselection/td-dft",
        help="Output directory for CSV (default: results/downselection/td-dft)",
    )
    ap.add_argument(
        "--num-states",
        type=int,
        default=3,
        help="Number of excited singlet states to parse (default: 3)",
    )
    ap.add_argument(
        "--std2-csv",
        default="results/std2/std2.csv",
        help="CSV with columns including 'mol' and 'smiles' (default: results/std2/std2.csv)",
    )

    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "VEE_wb97x-D3.csv"

    # Load SMILES mapping from STD2 CSV
    std2_path = Path(args.std2_csv).resolve()
    smiles_map = load_smiles_from_std2(std2_path)

    rows: List[Dict[str, object]] = []

    mol_dirs = sorted(p for p in root.iterdir() if p.is_dir() and MOL_RE.match(p.name))
    print(f"Found {len(mol_dirs)} molXXXX directories under {root}")

    for mol_dir in mol_dirs:
        mol = mol_dir.name
        td_dir = mol_dir / args.td_dir_name

        if not td_dir.is_dir():
            continue

        outs = [p for p in td_dir.glob("*.out") if not p.name.startswith("slurm-")]
        if not outs:
            print(f"[WARN] {mol}: no ORCA .out file found in {td_dir}")
            continue

        out_file = max(outs, key=lambda p: p.stat().st_size)

        data_dict = extract_excitation_data(out_file, num_states=args.num_states)
        if not data_dict:
            continue

        # Primary SMILES source: STD2 CSV
        smi = smiles_map.get(mol, "")

        # Fallback to obabel metadata if not found in STD2
        if not smi:
            ob_meta = read_json(mol_dir / "obabel" / "metadata.json")
            if ob_meta:
                for k in ("smiles", "SMILES", "canonical_smiles", "canonical"):
                    if k in ob_meta and isinstance(ob_meta[k], str) and ob_meta[k].strip():
                        smi = ob_meta[k].strip()
                        break

        row: Dict[str, object] = {
            "mol": mol,
            "smiles": smi,
        }
        for k, v in data_dict.items():
            row[k] = v

        rows.append(row)

    if not rows:
        print(f"No valid TD-DFT data found under {root}. No CSV written.")
        return

    rows.sort(key=lambda r: r["mol"])

    fieldnames = ["mol", "smiles"]
    extra_keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames and k not in extra_keys:
                extra_keys.append(k)
    fieldnames.extend(extra_keys)

    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {out_csv} with {len(rows)} rows.")


if __name__ == "__main__":
    """
    Usage example:
      python src/run/parse_orca_td-dft.py \
          --root downselected \
          --output-dir results/downselection/td-dft \
          --td-dir-name td-dft_wb97x-D3 \
          --num-states 3 \
          --std2-csv results/std2/std2.csv
    """
    main()
