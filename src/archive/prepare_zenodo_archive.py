#!/usr/bin/env python3
"""Prepare the approved OLED calculation/model dataset; safely resume identical runs."""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from resumable_archive import Member, ResumeStore, file_signature


ORCA_STAGES = ("opt_b97_3c", "freq_b97_3c", "td-dft_wb97x-D3")
ORCA_ENDINGS = (".in", ".out", ".xyz", ".hess", ".engrad", ".property.txt", ".bibtex")
CREST_FILES = {"struc.xyz", "crest_best.xyz", "crest_conformers.xyz", "crest.energies", "crest.out"}
STD2_FILES = {"xtb4stda.out", "std2_singlets.out", "std2_triplets.out", "tda_singlets.dat",
              "tda_triplets.dat", "charges", "wbo"}
MODEL_NAME = "finetuned_coldstart_v1.ckpt"
IR_NAME = "SI_ir_molecules_ranked_with_SA.csv"
CSV_SOURCES = (
    ("training_dataset/oled_all_with_predictions.csv", "training_dataset/oled_all_with_predictions.csv"),
    ("training_dataset/rafa_comp_combined.csv", "training_dataset/rafa_comp_combined.csv"),
    ("smiles.csv", "results/smiles.csv"),
    ("results/std2/std2.csv", "results/std2/std2.csv"),
    ("results/validation/std2/std2.csv", "results/validation/std2/std2.csv"),
    ("results/validation/td-dft/VEE_wb97x-D3.csv", "results/validation/td-dft/VEE_wb97x-D3.csv"),
    (f"plots_for_manuscript/validation_optical_properties/{IR_NAME}", f"supporting_information/{IR_NAME}"),
)
INDEX_FIELDS = ("mol", "smiles", "source_partition", "archive_group", "selected_for_validation",
                "has_compound_folder", "has_std2_folder", "has_td_folder", "in_td_dft_csv", "archive_directory")


@dataclass
class Selection:
    files: list[tuple[Path, str]]
    index: bytes
    infrared: list[tuple[str, dict]]
    paper: set[str]
    additional: set[str]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def molecule_map(root):
    result = {}
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(root.iterdir()):
        if path.is_dir() and re.fullmatch(r"mol\d+", path.name):
            result[path.name] = path
    return result


def keyed_rows(rows, label):
    result = {}
    for row in rows:
        molecule = row["mol"]
        if molecule in result:
            raise ValueError(f"Duplicate molecule {molecule} in {label}")
        result[molecule] = row
    return result


def select_dataset(project: Path, validation: Path) -> Selection:
    compounds = {}
    for partition in sorted((project / "compounds").glob("partition_*")):
        for molecule, directory in molecule_map(partition).items():
            if molecule in compounds:
                raise ValueError(f"Duplicate compound ID across partitions: {molecule}")
            compounds[molecule] = directory
    if not compounds:
        raise RuntimeError("No compound directories found")
    validation_dirs = molecule_map(validation)
    paper = {m for m, p in validation_dirs.items() if (p / "td-dft_wb97x-D3").is_dir()}
    if not paper:
        raise RuntimeError("No validation TD-DFT directories found")
    if paper - compounds.keys():
        raise ValueError(f"Paper molecules without compound matches: {sorted(paper - compounds.keys())}")
    additional = compounds.keys() - paper
    candidates = read_csv(project / "smiles.csv")
    identities = {f"mol{i:04d}": row["smiles"] for i, row in enumerate(candidates, 1)}
    validation_rows = keyed_rows(read_csv(project / "results/validation/std2/std2.csv"), "validation sTD-DFT")
    td_rows = keyed_rows(read_csv(project / "results/validation/td-dft/VEE_wb97x-D3.csv"), "validation TD-DFT")
    std_rows = keyed_rows(read_csv(project / "results/std2/std2.csv"), "complete sTD-DFT")
    if set(validation_rows) != set(validation_dirs):
        raise ValueError("Validation CSV selection differs from validation molecule folders")
    if set(td_rows) - paper or set(std_rows) - compounds.keys():
        raise ValueError("Result CSV contains an unmatched molecule ID")
    if compounds.keys() - identities.keys():
        raise ValueError("Compound IDs do not match the candidate CSV naming convention")
    for rows in (validation_rows, td_rows, std_rows):
        for molecule, row in rows.items():
            if identities.get(molecule) != row["smiles"]:
                raise ValueError(f"Molecular identity differs between original CSVs: {molecule}")
            if "partition" in row and molecule in compounds:
                if row["partition"] != compounds[molecule].parent.name:
                    raise ValueError(f"Partition differs from the result CSV: {molecule}")

    files = []
    for molecule, directory in sorted(compounds.items()):
        group = "paper_compounds" if molecule in paper else "additional_compounds"
        for stage in ("obabel", "crest", "std2"):
            folder = directory / stage
            if not folder.is_dir():
                continue
            allowed = ({f"{molecule}.xyz"} if stage == "obabel" else CREST_FILES if stage == "crest"
                       else STD2_FILES | {f"crest_mol_{molecule}.xyz"})
            for source in sorted(folder.iterdir()):
                if source.is_file() and not source.name.startswith(".") and source.name in allowed:
                    files.append((source, f"{group}/{molecule}/{stage}/{source.name}"))
        if molecule in paper:
            for stage in ORCA_STAGES:
                folder = validation_dirs[molecule] / stage
                if not folder.is_dir():
                    raise FileNotFoundError(f"Missing paper calculation stage: {folder}")
                for source in sorted(folder.iterdir()):
                    if (source.is_file() and source.name.startswith(molecule)
                            and source.name.endswith(ORCA_ENDINGS)):
                        files.append((source, f"{group}/{molecule}/{stage}/{source.name}"))
    files.extend((project / source, destination) for source, destination in CSV_SOURCES)
    for source, destination in files:
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() == ".json" or any(p.startswith(".") for p in Path(destination).parts):
            raise ValueError(f"Excluded metadata/sidecar selected: {source}")

    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, INDEX_FIELDS, lineterminator="\n")
    writer.writeheader()
    for molecule, smiles in identities.items():
        directory = compounds.get(molecule)
        group = "paper_compounds" if molecule in paper else "additional_compounds" if directory else "missing_source_data"
        writer.writerow(dict(
            mol=molecule, smiles=smiles, source_partition=directory.parent.name if directory else "",
            archive_group=group, selected_for_validation=molecule in validation_dirs,
            has_compound_folder=directory is not None,
            has_std2_folder=bool(directory and (directory / "std2").is_dir()),
            has_td_folder=molecule in paper, in_td_dft_csv=molecule in td_rows,
            archive_directory=f"{group}/{molecule}" if directory else "",
        ))
    infrared = []
    for row in read_csv(project / CSV_SOURCES[-1][0]):
        matches = [m for m, original in td_rows.items() if original["smiles"] == row["smiles"]]
        if len(matches) != 1:
            raise ValueError(f"SI infrared molecule does not have a unique TD-DFT match: {row['smiles']}")
        molecule = matches[0]
        for field in ("E_S1_SOC(eV)", "osc_S1_SOC"):
            if row[field] != td_rows[molecule][field]:
                raise ValueError(f"SI infrared property differs from TD-DFT CSV: {molecule}, {field}")
        infrared.append((molecule, row))
    return Selection(files, index.getvalue().encode(), infrared, paper, set(additional))


def make_readme(selection, model_name, model_bytes, record_id=None):
    model_link = (f"https://zenodo.org/records/{record_id}/files/{model_name}?download=1"
                  if record_id else model_name)
    text = f"""# OLED molecular generation: model and calculation dataset

Supporting data for *De novo molecular generation with optical property preconditioning at the token level*.

## Files

| File | Contents |
|---|---|
| [{model_name}]({model_link}) | Standalone trained model checkpoint ({model_bytes:,} bytes); download directly for use with the GitHub inference code |
| [paper_compounds.zip](paper_compounds.zip) | Molecules with a validation TD-DFT calculation folder: Open Babel, CREST, sTD-DFT, DFT optimization, frequencies, and TD-DFT |
| [additional_compounds.zip](additional_compounds.zip) | Other generated molecules: available Open Babel, CREST, and sTD-DFT files |
| [training_dataset.zip](training_dataset.zip) | Original `oled_all_with_predictions.csv` and `rafa_comp_combined.csv` training datasets |
| [tables.zip](tables.zip) | Original candidate/result tables and the SI infrared molecule table, listed below |
| [molecule_index.csv](molecule_index.csv) | Molecule IDs, SMILES, source partition, archive group, and calculation availability |

Extract the ZIP files into the same directory. Each preserves its directory root. Calculation paths are `paper_compounds/molXXXX/<stage>/` or `additional_compounds/molXXXX/<stage>/`. Stages are `obabel`, `crest`, `std2`, and, for paper compounds, `opt_b97_3c`, `freq_b97_3c`, and `td-dft_wb97x-D3`.

## Tables

`tables.zip` contains:

- `results/smiles.csv`: generated candidates and requested conditioning labels.
- `results/std2/std2.csv`: complete-library sTD-DFT results.
- `results/validation/std2/std2.csv`: selected validation candidates and their sTD-DFT results.
- `results/validation/td-dft/VEE_wb97x-D3.csv`: completed validation TD-DFT results.
- `supporting_information/{IR_NAME}`: SI infrared molecules, optical properties, and synthetic-accessibility (SA) scores.

CSV contents and calculation files are preserved unchanged. Excitation and orbital energies are in eV, wavelengths in nm, and oscillator strengths are dimensionless. Columns marked `SOC` contain spin-orbit-coupled quantities. Missing values remain as supplied.

## Calculation records

Geometries and conformer ensembles are XYZ files. ORCA folders retain inputs, full outputs, optimized structures/trajectories, Hessians, gradients, property outputs, and citations where available. Inputs and outputs record the actual methods and solvent settings. Metadata JSONs, large intermediate wavefunction/density files, scratch files, and plot-analysis products are excluded.

Archive groups reflect calculation-folder availability, not successful completion. Unsuccessful TD-DFT attempts remain with their input/error logs; `in_td_dft_csv` identifies molecules with tabulated TD-DFT results. Missing stages are not fabricated. Candidates without source folders are marked `missing_source_data` in the index.

## SI infrared molecules

The SI table order maps to the following calculation folders:

| SI order | Calculation directory |
|---:|---|
"""
    for order, (molecule, _) in enumerate(selection.infrared, 1):
        text += f"| {order} | `paper_compounds/{molecule}/` |\n"
    text += """
## Code and model access

Training, sampling, inference, and plotting code: [oled_generation_optical_preconditioning](https://github.com/aspuru-guzik-group/oled_generation_optical_preconditioning).
Calculation workflow and archive preparation: [OLED_accelerated_screening](https://github.com/aspuru-guzik-group/OLED_accelerated_screening).

The checkpoint is a separate file on this record; it is not inside a dataset ZIP. A downloader can request its file download URL independently of the calculation data.
"""
    return text.encode()


def verify_folder_selection(output, selection):
    for group in ("paper_compounds", "additional_compounds", "training_dataset", "results", "supporting_information"):
        expected = {destination for _, destination in selection.files if destination.split("/")[0] == group}
        actual = {p.relative_to(output).as_posix() for p in (output / group).rglob("*") if p.is_file()}
        if expected != actual:
            raise RuntimeError(f"Unexpected/missing files in {group}: extras={sorted(actual-expected)[:5]}, missing={sorted(expected-actual)[:5]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", "--archive-path", type=Path)
    parser.add_argument("--validation-path", type=Path)
    parser.add_argument("--model", type=Path, help="Defaults to the supplied checkpoint already in the output directory")
    parser.add_argument("--state-dir", type=Path, help="Local resume files; defaults to OUTPUT/.archive_state")
    parser.add_argument("--record-id", type=int, help="Optional published Zenodo record ID for the model link")
    parser.add_argument("--compression-level", type=int, choices=range(1, 10), default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Rehash all copies and recheck every completed ZIP member")
    parser.add_argument("--verify-only", action="store_true", help="Verify an existing completed archive without copying or rebuilding data")
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    output = (args.output or project / "TO_ZENODO").resolve()
    validation = (args.validation_path or project / "validation/validation").resolve()
    model = (args.model or output / MODEL_NAME).resolve()
    if output == project or output in project.parents:
        raise ValueError("Output must not be the project root or its parent")
    selection = select_dataset(project, validation)
    if not model.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model}")
    print(f"Selection: {len(selection.paper)} paper / {len(selection.additional)} additional molecule folders; "
          f"{len(selection.files)} scientific files, {sum(p.stat().st_size for p, _ in selection.files)/1e9:.3f} GB", flush=True)
    if args.dry_run:
        print(f"Output: {output}\nStandalone model: {model}\nZIPs: paper_compounds, additional_compounds, training_dataset, tables", flush=True)
        return 0

    state_dir = args.state_dir or output / ".archive_state"
    with ResumeStore(state_dir) as store:
        members = []
        started = time.monotonic()
        for position, (source, destination) in enumerate(selection.files, 1):
            target = output / destination
            digest = store.ensure_file(source, target, verify=args.verify or args.verify_only,
                                       require_existing=args.verify_only)
            members.append(Member(destination, target, digest, file_signature(target)))
            if time.monotonic() - started > 15:
                print(f"Copies checked: {position}/{len(selection.files)}; new={store.copied}, unchanged={store.skipped}", flush=True)
                started = time.monotonic()
        store.ensure_file(model, output / model.name, verify=args.verify or args.verify_only,
                          require_existing=args.verify_only)
        store.ensure_bytes(selection.index, output / "molecule_index.csv", require_existing=args.verify_only)
        store.ensure_bytes(make_readme(selection, model.name, model.stat().st_size, args.record_id), output / "README.md",
                           require_existing=args.verify_only)
        verify_folder_selection(output, selection)
        print(f"Copies finished: new={store.copied}, unchanged={store.skipped}; README/index ready", flush=True)
        for group in ("paper_compounds", "additional_compounds", "training_dataset", "tables"):
            roots = {"results", "supporting_information"} if group == "tables" else {group}
            store.ensure_zip(output / f"{group}.zip", [m for m in members if m.name.split("/")[0] in roots],
                             level=args.compression_level, verify=args.verify or args.verify_only,
                             require_existing=args.verify_only)
    upload = [output / "README.md", output / "molecule_index.csv", output / model.name]
    upload.extend(output / f"{group}.zip" for group in ("paper_compounds", "additional_compounds", "training_dataset", "tables"))
    print("Ready to upload these files only:", flush=True)
    for path in upload:
        print(f"  {path.name}: {path.stat().st_size / 1e6:.3f} MB", flush=True)
    print(f"Total upload size: {sum(p.stat().st_size for p in upload)/1e9:.3f} GB", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume from committed files/ZIP batches.", file=sys.stderr)
        raise SystemExit(130)
