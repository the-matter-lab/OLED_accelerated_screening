#!/usr/bin/env python3

import argparse
import csv
import gzip
import hashlib
import shutil
import tarfile
from pathlib import Path


STEP_DIRS = [
    "freq_b97_3c",
    "opt_b97_3c",
    "td-dft_wb97x-D3",
]

ESSENTIAL_SUFFIXES = (
    ".in",
    ".out",
    ".property.txt",
    ".json",
    ".xyz",
    ".hess",
    ".engrad",
    ".opt",
    ".cis",
)

OPTIONAL_SOLVENT_SUFFIXES = (
    ".cpcm",
    ".cpcm_corr",
)

HEAVY_SUFFIXES = (
    ".gbw",
    ".densities",
    ".densitiesinfo",
)

COMPRESS_SUFFIXES = (
    ".out",
    ".property.txt",
    ".hess",
    ".engrad",
    ".opt",
    ".cis",
    "_trj.xyz",
    ".cpcm",
    ".cpcm_corr",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def natural_mol_key(path: Path):
    digits = "".join(ch for ch in path.name if ch.isdigit())
    return int(digits) if digits else path.name


def collect_molecule_dirs(validation_path: Path) -> list[Path]:
    return sorted(
        [p for p in validation_path.glob("mol*") if p.is_dir()],
        key=natural_mol_key,
    )


def should_keep_file(
    path: Path,
    include_solvent_files: bool = False,
    include_heavy_files: bool = False,
) -> bool:
    name = path.name

    if name == "calculated_file":
        return True

    if name == "crest_best.xyz":
        return True

    if name.endswith(HEAVY_SUFFIXES):
        return include_heavy_files

    if name.endswith(OPTIONAL_SOLVENT_SUFFIXES):
        return include_solvent_files

    return name.endswith(ESSENTIAL_SUFFIXES)


def should_compress(path: Path) -> bool:
    return path.name.endswith(COMPRESS_SUFFIXES)


def copy_or_gzip(src: Path, dst: Path, compress: bool) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if compress:
        dst = dst.with_name(dst.name + ".gz")
        with src.open("rb") as f_in, gzip.open(dst, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy2(src, dst)

    return dst


def write_readme(archive_path: Path) -> None:
    readme = """# OLED TD-DFT validation raw calculation archive

This archive contains reproducibility-oriented raw files for the OLED TD-DFT validation calculations.

Directory structure:

- `molecules/<mol_id>/crest_best.xyz`: selected CREST conformer used as input to DFT optimization.
- `molecules/<mol_id>/opt_b97_3c/`: ORCA B97-3c geometry-optimization input, output, geometries, trajectory and metadata.
- `molecules/<mol_id>/freq_b97_3c/`: ORCA B97-3c frequency-calculation input, output, Hessian and frequency metadata.
- `molecules/<mol_id>/td-dft_wb97x-D3/`: ORCA TD-DFT input, output, excited-state auxiliary files and metadata.

Large ORCA binary/intermediate files such as `.gbw`, `.densities`, and `.densitiesinfo` are excluded by default.
They can be included with `--include-heavy-files`.

CPCM solvent files are excluded by default.
They can be included with `--include-solvent-files`.

Most large text-like outputs are gzip-compressed.
"""
    (archive_path / "README.md").write_text(readme)


def make_tarball(archive_path: Path) -> Path:
    tar_path = archive_path.with_suffix(".tar.gz")

    if tar_path.exists():
        tar_path.unlink()

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(archive_path, arcname=archive_path.name)

    return tar_path


def prepare_zenodo_archive(
    validation_path: Path,
    archive_path: Path,
    overwrite: bool = False,
    include_solvent_files: bool = False,
    include_heavy_files: bool = False,
    make_tar: bool = False,
    dry_run: bool = False,
) -> None:
    validation_path = validation_path.resolve()
    archive_path = archive_path.resolve()

    if not validation_path.exists():
        raise FileNotFoundError(f"validation path does not exist: {validation_path}")

    if not validation_path.is_dir():
        raise NotADirectoryError(f"validation path is not a directory: {validation_path}")

    mol_dirs = collect_molecule_dirs(validation_path)
    if not mol_dirs:
        raise RuntimeError(f"No mol* directories found in {validation_path}")

    if archive_path.exists():
        if not overwrite and not dry_run:
            raise FileExistsError(
                f"archive path already exists: {archive_path}\n"
                "Use --overwrite to replace it."
            )
        if overwrite and not dry_run:
            shutil.rmtree(archive_path)

    manifest_rows = []
    n_files = 0

    if not dry_run:
        archive_path.mkdir(parents=True, exist_ok=True)
        write_readme(archive_path)

    for mol_dir in mol_dirs:
        mol_id = mol_dir.name
        files_to_copy = []

        crest_best = mol_dir / "crest_best.xyz"
        if crest_best.exists():
            files_to_copy.append(
                (crest_best, Path("molecules") / mol_id / "crest_best.xyz")
            )

        for step in STEP_DIRS:
            step_dir = mol_dir / step
            if not step_dir.exists():
                continue

            for src in sorted(step_dir.iterdir()):
                if not src.is_file():
                    continue

                if should_keep_file(
                    src,
                    include_solvent_files=include_solvent_files,
                    include_heavy_files=include_heavy_files,
                ):
                    rel_dst = Path("molecules") / mol_id / step / src.name
                    files_to_copy.append((src, rel_dst))

        for src, rel_dst in files_to_copy:
            compress = should_compress(src)
            dst = archive_path / rel_dst

            if dry_run:
                final_dst = str(dst) + (".gz" if compress else "")
                print(f"{src} -> {final_dst}")
                continue

            written = copy_or_gzip(src, dst, compress=compress)

            manifest_rows.append(
                {
                    "molecule_id": mol_id,
                    "source_path": str(src),
                    "archive_path": str(written.relative_to(archive_path)),
                    "original_size_bytes": src.stat().st_size,
                    "archive_size_bytes": written.stat().st_size,
                    "compressed": compress,
                    "sha256_archive_file": sha256_file(written),
                }
            )
            n_files += 1

    if dry_run:
        print(f"Dry run complete. Molecules found: {len(mol_dirs)}")
        return

    manifest_path = archive_path / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "molecule_id",
                "source_path",
                "archive_path",
                "original_size_bytes",
                "archive_size_bytes",
                "compressed",
                "sha256_archive_file",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Prepared archive folder: {archive_path}")
    print(f"Molecules found: {len(mol_dirs)}")
    print(f"Files copied: {n_files}")
    print(f"Manifest: {manifest_path}")

    if make_tar:
        tar_path = make_tarball(archive_path)
        print(f"Tarball: {tar_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Zenodo-ready archive from validation/mol* OLED TD-DFT "
            "calculation folders."
        )
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        required=True,
        help="Path to the validation directory containing mol* folders.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        required=True,
        help="Output archive directory to create.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite archive path if it already exists.",
    )
    parser.add_argument(
        "--include-solvent-files",
        action="store_true",
        help="Include ORCA .cpcm and .cpcm_corr files.",
    )
    parser.add_argument(
        "--include-heavy-files",
        action="store_true",
        help="Include heavy ORCA intermediate files such as .gbw and .densities.",
    )
    parser.add_argument(
        "--make-tar",
        action="store_true",
        help="Create archive_path.tar.gz after preparing the folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without creating files.",
    )

    args = parser.parse_args()

    prepare_zenodo_archive(
        validation_path=args.validation_path,
        archive_path=args.archive_path,
        overwrite=args.overwrite,
        include_solvent_files=args.include_solvent_files,
        include_heavy_files=args.include_heavy_files,
        make_tar=args.make_tar,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    """
    Usage example from the repository root:

      python src/archive/prepare_zenodo_archive.py \\
          --validation-path validation \\
          --archive-path zenodo_oled_tddft_raw \\
          --overwrite \\
          --make-tar

    To include CPCM files:

      python src/archive/prepare_zenodo_archive.py \\
          --validation-path validation \\
          --archive-path zenodo_oled_tddft_raw \\
          --overwrite \\
          --include-solvent-files \\
          --make-tar

    To include heavy ORCA restart/intermediate files:

      python src/archive/prepare_zenodo_archive.py \\
          --validation-path validation \\
          --archive-path zenodo_oled_tddft_raw_heavy \\
          --overwrite \\
          --include-solvent-files \\
          --include-heavy-files \\
          --make-tar
    """
    main()
