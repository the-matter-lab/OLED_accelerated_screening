# OLED Zenodo archive preparation

Prepare the approved dataset locally; this command does not publish a deposit.
Python 3.10+ on Linux/macOS and the standard library are sufficient.

From the project root:

```bash
python -B src/archive/prepare_zenodo_archive.py --project-root . --dry-run
python -B src/archive/prepare_zenodo_archive.py --project-root .
```

Place `finetuned_coldstart_v1.ckpt` in `TO_ZENODO` first, or provide `--model`.
The supplied checkpoint remains a standalone upload file. The other uploads are
`README.md`, `molecule_index.csv`, `paper_compounds.zip`,
`additional_compounds.zip`, `training_dataset.zip`, and `tables.zip`.
The raw organized folders also remain locally for inspection.

`tables.zip` contains the original candidate CSV, full-library and validation
sTD-DFT CSVs, validation TD-DFT CSV, and SI infrared CSV. Calculation metadata
JSONs, analysis/plot products, scratch files, and large intermediate binaries are
excluded. The two original training CSVs are in `training_dataset.zip`.

## Resume

Rerun the identical command after an interruption. Completed copies are skipped
using verified SHA-256 values and file fingerprints. Copies use atomic renames.
ZIP central-directory checkpoints retain completed compressed batches, including
after an abrupt process termination during append; only the unfinished batch
is repeated. Checkpoints are saved after at most 100 members or 256 MiB of
uncompressed input, whichever occurs first (a single larger member may exceed
that byte threshold). The largest ongoing file or batch can take time to finish.

Resume state and partial ZIPs are in `TO_ZENODO/.archive_state`; keep this local
directory until preparation/verification is complete. It is not an upload.
Only one process may use a resume directory at a time.

To check every copied byte and every member of the completed ZIPs:

```bash
python -B src/archive/prepare_zenodo_archive.py --project-root . --verify-only
```

Ordinary reruns reuse completed verification when fingerprints are unchanged.
Modified scientific copies or changed source contents stop the run. Use a fresh
`--output` for an intentionally different source snapshot, with `--model` pointing
to the checkpoint. Existing project data, manuscript/plans, and unrelated files
are never removed or overwritten. An optional `--state-dir` relocates local
checkpoints; it must be on the same filesystem as the output for the final ZIP
rename.

`--record-id` optionally adds the real published record's direct model download
URL to the generated README. It does not create, upload, or publish a record.
Keep the same options for later verification runs.

## Validation

```bash
python -B -m unittest discover -s src/archive -p 'test_*.py'
```

The checks cover selection, missing/failed molecules, protected scientific data,
unchanged reruns, and recovery from abrupt copy/ZIP interruptions.
