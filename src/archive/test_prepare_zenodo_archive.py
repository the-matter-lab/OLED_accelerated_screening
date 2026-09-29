"""Behavior checks for dataset selection, protected data, and hard-stop recovery."""

import csv
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from prepare_zenodo_archive import CSV_SOURCES, IR_NAME, main
from resumable_archive import Member, ResumeStore, file_signature, sha256_file


CODE_DIR = Path(__file__).resolve().parent


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    output = root / "TO_ZENODO"
    output.mkdir()
    (output / "finetuned_coldstart_v1.ckpt").write_bytes(b"checkpoint\x00bytes")
    (output / "Oled_Manuscript.pdf").write_bytes(b"existing manuscript")
    molecules = [("mol0001", "CC"), ("mol0002", "CO"), ("mol0003", "CN")]
    for molecule, _ in molecules:
        base = root / "compounds/partition_0" / molecule
        (base / "obabel").mkdir(parents=True)
        (base / "obabel" / f"{molecule}.xyz").write_bytes((molecule + " initial XYZ").encode())
        (base / "obabel/metadata.json").write_text('{"bad": true}')
        (base / "crest").mkdir()
        (base / "crest/crest.out").write_text("CREST output")
        (base / "crest/gfnff_topo").write_bytes(b"excluded topology")
        if molecule != "mol0003":
            (base / "std2").mkdir()
            (base / "std2/std2_singlets.out").write_text("singlet results")
            (base / "std2/wfn.xtb").write_text("excluded wavefunction")
    for molecule in ("mol0001", "mol0002"):
        for stage in ("opt_b97_3c", "freq_b97_3c", "td-dft_wb97x-D3"):
            folder = root / "validation/validation" / molecule / stage
            folder.mkdir(parents=True)
            (folder / f"{molecule}.in").write_text("scientific input")
            (folder / f"{molecule}.out").write_text("results" if molecule == "mol0001" else "error")
            (folder / "td_metadata.json").write_text("{}")
            (folder / f"{molecule}.cis").write_bytes(b"excluded binary")
            (folder / f"._{molecule}.out").write_bytes(b"excluded sidecar")
    write_csv(root / "smiles.csv", ["smiles", "strength", "absorption"],
              [{"smiles": s, "strength": 0, "absorption": 0} for _, s in molecules] +
              [{"smiles": "CF", "strength": 1, "absorption": 1}])
    rows = [{"mol": m, "smiles": s, "partition": "partition_0"} for m, s in molecules[:2]]
    fields = ["mol", "smiles", "partition"]
    write_csv(root / "results/validation/std2/std2.csv", fields, rows)
    write_csv(root / "results/std2/std2.csv", fields, rows)
    optical = {"mol": "mol0001", "smiles": "CC", "E_S1_SOC(eV)": "1.2", "osc_S1_SOC": "0.1"}
    write_csv(root / "results/validation/td-dft/VEE_wb97x-D3.csv", list(optical), [optical])
    infrared = {k: v for k, v in optical.items() if k != "mol"}
    infrared["SA_score"] = "2.5"
    write_csv(root / f"plots_for_manuscript/validation_optical_properties/{IR_NAME}", list(infrared), [infrared])
    for name in ("oled_all_with_predictions.csv", "rafa_comp_combined.csv"):
        write_csv(root / "training_dataset" / name, ["smiles"], [{"smiles": "CC"}])
    (root / "results/validation/confusion").mkdir()
    (root / "results/validation/confusion/analysis.csv").write_text("exclude this")
    return output


class ArchiveTests(unittest.TestCase):
    def test_pipeline_selection_integrity_and_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = fixture(root)
            args = ["--project-root", str(root)]
            self.assertEqual(main(args), 0)
            self.assertTrue((output / "paper_compounds/mol0002/td-dft_wb97x-D3/mol0002.out").exists())
            self.assertFalse((output / "additional_compounds/mol0003/std2").exists())
            for source, destination in CSV_SOURCES:
                self.assertEqual((root / source).read_bytes(), (output / destination).read_bytes())
            for name in ("paper_compounds", "additional_compounds", "training_dataset", "tables"):
                with zipfile.ZipFile(output / f"{name}.zip") as archive:
                    self.assertIsNone(archive.testzip())
                    self.assertFalse(any(p.endswith(".json") or "confusion" in p or p.endswith(".cis") for p in archive.namelist()))
            with zipfile.ZipFile(output / "tables.zip") as archive:
                self.assertEqual(set(archive.namelist()), {d for _, d in CSV_SOURCES if not d.startswith("training_dataset/")})
            with (output / "molecule_index.csv").open(newline="") as stream:
                rows = {r["mol"]: r for r in csv.DictReader(stream)}
            self.assertEqual(rows["mol0002"]["archive_group"], "paper_compounds")
            self.assertEqual(rows["mol0002"]["in_td_dft_csv"], "False")
            self.assertEqual(rows["mol0004"]["archive_group"], "missing_source_data")
            before = {p.relative_to(output): p.stat().st_mtime_ns for p in output.rglob("*") if p.is_file() and ".archive_state" not in p.parts}
            self.assertEqual(main(args), 0)
            self.assertEqual(before, {p.relative_to(output): p.stat().st_mtime_ns for p in output.rglob("*") if p.is_file() and ".archive_state" not in p.parts})
            self.assertEqual((output / "finetuned_coldstart_v1.ckpt").read_bytes(), b"checkpoint\x00bytes")
            self.assertEqual((output / "Oled_Manuscript.pdf").read_bytes(), b"existing manuscript")
            self.assertEqual(main(args + ["--verify-only"]), 0)

    def test_modified_scientific_copy_is_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "original", root / "copy"
            source.write_bytes(b"original")
            with ResumeStore(root / "state") as store:
                store.ensure_file(source, target)
            target.write_bytes(b"modified")
            with ResumeStore(root / "state") as store:
                with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                    store.ensure_file(source, target)
            self.assertEqual(target.read_bytes(), b"modified")

    def test_hard_stop_during_copy_resumes_without_rewriting_finished_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").write_bytes(b"first")
            (root / "b").write_bytes(os.urandom(200000))
            script = """
import os,sys
from pathlib import Path
from resumable_archive import ResumeStore
p=Path(sys.argv[1])
with ResumeStore(p/'state') as store:
    store.ensure_file(p/'a',p/'copy-a')
    def stop(*args): os._exit(77)
    os.replace=stop
    store.ensure_file(p/'b',p/'copy-b')
"""
            result = subprocess.run([sys.executable, "-c", script, str(root)], cwd=CODE_DIR)
            self.assertEqual(result.returncode, 77)
            before = (root / "copy-a").stat().st_mtime_ns
            self.assertFalse((root / "copy-b").exists())
            self.assertTrue((root / "copy-b.archive-part").exists())
            with ResumeStore(root / "state") as store:
                store.ensure_file(root / "a", root / "copy-a")
                store.ensure_file(root / "b", root / "copy-b")
                self.assertEqual(store.skipped, 1)
            self.assertEqual(before, (root / "copy-a").stat().st_mtime_ns)
            self.assertEqual((root / "b").read_bytes(), (root / "copy-b").read_bytes())
            self.assertFalse((root / "copy-b.archive-part").exists())

    def test_hard_stop_during_zip_append_keeps_committed_compressed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("a", "b", "c"):
                (root / name).write_bytes(os.urandom(100000))
            script = """
import os,sys,zipfile
from pathlib import Path
from resumable_archive import Member,ResumeStore,file_signature,sha256_file
p=Path(sys.argv[1]); members=[Member(n,p/n,sha256_file(p/n),file_signature(p/n)) for n in ['a','b','c']]
original=ResumeStore._write_member
def interrupted(self, archive, member):
    original(self,archive,member)
    if member.name=='b': os._exit(88)
ResumeStore._write_member=interrupted
with ResumeStore(p/'state') as store: store.ensure_zip(p/'data.zip',members,batch_files=1)
"""
            result = subprocess.run([sys.executable, "-c", script, str(root)], cwd=CODE_DIR)
            self.assertEqual(result.returncode, 88)
            with sqlite3.connect(root / "state/progress.sqlite3") as db:
                offset = db.execute("SELECT directory_offset FROM archives").fetchone()[0]
            partial = next((root / "state").glob("*.zip.part"))
            committed_prefix = partial.read_bytes()[:offset]
            members = [Member(n, root / n, sha256_file(root / n), file_signature(root / n)) for n in ("a", "b", "c")]
            written = []
            original = ResumeStore._write_member
            def tracked(store, archive, member):
                written.append(member.name)
                return original(store, archive, member)
            ResumeStore._write_member = tracked
            try:
                with ResumeStore(root / "state") as store:
                    store.ensure_zip(root / "data.zip", members, batch_files=1)
            finally:
                ResumeStore._write_member = original
            self.assertEqual(written, ["b", "c"])
            self.assertEqual((root / "data.zip").read_bytes()[:offset], committed_prefix)
            with zipfile.ZipFile(root / "data.zip") as archive:
                self.assertIsNone(archive.testzip())
                for member in members:
                    self.assertEqual(archive.read(member.name), member.path.read_bytes())

    def test_duplicate_identity_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = fixture(root)
            (root / "compounds/partition_1/mol0001").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "Duplicate compound ID"):
                main(["--project-root", str(root)])
            self.assertFalse((output / ".archive_state").exists())


if __name__ == "__main__":
    unittest.main()
