#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

SUCCESS_BANNER = "CREST terminated normally."


def extract_error_message(crest_text: str, max_chars: int = 4000) -> str:
    """
    Extract a meaningful error block from crest.out.
    Priority:
      1. Blocks around lines with ERROR/FATAL/STOP
      2. Otherwise, the last ~80 lines.
    """
    lines = crest_text.splitlines()

    err_idx = [
        i for i, L in enumerate(lines)
        if re.search(r"\b(ERROR|Error|FATAL|Fatal|STOP)\b", L)
    ]

    snippets = []

    if err_idx:
        # Collect windows ~10 lines above and 30 below each error line.
        for i in err_idx:
            start = max(0, i - 10)
            end = min(len(lines), i + 30)
            block = "\n".join(lines[start:end]).strip()
            if block and block not in snippets:
                snippets.append(block)
    else:
        # Fallback: the tail
        tail = lines[-80:] if len(lines) > 80 else lines
        snippets = ["\n".join(tail).strip()]

    msg = "\n\n---\n\n".join(snippets).strip()

    if len(msg) > max_chars:
        msg = msg[:max_chars] + "\n...[truncated]..."

    return msg


def read_smiles(obabel_meta_path: Path) -> str:
    try:
        meta = json.loads(obabel_meta_path.read_text())
        return meta.get("smiles", "")
    except Exception:
        return ""


def scan_compounds(compounds_dir: Path):
    failures = []

    # Scan directories like root/partition_*
    for part_dir in sorted(compounds_dir.glob("partition_*")):
        if not part_dir.is_dir():
            continue

        for mol_dir in sorted(
            d for d in part_dir.iterdir() if d.is_dir() and d.name.startswith("mol")
        ):
            crest_dir = mol_dir / "crest"
            crest_out = crest_dir / "crest.out"

            # Missing crest.out → failure
            if not crest_out.is_file():
                obabel_meta = mol_dir / "obabel" / "metadata.json"
                failures.append({
                    "partition": part_dir.name,
                    "molecule": mol_dir.name,
                    "crest_out": str(crest_out),
                    "crest_xyz": str(crest_dir / f"{mol_dir.name}.xyz"),
                    "smiles": read_smiles(obabel_meta),
                    "error_message": "crest.out not found",
                })
                continue

            txt = crest_out.read_text(errors="ignore")

            # Success?
            if SUCCESS_BANNER in txt:
                continue

            # Failure → extract meaningful error snippet
            error_excerpt = extract_error_message(txt)
            obabel_meta = mol_dir / "obabel" / "metadata.json"

            failures.append({
                "partition": part_dir.name,
                "molecule": mol_dir.name,
                "crest_out": str(crest_out),
                "crest_xyz": str(crest_dir / f"{mol_dir.name}.xyz"),
                "smiles": read_smiles(obabel_meta),
                "error_message": error_excerpt,
            })

    return failures


def main():
    ap = argparse.ArgumentParser(
        description="Scan CREST failures across partitions and save a JSON report."
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Root directory containing partition_* folders (formerly --compounds_dir)",
    )
    ap.add_argument(
        "--logs",
        required=True,
        help="Directory where the output JSON report will be written",
    )
    ap.add_argument(
        "--outfile",
        default="failed_crest.json",
        help="Output JSON filename inside the logs directory (default: failed_crest.json)",
    )

    args = ap.parse_args()

    compounds_dir = Path(args.root).resolve()
    logs_dir = Path(args.logs).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    failures = scan_compounds(compounds_dir)

    out_path = logs_dir / args.outfile
    out_path.write_text(json.dumps(failures, indent=2))

    print(f"Wrote {len(failures)} failures → {out_path}")


if __name__ == "__main__":
    """
    Usage:
    python src/run/filter_crest.py --root compounds --logs logs
    """
    main()
