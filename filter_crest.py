#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

SUCCESS_BANNER = "CREST terminated normally."

def extract_error_message(crest_text: str, max_chars: int = 4000) -> str:
    """
    Try to pull a meaningful error excerpt from crest.out.
    - Prefer blocks around lines containing "ERROR", "Error", "FATAL"
    - Else, fall back to the last ~80 lines.
    """
    lines = crest_text.splitlines()
    # collect indices that look like errors
    err_idx = [i for i, L in enumerate(lines)
               if re.search(r"\b(ERROR|Error|FATAL|Fatal|STOP)\b", L)]
    snippets = []

    if err_idx:
        # Grab a small window around each error line
        for i in err_idx:
            start = max(0, i - 10)
            end = min(len(lines), i + 30)
            block = "\n".join(lines[start:end]).strip()
            if block and block not in snippets:
                snippets.append(block)

    if not snippets:
        # Fallback: the tail
        tail_lines = lines[-80:] if len(lines) > 80 else lines
        snippets = ["\n".join(tail_lines).strip()]

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
    # partitions: compounds/partition_*
    for part_dir in sorted(compounds_dir.glob("partition_*")):
        if not part_dir.is_dir():
            continue
        # molXXXX folders
        for mol_dir in sorted(d for d in part_dir.iterdir() if d.is_dir() and d.name.startswith("mol")):
            crest_dir = mol_dir / "crest"
            crest_out = crest_dir / "crest.out"
            if not crest_out.is_file():
                # treat missing crest.out as a failure entry too
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
            if SUCCESS_BANNER in txt:
                continue  # success

            # extract an error excerpt
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
        description="Collect CREST failures across partitions and write logs/failed_obabel.json"
    )
    ap.add_argument("--compounds_dir", default="compounds",
                    help="Root directory containing partition_* folders")
    ap.add_argument("--logs_dir", default="logs",
                    help="Where to write the JSON report (default: logs)")
    ap.add_argument("--outfile", default="failed_crest.json",
                    help="Output JSON filename inside logs_dir")
    args = ap.parse_args()

    compounds_dir = Path(args.compounds_dir).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    failures = scan_compounds(compounds_dir)
    out_path = logs_dir / args.outfile
    out_path.write_text(json.dumps(failures, indent=2))
    print(f"Wrote {len(failures)} failures → {out_path}")

if __name__ == "__main__":
    """Example usage:
    python filter_crest.py --compounds_dir compounds --logs_dir logs
    """

    main()

