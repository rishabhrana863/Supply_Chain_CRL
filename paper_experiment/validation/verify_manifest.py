"""Verify the validation package against MANIFEST.json.

Checks the SHA-256 of every file recorded as present, and reports which files
are still missing. When a missing file has been supplied, validates it against
the expected shape recorded in the manifest and records its checksum.

Usage:
    python verify_manifest.py             # check what is present
    python verify_manifest.py --record    # also record checksums for supplied files
"""
import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = HERE / "MANIFEST.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def open_maybe_gzip(path):
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else open(path, newline="")


def check_shape(path, entry):
    """Confirm a supplied stream file is the run the report describes."""
    import csv

    spec = entry["expected_shape"]
    expected = {(c["profile"], c["agent"]): c["dipped_streams"] for c in spec["cells"]}
    with open_maybe_gzip(path) as fh:
        reader = csv.DictReader(fh)
        missing_cols = [c for c in entry["required_columns"] if c not in (reader.fieldnames or [])]
        if missing_cols:
            return [f"missing required columns: {', '.join(missing_cols)}"]
        counts = Counter()
        for row in reader:
            if str(row.get("dipped", "")).strip().lower() in ("true", "1"):
                counts[(row.get("profile"), row.get("agent"))] += 1

    problems = []
    total = sum(counts.values())
    if total != spec["dipped_streams_total"]:
        problems.append(f"dipped rows: found {total}, expected {spec['dipped_streams_total']}")
    for key, want in expected.items():
        got = counts.get(key, 0)
        if got != want:
            problems.append(f"{key[0]} / {key[1]}: found {got} dipped streams, expected {want}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true",
                    help="record checksums for missing files that have since been supplied")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = 0

    for entry in manifest["present"]:
        path = HERE / entry["path"]
        if not path.exists():
            print(f"MISSING   {entry['path']}")
            failures += 1
            continue
        actual = sha256(path)
        if actual == entry["sha256"]:
            print(f"ok        {entry['path']}")
        else:
            print(f"CHANGED   {entry['path']}\n          recorded {entry['sha256']}\n          actual   {actual}")
            failures += 1

    for entry in manifest.get("missing", []):
        path = HERE / entry["path"]
        if not path.exists():
            print(f"ABSENT    {entry['path']} - {entry['status']}")
            failures += 1
            continue
        problems = check_shape(path, entry)
        if problems:
            print(f"MISMATCH  {entry['path']} is present but does not match the expected run:")
            for p in problems:
                print(f"          {p}")
            failures += 1
            continue
        digest = sha256(path)
        print(f"ok        {entry['path']} (shape verified, sha256 {digest})")
        if args.record:
            entry["sha256"] = digest
            entry["status"] = "supplied"
            manifest["present"].append({"path": entry["path"], "sha256": digest})
            manifest["missing"] = [m for m in manifest["missing"] if m["path"] != entry["path"]]
            MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"          recorded in {MANIFEST.name}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
