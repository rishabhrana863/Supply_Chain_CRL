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
    """Confirm a supplied stream file is the run the report describes.

    Matches on the agent *key* stored in the file (the agent class `name`
    attribute, e.g. "crl"/"ppo"), not the workbook's display labels. When
    nothing matches, reports the values actually present so a naming-convention
    mismatch is distinguishable from a genuinely wrong dataset.
    """
    import csv

    spec = entry["expected_shape"]
    expected = {(c["profile"], c["agent_key"]): c["dipped_streams"] for c in spec["cells"]}
    counts = Counter()
    seen = Counter()
    seeds = set()
    total_rows = 0

    with open_maybe_gzip(path) as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        missing_cols = [c for c in entry["required_columns"] if c not in fields]
        if missing_cols:
            return [f"missing required columns: {', '.join(missing_cols)}",
                    f"columns present: {', '.join(fields) or '(none)'}"]
        for row in reader:
            total_rows += 1
            seen[(row.get("profile"), row.get("agent"))] += 1
            if "seed" in fields:
                seeds.add(row.get("seed"))
            if str(row.get("dipped", "")).strip().lower() in ("true", "1"):
                counts[(row.get("profile"), row.get("agent"))] += 1

    problems = []
    if not set(counts) & set(expected):
        problems.append("no (profile, agent) pair in the file matches any expected pair")
        problems.append(f"expected agents: {sorted({k[1] for k in expected})}")
        problems.append(f"found agents:    {sorted({k[1] for k in seen if k[1] is not None})}")
        problems.append(f"expected profiles: {sorted({k[0] for k in expected})}")
        problems.append(f"found profiles:    {sorted({k[0] for k in seen if k[0] is not None})}")
        problems.append("if the agent labels differ only by naming convention, fix MANIFEST.json"
                        " rather than the data file")
        return problems

    dipped_total = sum(counts.values())
    if dipped_total != spec["dipped_streams_total"]:
        problems.append(f"dipped rows: found {dipped_total}, expected {spec['dipped_streams_total']}")
    if "total_rows" in spec and total_rows != spec["total_rows"]:
        problems.append(f"total rows: found {total_rows}, expected {spec['total_rows']}")
    if seeds and "seeds" in spec and len(seeds) != spec["seeds"]["count"]:
        problems.append(f"distinct seeds: found {len(seeds)}, expected {spec['seeds']['count']}")
    for key, want in sorted(expected.items()):
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
        recorded = entry.get("expected_sha256_gzip")
        if recorded and recorded != digest:
            print(f"          note: differs from the recorded gzip sha256 {recorded}.")
            print( "          gzip output depends on compression level and stored mtime, so a")
            print( "          re-compressed copy of the same CSV differs here. The shape check")
            print( "          above is the authoritative test of which run this is.")
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
