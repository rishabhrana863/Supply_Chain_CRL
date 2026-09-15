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
from collections import Counter, defaultdict
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
    expected = {(c["profile"], c["agent_key"]): c for c in spec["cells"]}
    counts = Counter()
    recovered_by_110 = Counter()
    censored = Counter()
    observations = defaultdict(list)
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
        seed_field = "model_seed" if "model_seed" in fields else "seed" if "seed" in fields else None
        if "seeds" in spec and seed_field is None:
            return ["missing seed column: expected 'model_seed' or 'seed'",
                    f"columns present: {', '.join(fields) or '(none)'}"]
        for row in reader:
            total_rows += 1
            key = (row.get("profile"), row.get("agent"))
            seen[key] += 1
            if seed_field:
                seeds.add(row.get(seed_field))
            if str(row.get("dipped", "")).strip().lower() in ("true", "1"):
                counts[key] += 1
                try:
                    duration = float(row["recovery_days"])
                except (TypeError, ValueError):
                    return [f"non-numeric recovery_days in {key}: {row.get('recovery_days')!r}"]
                event = str(row.get("recovered", "")).strip().lower() in ("true", "1")
                observations[key].append((duration, event))
                if event and duration <= 110:
                    recovered_by_110[key] += 1
                if not event:
                    censored[key] += 1

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
    if "seeds" in spec:
        try:
            numeric_seeds = {int(float(s)) for s in seeds}
        except (TypeError, ValueError):
            problems.append(f"seed values are not numeric: {sorted(seeds)[:5]}")
        else:
            seed_spec = spec["seeds"]
            if len(numeric_seeds) != seed_spec["count"]:
                problems.append(f"distinct seeds: found {len(numeric_seeds)}, expected {seed_spec['count']}")
            if numeric_seeds and min(numeric_seeds) != seed_spec["min"]:
                problems.append(f"minimum seed: found {min(numeric_seeds)}, expected {seed_spec['min']}")
            if numeric_seeds and max(numeric_seeds) != seed_spec["max"]:
                problems.append(f"maximum seed: found {max(numeric_seeds)}, expected {seed_spec['max']}")

    def km_median(items):
        by_time = defaultdict(lambda: [0, 0])
        for duration, event in items:
            by_time[duration][0 if event else 1] += 1
        at_risk = len(items)
        survival = 1.0
        for duration in sorted(by_time):
            events, withdrawals = by_time[duration]
            if events:
                survival *= 1.0 - events / at_risk
                if survival <= 0.5:
                    return duration
            at_risk -= events + withdrawals
        return None

    for key, want in sorted(expected.items()):
        got = counts.get(key, 0)
        if got != want["dipped_streams"]:
            problems.append(
                f"{key[0]} / {key[1]}: found {got} dipped streams, "
                f"expected {want['dipped_streams']}"
            )
        got_recovered = recovered_by_110.get(key, 0)
        if got_recovered != want["recovered_by_110"]:
            problems.append(
                f"{key[0]} / {key[1]}: found {got_recovered} recovered by day 110, "
                f"expected {want['recovered_by_110']}"
            )
        got_censored = censored.get(key, 0)
        if got_censored != want["censored"]:
            problems.append(
                f"{key[0]} / {key[1]}: found {got_censored} censored, "
                f"expected {want['censored']}"
            )
        got_median = km_median(observations.get(key, []))
        if got_median != want["km_median_days"]:
            problems.append(
                f"{key[0]} / {key[1]}: found KM median {got_median}, "
                f"expected {want['km_median_days']}"
            )
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
        if entry.get("expected_shape"):
            problems = check_shape(path, entry)
            if problems:
                print(f"MISMATCH  {entry['path']} does not match the expected run:")
                for problem in problems:
                    print(f"          {problem}")
                failures += 1
                continue
        actual = sha256(path)
        if actual == entry["sha256"]:
            suffix = " (shape verified)" if entry.get("expected_shape") else ""
            print(f"ok        {entry['path']}{suffix}")
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
            recorded_entry = {key: value for key, value in entry.items() if key != "status"}
            recorded_entry["sha256"] = digest
            manifest["present"].append(recorded_entry)
            manifest["missing"] = [m for m in manifest["missing"] if m["path"] != entry["path"]]
            MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"          recorded in {MANIFEST.name}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
