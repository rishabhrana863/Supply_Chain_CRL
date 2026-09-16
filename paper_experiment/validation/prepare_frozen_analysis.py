"""Verify and stage the frozen archive without changing its preserved bytes.

The original archive stores outputs at results/, while its frozen analysis
modules expect aor_experiment/results/. This adapter fixes only the working
copy's layout. With --run, the original analysis and sensitivity modules run
unchanged against that copy. Use a new --work directory for each invocation.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from frozen_archive import open_frozen_archive


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,default=Path(__file__).parent/'frozen/AOR_Reproducibility_Package_Prospective_Replication.zip')
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--run',action='store_true')
    args=ap.parse_args()
    if args.work.exists():raise FileExistsError('Use a new work directory to preserve any prior results')
    with open_frozen_archive(args.archive) as archive, zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            p=Path(entry.filename)
            if p.is_absolute() or '..' in p.parts:raise ValueError(f'Unsafe archive member: {p}')
        z.extractall(args.work)
    count=0
    for line in (args.work/'MANIFEST_SHA256.txt').read_text().splitlines():
        expected,relative=line.split(maxsplit=1)
        actual=hashlib.sha256((args.work/relative).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'Checksum mismatch: {relative}')
        count+=1
    print(f'Verified {count} frozen files.',flush=True)
    target=args.work/'aor_experiment/results'
    shutil.copytree(args.work/'results',target)
    print(f'Staged unchanged frozen code and a separate working results copy at {args.work}',flush=True)
    if args.run:
        subprocess.run([sys.executable,'-m','aor_experiment.prospective_extension_analysis','--repetitions','10000'],cwd=args.work,check=True)
        subprocess.run([sys.executable,'-m','aor_experiment.postrun_sensitivity'],cwd=args.work,check=True)


if __name__=='__main__':main()
