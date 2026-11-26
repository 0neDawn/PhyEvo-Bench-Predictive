"""Fetch exact OCVP dependencies from upstream; preserve the author's terms."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = '01d0d57adc48efca1e57685a14c96a43cac4be5e'
BASE = 'https://raw.githubusercontent.com/AIS-Bonn/OCVP-object-centric-video-prediction/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    files = [f for f in json.loads((ROOT / 'audit/source_manifest.json').read_text())
             if f['distribution'] == 'upstream_fetch']
    for item in files:
        dest = ROOT / item['path']
        if dest.exists():
            data = dest.read_bytes()
        elif args.verify_only:
            raise FileNotFoundError(dest)
        else:
            relative = item['path'].removeprefix('third_party/OCVP/')
            data = urlopen(BASE + REVISION + '/' + relative, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != item['archived_sha256']:
            raise RuntimeError('Content differs from the archived version: ' + str(dest))
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    print(f'Verified {len(files)} upstream OCVP files at {REVISION}.')


if __name__ == '__main__':
    main()
