"""Verify and restore split checkpoints/data without deserializing any model."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 4 * 1024 * 1024


def safe_path(root, relative):
    root = root.resolve()
    target = (root / relative).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f'Unsafe asset path: {relative}')
    return target


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(BLOCK):
            digest.update(data)
    return digest.hexdigest()


def chunks(asset, directory):
    full = hashlib.sha256()
    total = 0
    for part in asset['parts']:
        path = safe_path(directory, part['file'])
        digest, count = hashlib.sha256(), 0
        with path.open('rb') as stream:
            while data := stream.read(BLOCK):
                full.update(data); digest.update(data)
                total += len(data); count += len(data)
                yield data
        if count != part['bytes'] or digest.hexdigest() != part['sha256']:
            raise RuntimeError('Asset chunk failed verification: ' + str(path))
    if total != asset['bytes'] or full.hexdigest() != asset['sha256']:
        raise RuntimeError('Full asset failed verification: ' + asset['id'])


class JoinedParts(io.RawIOBase):
    def __init__(self, values):
        self.values, self.pending = iter(values), memoryview(b'')

    def readable(self):
        return True

    def readinto(self, buffer):
        if not self.pending:
            try:
                self.pending = memoryview(next(self.values))
            except StopIteration:
                return 0
        n = min(len(buffer), len(self.pending))
        buffer[:n] = self.pending[:n]
        self.pending = self.pending[n:]
        return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset_directory', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--only', nargs='+', help='Asset IDs, e.g. data_sp weight_00')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.asset_directory / 'assets_manifest.json').read_text())
    available = {a['id'] for a in manifest['assets']}
    if args.only and not set(args.only) <= available:
        parser.error('Unknown asset IDs: ' + str(set(args.only) - available))
    selected = [a for a in manifest['assets'] if not args.only or a['id'] in args.only]
    for asset in selected:
        # Validate every byte before writing extracted files.
        for _ in chunks(asset, args.asset_directory):
            pass
        print('SHA256 OK', asset['id'], asset['bytes'], flush=True)
        if args.verify_only:
            continue
        if asset['kind'] == 'file':
            dest = safe_path(args.root, asset['target'])
            if dest.exists():
                if file_hash(dest) != asset['sha256']:
                    raise FileExistsError('Different existing file; refusing to overwrite: ' + str(dest))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_name(dest.name + '.restore-partial')
            with temp.open('wb') as out:
                for data in chunks(asset, args.asset_directory):
                    out.write(data)
            temp.replace(dest)
        elif asset['kind'] == 'tar':
            stream = io.BufferedReader(JoinedParts(chunks(asset, args.asset_directory)))
            count = 0
            with tarfile.open(fileobj=stream, mode='r|*') as tar:
                for member in tar:
                    if not member.isfile():
                        raise ValueError('Only regular files are supported: ' + member.name)
                    dest = safe_path(args.root, member.name)
                    incoming = tar.extractfile(member)
                    if dest.exists():
                        digest = hashlib.sha256()
                        while data := incoming.read(BLOCK):
                            digest.update(data)
                        if file_hash(dest) != digest.hexdigest():
                            raise FileExistsError('Different existing data: ' + str(dest))
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        temp = dest.with_name(dest.name + '.restore-partial')
                        with temp.open('wb') as out:
                            shutil.copyfileobj(incoming, out, BLOCK)
                        temp.replace(dest)
                    count += 1
            while stream.read(BLOCK):
                pass
            if count != asset['files']:
                raise RuntimeError('Unexpected data file count')
        else:
            raise ValueError(asset['kind'])
    print(f'Completed {len(selected)} assets; no neural networks loaded.')


if __name__ == '__main__':
    main()
