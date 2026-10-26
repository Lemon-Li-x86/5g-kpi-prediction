"""Read-only delivery checks, or explicit restoration of quarantined bytecode.

Inventory mode never modifies project files. --restore-quarantine performs a
preflight check on every entry before moving any files back into the project.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restore(root, manifest, quarantine_root):
    quarantine_root = quarantine_root.resolve(strict=True)
    entries = list(csv.DictReader(manifest.open(encoding='utf-8-sig', newline='')))
    planned = []
    for row in entries:
        relative = Path(row['path'])
        destination = (root / relative).resolve()
        source = (quarantine_root / relative).resolve()
        if (relative.is_absolute() or not destination.is_relative_to(root)
                or not source.is_relative_to(quarantine_root)):
            raise ValueError(f'Unsafe manifest path: {relative}')
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite: {destination}')
        if not source.is_file() or source.is_symlink() or sha256(source) != row['sha256']:
            raise ValueError(f'Missing or changed quarantine entry: {relative}')
        planned.append((source, destination))
    for source, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    print(f'Restored {len(planned)} files; nothing overwritten.')


def inventory(root, output):
    output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Inventory output must be outside the project.')
    if output.exists():
        raise FileExistsError(output)
    files = []
    for base, dirs, names in os.walk(root, followlinks=False):
        # Environment and Git internals are covered by the original full snapshot.
        if Path(base) == root:
            dirs[:] = [d for d in dirs if d not in ('.git', '.venv')]
        for name in names:
            path = Path(base) / name
            if path.is_symlink():
                raise ValueError(f'Symlink requires manual inspection: {path}')
            files.append(dict(path=path.relative_to(root).as_posix(),
                              size=path.stat().st_size, sha256=sha256(path)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(files, stream, ensure_ascii=False, indent=2)
    print(f'Inventoried {len(files)} files, {sum(f["size"] for f in files)} bytes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--restore-quarantine', type=Path, metavar='MANIFEST_CSV')
    parser.add_argument('--quarantine-root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.restore_quarantine:
        if not args.quarantine_root or args.output:
            parser.error('Restoration requires --quarantine-root and no --output.')
        restore(root, args.restore_quarantine, args.quarantine_root)
    else:
        if not args.output or args.quarantine_root:
            parser.error('Inventory requires --output outside the project.')
        inventory(root, args.output)


if __name__ == '__main__':
    main()
