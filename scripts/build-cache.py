#!/usr/bin/env python3
"""Content keys and verified output receipts for generated build inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def hashes(paths):
    result = {}
    for path in paths:
        if not path.is_file(): raise FileNotFoundError(path)
        content = str(path.stat().st_mode & 0o777).encode() + b'\0' + path.read_bytes()
        if path.is_symlink(): content = os.readlink(path).encode() + b'\0' + content
        result[str(path)] = hashlib.sha256(content).hexdigest()
    return result


def key(paths, context=''):
    files = []
    for path in paths:
        files.extend(p for p in path.rglob('*') if p.is_file() and '.git' not in p.parts) if path.is_dir() else files.append(path)
    return hashlib.sha256(json.dumps([context, hashes(sorted(set(files)))], sort_keys=True).encode()).hexdigest()


def candidate_key(root):
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root).decode().split('\0')
    paths = [root / f for f in sorted(set(files)) if f and (root / f).is_file()]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    links = {f: os.readlink(root / f) for f in files if f and (root / f).is_symlink()}
    checkouts = {}
    for build in [root / '.build', root / 'Workers/.build']:
        for checkout in sorted((build / 'checkouts').glob('*')):
            if not checkout.is_dir(): continue
            revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
            # Ignored SwiftPM checkouts can be edited without making the app's tree dirty.
            # Hash their actual files too, so a dirty dependency never inherits the old candidate.
            checkouts[str(checkout)] = [revision, key([checkout])]
    return key(paths, json.dumps([commit, links, checkouts], sort_keys=True))


def metal_identity(developer):
    environment = {**os.environ, 'DEVELOPER_DIR': str(developer)}
    sdk_root = Path(developer) / 'Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk'
    if sdk_root.is_dir():  # Xcode's own SDK, as pkg-lib.sh's xcode_xcrun names it
        environment['SDKROOT'] = str(sdk_root)
    def xcrun(*args):
        return subprocess.check_output(['/usr/bin/xcrun', *args], env=environment, text=True, timeout=30).strip()
    sdk = Path(xcrun('--show-sdk-path'))
    tools = [Path(xcrun('--find', tool)) for tool in ('metal', 'metallib')]
    settings = [sdk / name for name in ('SDKSettings.json', 'SDKSettings.plist') if (sdk / name).is_file()]
    return key(tools + settings, json.dumps([str(developer), str(sdk), xcrun('--show-sdk-version'), xcrun('metal', '--version')]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['key', 'candidate', 'metal', 'check', 'record'])
    parser.add_argument('--context', default='')
    parser.add_argument('--key', default='')
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('paths', nargs='*', type=Path)
    args = parser.parse_intermixed_args()
    try:
        if args.operation == 'key': print(key(args.paths, args.context))
        elif args.operation == 'candidate': print(candidate_key(Path.cwd()))
        elif args.operation == 'metal': print(metal_identity(args.paths[0]))
        else:
            current = {'key': args.key, 'outputs': hashes(args.paths)}
            if args.operation == 'check': sys.exit(0 if json.loads(args.receipt.read_text()) == current else 1)
            else:
                args.receipt.parent.mkdir(parents=True, exist_ok=True)
                args.receipt.write_text(json.dumps(current, sort_keys=True) + '\n')
    except (OSError, ValueError):
        if args.operation == 'check': sys.exit(1)
        raise


if __name__ == '__main__': main()
