#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Remove task-owned bundle registrations and guard the production app identity."""
import argparse
import pathlib
import re
import subprocess
import time

LSREGISTER = '/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister'
PRODUCTION_ID = 'dev.arsenal.app'
INSTALLED = pathlib.Path.home() / 'Applications/Arsenal.app'


def registrations(dump):
    record = {}
    for line in dump.splitlines():
        if line.startswith('bundle id:'):
            if 'path' in record and 'identifier' in record:
                yield record
            record = {}
        match = re.match(r'^(path|identifier):\s+(.+?)\s*$', line)
        if match:
            record[match[1]] = re.sub(r' \(0x[0-9a-fA-F]+\)$', '', match[2])
    if 'path' in record and 'identifier' in record:
        yield record


def dump_registrations():
    return list(registrations(subprocess.run(
        [LSREGISTER, '-dump'], check=True, capture_output=True, text=True, timeout=30).stdout))


def unregister(paths):
    paths = sorted(set(map(pathlib.Path, paths)))
    for path in paths:
        if path.resolve() == INSTALLED.resolve() or INSTALLED.resolve() in path.resolve().parents:
            raise RuntimeError('Refusing to unregister the installed Arsenal: ' + str(path))
    diagnostics = []
    # -u can remove a stale record but report a scan error for an absent/renamed non-.app path.
    # The database, rather than that scan status, decides whether unregistering succeeded.
    pending = paths
    owned = {p.resolve() for p in paths}
    for attempt in range(3):
        if not pending:
            return
        for path in pending:
            result = subprocess.run([LSREGISTER, '-u', str(path)], capture_output=True, text=True, timeout=15)
            if result.returncode:
                diagnostics.append(result.stdout.strip() + result.stderr.strip())
        pending = sorted({pathlib.Path(r['path']) for r in dump_registrations() if pathlib.Path(r['path']).resolve() in owned})
        if pending and attempt < 2:
            # AppKit/LaunchServices can finish an outstanding registration after the first -u returns.
            time.sleep(1)
    if pending:
        raise RuntimeError('Bundle registrations remain: ' + ', '.join(map(str, pending)) + '; ' + '; '.join(diagnostics))


def cleanup_tree(root):
    root = pathlib.Path(root).resolve()
    # Include renamed backups and registrations whose files have already been removed by the updater.
    paths = {p.parent.parent for p in root.rglob('Contents/Info.plist')}
    paths.update(pathlib.Path(r['path']) for r in dump_registrations()
                 if root in pathlib.Path(r['path']).resolve().parents)
    unregister(paths)


def check(dump):
    paths = [r['path'] for r in registrations(dump) if r['identifier'] == PRODUCTION_ID]
    unexpected = [p for p in paths if pathlib.Path(p) != INSTALLED]
    if unexpected:
        raise RuntimeError('Unexpected production Arsenal registrations:\n' + '\n'.join(unexpected))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'unregister', 'cleanup-tree'])
    parser.add_argument('paths', nargs='*', type=pathlib.Path)
    args = parser.parse_args()
    if args.action == 'check':
        if args.paths:
            parser.error('check takes no paths')
        dump = subprocess.run([LSREGISTER, '-dump'], check=True, capture_output=True, text=True, timeout=30).stdout
        paths = check(dump)
        print(f'LaunchServices: {len(paths)} production registration(s); only {INSTALLED} allowed')
    else:
        if not args.paths:
            parser.error('cleanup requires explicit task-owned paths')
        if args.action == 'unregister':
            unregister(args.paths)
        else:
            for root in args.paths:
                cleanup_tree(root)


if __name__ == '__main__':
    main()
