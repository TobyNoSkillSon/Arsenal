#!/usr/bin/env python3
"""Invalidate local Swift ABI consumers while keeping dependencies and producer builds."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

def fingerprint(paths, extra=''):
    digest = hashlib.sha256(extra.encode())
    for path in sorted(paths):
        digest.update(str(path).encode()); digest.update(b'\0'); digest.update(path.read_bytes())
    return digest.hexdigest()


def package_graph(root):
    """Read SwiftPM's declared edges; cache manifest evaluation between iterations."""
    manifests = [root / 'Package.swift', root / 'Packages/ArsenalWire/Package.swift']
    sdk_manifest = root / 'Packages/swift-sdk/Package.swift'
    if sdk_manifest.exists(): manifests.append(sdk_manifest)
    swift = Path('/Library/Developer/CommandLineTools/usr/bin/swift')
    manifest_key = fingerprint(manifests, str(swift.stat().st_mtime_ns))
    receipt = root / '.build/arsenal-target-graph.json'
    try:
        cached = json.loads(receipt.read_text())
        if cached['key'] == manifest_key and 'package_name' in cached: return cached
    except (OSError, ValueError, KeyError): pass
    environment = {**os.environ, 'DEVELOPER_DIR': '/Library/Developer/CommandLineTools'}
    package = json.loads(subprocess.check_output(
        ['/Library/Developer/CommandLineTools/usr/bin/swift', 'package', '--package-path', str(root), 'dump-package'],
        env=environment, text=True, timeout=30))
    graph = {target['name']: [] for target in package['targets']}
    folders = {}
    tests = []
    for name, folder in [('ArsenalWire', 'Packages/ArsenalWire/Sources/ArsenalWire'), ('MCP', 'Packages/swift-sdk/Sources/MCP')]:
        if (root / folder).exists(): graph[name] = []; folders[name] = folder
    for target in package['targets']:
        name = target['name']
        folders[name] = target.get('path') or ('Tests/' if target['type'] == 'test' else 'Sources/') + name
        if target['type'] == 'test': tests.append(name)
        for dependency in target.get('dependencies', []):
            dependency_name = next(iter(dependency.values()))[0]
            if dependency_name in graph: graph[name].append(dependency_name)
    graph['ArsenalPackageTests'] = tests
    cached = {'key': manifest_key, 'dependencies': graph, 'folders': folders, 'tests': tests, 'package_name': package['name']}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(cached, sort_keys=True) + '\n')
    return cached


def snapshot(root, metadata=None):
    metadata = metadata or package_graph(root)
    state = {'manifest': metadata['key']}
    for target, folder in metadata['folders'].items():
        folder = root / folder
        if folder.exists(): state[target] = fingerprint(folder.rglob('*.swift'))
    return state


def consumers(changed, dependencies):
    affected = set(changed)
    while True:
        added = {name for name, deps in dependencies.items() if affected.intersection(deps)} - affected
        if not added: break
        affected.update(added)
    # Producers retain incremental Swift dependency data. Any producer which is also a consumer must rebuild.
    return affected - set(changed) | {name for name in changed if any(dep in affected for dep in dependencies[name])}


def prepare(root, scratch, configuration):
    metadata = package_graph(root)
    dependencies = metadata['dependencies']
    current = snapshot(root, metadata)
    receipt = scratch / ('arsenal-abi-' + configuration + '.json')
    try: previous = json.loads(receipt.read_text())
    except (OSError, ValueError): previous = None
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / ('arsenal-abi-' + configuration + '.pending.json')).write_text(json.dumps(current, sort_keys=True) + '\n')
    if previous == current: return current
    # On first adoption invalidate local outputs too: an existing vanilla Swift build has no ABI receipt.
    changed = set(dependencies) if previous is None or previous.get('manifest') != current['manifest'] else {
        name for name in dependencies if previous.get(name) != current.get(name)
    }
    affected = set(dependencies) if previous is None or 'manifest' in changed else consumers(changed, dependencies)
    if previous is not None and previous.get('manifest') != current['manifest']: affected = set(dependencies)
    # Native SwiftPM output names replace executable hyphens with underscores for the module only.
    for output in scratch.glob('*/' + configuration):
        if output.is_symlink(): continue
        for target in affected:
            build = output / (target.replace('-', '_') + '.build')
            if build.is_dir():
                # SwiftPM's cached plan references its output-file map: preserve metadata, remove only compiled artifacts.
                for file in build.rglob('*'):
                    if file.is_file() and file.suffix in ('.o', '.swiftdeps', '.priors'): file.unlink()
            for file in (output / 'Modules').glob(target.replace('-', '_') + '.*'):
                if file.is_file(): file.unlink()
    if affected: print('incremental: refreshing ABI consumers: ' + ', '.join(sorted(affected)), flush=True)
    return current


def record(root, scratch, configuration):
    receipt = scratch / ('arsenal-abi-' + configuration + '.json')
    pending = scratch / ('arsenal-abi-' + configuration + '.pending.json')
    before = json.loads(pending.read_text())
    if before != snapshot(root):
        receipt.unlink(missing_ok=True)
        pending.unlink(missing_ok=True)
        raise RuntimeError('Sources changed during compilation; rebuild before using these outputs')
    receipt.write_text(json.dumps(before, sort_keys=True) + '\n')
    pending.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'record'])
    parser.add_argument('scratch', type=Path)
    parser.add_argument('configuration')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.operation == 'prepare': prepare(root, args.scratch, args.configuration)
    else:
        try: record(root, args.scratch, args.configuration)
        except (OSError, ValueError, RuntimeError) as error: parser.exit(1, 'incremental: ' + str(error) + '\n')


if __name__ == '__main__': main()
