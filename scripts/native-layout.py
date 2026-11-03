#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Ship only the linked product's resources and verify the single-runtime layout."""
import argparse
import json
from pathlib import Path
import re
import sys


def resource_bundles(binary_directory, product='arsenal-worker-native'):
    """SwiftPM can leave test/other-product bundles in the same build directory."""
    description = json.loads((binary_directory / 'description.json').read_text())
    module = product.replace('-', '_')
    graph = description['targetDependencyMap']
    if module not in graph:
        raise ValueError(f'Missing SwiftPM dependency closure for {product}')
    closure = set(graph[module]) | {module}
    bundles = set()
    for command in description['swiftCommands'].values():
        if command['moduleName'] not in closure:
            continue
        for source in command['sources']:
            if Path(source).name != 'resource_bundle_accessor.swift':
                continue
            names = re.findall(r'appendingPathComponent\("([^"/]+\.bundle)"\)', Path(source).read_text())
            if len(names) != 1:
                raise ValueError(f'Unrecognized SwiftPM resource accessor: {source}')
            name = names[0]
            if name == 'mlx-swift_Cmlx.bundle':
                continue  # assemble.sh installs the single locally compiled Metal library.
            bundle = binary_directory / name
            if not bundle.is_dir() or bundle.is_symlink():
                raise ValueError(f'Missing linked resource bundle: {bundle}')
            bundles.add(bundle)
    return sorted(bundles)


def compact_resources(directory):
    """Remove JSON formatting bytes only; token strings and numerical values survive."""
    saved = 0
    for path in sorted(directory.rglob('*.json')):
        if path.is_symlink() or not path.is_file():
            continue
        original = path.read_bytes()
        # Decimal floats round-trip as their original tokens, without numeric conversion.
        tokens = re.findall(rb'"(?:[^"\\]|\\.)*"|[^\s]', original)
        compact = b''.join(tokens)
        if len(compact) < len(original):
            json.loads(compact)  # Fail on unexpected non-JSON resources before writing.
            path.write_bytes(compact)
            saved += len(original) - len(compact)
    return saved


def inventory(app, families):
    helpers = app / 'Contents/Helpers'
    native = helpers / 'arsenal-worker-native'
    if native.is_symlink() or not native.is_file():
        raise ValueError('Missing single native executable')
    expected = {'arsenal-worker-' + name for name in families}
    for name in expected:
        link = helpers / name
        if not link.is_symlink() or link.readlink() != Path('arsenal-worker-native'):
            raise ValueError(f'{name} must be a relative link to arsenal-worker-native')
    for path in helpers.glob('arsenal-worker-*'):
        if path != native and (path.name not in expected or not path.is_symlink()):
            raise ValueError(f'Unexpected separate worker: {path.name}')
    metal = list(helpers.rglob('default.metallib'))
    expected_metal = helpers / 'mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib'
    if metal != [expected_metal] or not expected_metal.is_file() or expected_metal.is_symlink():
        raise ValueError('Expected exactly one MLX Metal library')
    if any('mlx' in path.name.lower() for path in helpers.rglob('*.dylib')):
        raise ValueError('Unexpected second MLX runtime')
    groups = {'native': 0, 'host_and_helpers': 0, 'metal': 0, 'worker_resources': 0, 'instructions_and_assets': 0}
    for path in app.rglob('*'):
        if path.is_symlink() or not path.is_file():
            continue
        if path == native:
            key = 'native'
        elif path == expected_metal:
            key = 'metal'
        elif path.parent == helpers or path.parent == app / 'Contents/MacOS':
            key = 'host_and_helpers'
        elif helpers in path.parents:
            key = 'worker_resources'
        else:
            key = 'instructions_and_assets'
        groups[key] += path.stat().st_size
    total = sum(groups.values())
    return {'bytes': total, 'mib': total / 1048576, 'budget_bytes': 45 * 1048576,
            'over_budget_bytes': max(0, total - 45 * 1048576), 'families': sorted(families), 'components_bytes': groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    resources = commands.add_parser('resources'); resources.add_argument('binary_directory', type=Path)
    compact = commands.add_parser('compact'); compact.add_argument('directory', type=Path)
    verify = commands.add_parser('verify'); verify.add_argument('app', type=Path); verify.add_argument('families', nargs='+')
    args = parser.parse_args()
    if args.command == 'resources':
        for path in resource_bundles(args.binary_directory):
            print(path)
    elif args.command == 'compact':
        print(f'pkg: resource JSON formatting removed: {compact_resources(args.directory)} bytes', file=sys.stderr)
    else:
        print(json.dumps(inventory(args.app, args.families), sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f'native layout: {error}')
