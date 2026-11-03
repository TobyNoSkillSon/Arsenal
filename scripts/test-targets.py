#!/usr/bin/env python3
"""Run complete root target suites in isolated processes; keep AppKit in its own phase.

The AppKit target is split by top-level suite across ARSENAL_APP_SHARDS processes (default 3). Its menu-state tests
hold one process-wide gate (MenuStateIsolation), so one process runs them one at a time; separate processes each own
their AppKit state. Every shard runs under its own process name, so each has its own UserDefaults domain. The last
shard takes everything the others do not name (--skip), and the shards' counts must add up to the target's listed
count, so no test can fall between shards."""
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from incremental import consumers, fingerprint, package_graph

TARGETS = ['ArsenalCoreTests', 'ArsenalAPITests', 'ArsenalBenchTests', 'ArsenalAppTests', 'ArsenalRunnerCheckTests']
# The one native graph (Workers/Package.swift). Changes to these packages, or to anything not owned by one family,
# rerun every native suite; a family change reruns that family's suites, its consumers and the combined-worker suites.
NATIVE_SHARED = {'integrationkit', 'transport', 'kernels', 'arsenalwire', 'modelprecision'}
NATIVE_COMBINED = ['NativeModelContractTests', 'NativeMeasurementTests']
NATIVE_IGNORED = ('docs/', 'Sources/Arsenal/', 'Sources/ArsenalCommands/', 'Sources/ArsenalBench/', 'Sources/ArsenalUpdateHelper/', 'Sources/arsenal-',
                  'Tests/ArsenalAppTests/', 'Tests/ArsenalAPITests/', 'Tests/ArsenalCoreTests/', 'Tests/ArsenalBenchTests/', 'Tests/ArsenalRunnerCheckTests/')


def source_readers(root, paths, targets):
    """Tests may inspect source text without importing that source's Swift module."""
    selected = set()
    for target in targets:
        for file in (root / 'Tests' / target).rglob('*.swift'):
            # Match file and directory literals, including paths assembled below a directory URL.
            references = re.findall(r'"((?:Sources|Packages|Workers)/[^"\n]+)"', file.read_text())
            if any(path == ref or path.startswith(ref.rstrip('/') + '/') for path in paths for ref in references):
                selected.add(target); break
    return selected


def changed_targets(root, reference='HEAD', *, targets, dependencies):
    paths = set(subprocess.check_output(['git', 'diff', '--name-only', reference, '--'], cwd=root, text=True).splitlines())
    paths.update(subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=root, text=True).splitlines())
    affected = source_readers(root, paths, targets)
    for path in paths:
        parts = Path(path).parts
        if len(parts) >= 3 and parts[0] in ('Sources', 'Tests') and parts[1] in dependencies:
            affected.update({parts[1]} | consumers({parts[1]}, dependencies))
        elif path.startswith('Packages/ArsenalWire/'):
            return targets
        else:
            # Unknown/resource/script/catalog changes retain the complete contract suite.
            return targets
    selected = [target for target in targets if target in affected]
    return selected or targets


def listed_suites(lines, target):
    """Top-level suite -> listed test count for one target, from `swift test list` output."""
    suites = {}
    for line in lines:
        match = re.match(re.escape(target) + r'\.([A-Za-z_][A-Za-z0-9_]*)/', line)
        if match: suites[match.group(1)] = suites.get(match.group(1), 0) + 1
    return suites


def shard_suites(suites, weights, shards):
    """Longest-first greedy split of suites into at most `shards` groups by gate seconds (test count when unknown)."""
    known = [weights[name] for name in suites if name in weights]
    per_test = (sum(known) / sum(suites[name] for name in suites if name in weights)) if known and sum(known) > 0 else 1.0
    cost = {name: weights.get(name, suites[name] * per_test) for name in suites}
    groups = [[] for _ in range(max(1, min(shards, len(suites))))]
    totals = [0.0] * len(groups)
    for name in sorted(suites, key=lambda name: (-cost[name], name)):
        index = totals.index(min(totals))
        groups[index].append(name); totals[index] += cost[name]
    return groups


def shard_filters(target, groups):
    """Filter/skip arguments per shard. The last shard runs the whole target minus every other shard's suites, so a
    suite or free function the list parser missed still runs exactly once."""
    patterns = [re.escape(target) + r'\.(?:' + '|'.join(re.escape(name) for name in sorted(group)) + ')/' for group in groups[:-1]]
    filters = [['--filter', pattern] for pattern in patterns]
    last = ['--filter', target]
    for pattern in patterns: last += ['--skip', pattern]
    return filters + [last]


def target_stats(repo, target):
    """(heavy, light) @Test counts in the source directory named after the target; None when no directory matches."""
    for directory, names, _ in os.walk(repo / 'Workers', followlinks=True):
        if '.build' in directory.split(os.sep): names[:] = []; continue
        if os.path.basename(directory) != target: continue
        heavy = light = 0
        for file in Path(directory).rglob('*.swift'):
            for line in file.read_text(errors='replace').splitlines():
                if re.match(r'\s*@Test\b', line):
                    if '.heavy' in line: heavy += 1
                    else: light += 1
        return heavy, light
    return None


def native_graph(repo):
    """Test targets of Workers/Package.swift with the repository paths each one depends on."""
    workers = repo / 'Workers'
    swift = Path('/Library/Developer/CommandLineTools/usr/bin/swift')
    key = fingerprint([workers / 'Package.swift'], str(swift.stat().st_mtime_ns))
    receipt = workers / '.build/arsenal-native-graph.json'
    try:
        cached = json.loads(receipt.read_text())
        if cached['key'] == key: return cached
    except (OSError, ValueError, KeyError): pass
    environment = {**os.environ, 'DEVELOPER_DIR': '/Library/Developer/CommandLineTools'}
    package = json.loads(subprocess.check_output([str(swift), 'package', '--package-path', str(workers), 'dump-package'], env=environment, text=True, timeout=60))
    packages = {}
    for dependency in package['dependencies']:
        for kind, values in dependency.items():
            if kind != 'fileSystem': continue
            value = values[0]; path = Path(value['path'])
            prefixes = sorted({os.path.relpath(path, repo), os.path.relpath(path.resolve(), repo)} - {'.'})
            if not prefixes or any(prefix.startswith('..') for prefix in prefixes): continue
            for name in {value.get('identity'), value.get('nameForTargetDependencyResolutionOnly')} - {None}:
                packages[name.lower()] = prefixes
    tests = []
    for target in package['targets']:
        if target['type'] != 'test': continue
        used = set()
        for dependency in target.get('dependencies', []):
            if 'product' in dependency and dependency['product'][1]: used.add(dependency['product'][1].lower())
        tests.append({'name': target['name'], 'packages': sorted(used)})
    cached = {'key': key, 'packages': packages, 'tests': tests}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(cached, sort_keys=True) + '\n')
    return cached


def native_changed_targets(root, metadata, reference='HEAD'):
    names = [test['name'] for test in metadata['tests']]
    paths = set(subprocess.check_output(['git', 'diff', '--name-only', reference, '--'], cwd=root, text=True).splitlines())
    paths.update(subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=root, text=True).splitlines())
    selected = set()
    for path in paths:
        if path.endswith('.md') or path.startswith(NATIVE_IGNORED): continue
        owner = next((name for name, prefixes in metadata['packages'].items() if any(path.startswith(prefix + '/') for prefix in prefixes)), None)
        if owner is None or owner in NATIVE_SHARED: return names
        selected.update(test['name'] for test in metadata['tests'] if owner in test['packages'])
        selected.update(NATIVE_COMBINED)
    return [name for name in names if name in selected]


def main():
    argv = sys.argv[1:]
    native = argv[:1] == ['--native']
    if native: argv.pop(0)
    affected_only = argv[:1] == ['--affected']
    if affected_only: argv.pop(0)
    filtered = bool(argv and argv[0] == '--filtered')
    if filtered: argv.pop(0)
    repo = Path(__file__).resolve().parent.parent
    if native:
        metadata = native_graph(repo)
        targets = [test['name'] for test in metadata['tests']]
        app_targets = set()
    else:
        metadata = package_graph(repo)
        targets = sorted(metadata['tests'], key=lambda target: TARGETS.index(target) if target in TARGETS else len(TARGETS))
        app_targets = set(targets).intersection(consumers({'Arsenal'}, metadata['dependencies']))
    if not targets: sys.exit('No test targets declared')
    if native:
        # Heavy tests (`.heavy`) are opt-in: default runs skip heavy-only targets, --heavy runs only targets that have heavy tests.
        heavy_mode = os.environ.get('ARSENAL_HEAVY_TESTS') == '1'
        counts = {t: target_stats(repo, t) for t in targets}
        targets = [t for t in targets if counts[t] is None or (counts[t][0] > 0 if heavy_mode else counts[t][1] > 0)]
    if affected_only:
        # Native selection only: an empty list means no native suite is affected and nothing needs building.
        print('\n'.join(native_changed_targets(Path.cwd(), metadata) if native else changed_targets(Path.cwd(), targets=targets, dependencies=metadata['dependencies'])))
        return
    if argv and argv[0] == '--changed':
        argv.pop(0)
        targets = native_changed_targets(Path.cwd(), metadata) if native else changed_targets(Path.cwd(), targets=targets, dependencies=metadata['dependencies'])
        if not targets: print('Tests: no native suite affected', flush=True); return
    if not argv or argv[0] != '--': sys.exit('usage: test-targets.py [--native] [--affected] [--changed|--filtered] -- swift-test-command...')
    command = argv[1:]
    results_dir = Path('.build/test-results'); results_dir.mkdir(parents=True, exist_ok=True)
    # Per-target logs go to ARSENAL_TEST_LOG_DIR when set (the entry scripts give each run its own directory).
    out = Path(os.environ.get('ARSENAL_TEST_LOG_DIR') or results_dir); out.mkdir(parents=True, exist_ok=True)
    prefix = 'native-' if native else ''
    results = []
    def run(target, selection=None, name=None, executable=None, environment=None):
        start = time.monotonic()
        log = out / (prefix + (name or target) + '.log')
        arguments = [executable or command[0]] + command[1:] + ([] if filtered else (selection or ['--filter', target]))
        with log.open('w') as handle:
            process = subprocess.run(arguments, stdout=handle, stderr=subprocess.STDOUT, env=environment)
        output = log.read_text(errors='replace')
        print('Tests: ' + (name or target), flush=True)
        summaries = re.findall(r'^.*Test run with .*$', output, re.MULTILINE)
        print('\n'.join(summaries), flush=True)
        count = re.findall(r'Test run with (\d+) tests?\b', output)
        passed = process.returncode == 0 and bool(count) and int(count[-1]) > 0
        if not passed:
            print('Tests: failed or zero tests matched: ' + (name or target) + '\n' + output, file=sys.stderr)
        return {'target': target, 'seconds': round(time.monotonic() - start, 3), 'tests': int(count[-1]) if count else 0,
                'exit': process.returncode, 'passed': passed,
                'slowest': sorted([(float(seconds), name) for name, seconds in re.findall(r'Test (?!run with)(.+?) passed after ([\d.]+) seconds', output)], reverse=True)[:20]}

    def run_sharded(target):
        """One AppKit target across processes; one result whose count is the shards' sum, checked against the list."""
        listing = Path(os.environ.get('ARSENAL_TEST_LIST', '.build/test-list.txt'))
        lines = listing.read_text(errors='replace').splitlines() if listing.is_file() else []
        suites = listed_suites(lines, target)
        listed = sum(1 for line in lines if line.startswith(target + '.'))
        shards = int(os.environ.get('ARSENAL_APP_SHARDS', '3'))
        if shards <= 1 or len(suites) < 2 or filtered: return run(target)
        weights_file = results_dir / (target + '-gate-seconds.json')
        try: weights = json.loads(weights_file.read_text())
        except (OSError, ValueError): weights = {}
        groups = shard_suites(suites, weights, shards)
        selections = shard_filters(target, groups)
        helpers = results_dir / 'bin'; helpers.mkdir(exist_ok=True)
        def shard(index):
            # A distinct process name gives the shard its own UserDefaults domain (window frames, sidebar, columns).
            link = helpers / ('arsenal-app-tests-' + str(index + 1))
            if link.is_symlink() or link.exists(): link.unlink()
            link.symlink_to(command[0])
            times = out / (prefix + target + '-' + str(index + 1) + '.gate.tsv')
            times.unlink(missing_ok=True)
            environment = {**os.environ, 'ARSENAL_TEST_GATE_TIMES': str(times.resolve())}
            result = run(target, selections[index], target + '-' + str(index + 1), str(link), environment)
            result['suites'] = 'rest' if index == len(groups) - 1 else sorted(groups[index])
            return result, times
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as pool:
            parts = list(pool.map(shard, range(len(groups))))
        total = sum(part['tests'] for part, _ in parts)
        complete = listed == 0 or total == listed
        if not complete:
            print(f'Tests: {target} shards ran {total} tests, the list has {listed}', file=sys.stderr)
        # Learn each suite's gate seconds for the next split; a failed shard's partial times are still a fair estimate.
        observed = {}
        for _, times in parts:
            try:
                for line in times.read_text().splitlines():
                    suite, _, seconds = line.partition('\t')
                    if suite and seconds: observed[suite] = observed.get(suite, 0.0) + float(seconds)
            except (OSError, ValueError): pass
        if observed: weights_file.write_text(json.dumps({**weights, **{k: round(v, 3) for k, v in observed.items()}}, indent=2, sort_keys=True) + '\n')
        return {'target': target, 'seconds': round(time.monotonic() - start, 3), 'tests': total, 'listed': listed,
                'exit': max(part['exit'] for part, _ in parts), 'passed': complete and all(part['passed'] for part, _ in parts),
                'shards': [{key: part[key] for key in ('seconds', 'tests', 'exit', 'passed', 'suites')} for part, _ in parts],
                'slowest': sorted((item for part, _ in parts for item in part['slowest']), reverse=True)[:20]}

    start = time.monotonic()
    workers = int(os.environ.get('ARSENAL_TEST_TARGET_JOBS', '4'))
    if workers < 1: sys.exit('ARSENAL_TEST_TARGET_JOBS must be positive')
    if filtered:
        results.append(run('filtered'))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results.extend(pool.map(run, [t for t in targets if t not in app_targets]))
        for target in targets:
            if target in app_targets: results.append(run_sharded(target))
    summary = json.dumps({'seconds': round(time.monotonic() - start, 3), 'results': results}, indent=2) + '\n'
    name = prefix + ('filtered.json' if filtered else 'last.json')
    (results_dir / name).write_text(summary)
    if out != results_dir: (out / name).write_text(summary)
    sys.exit(0 if all(item['passed'] for item in results) else 1)

if __name__ == '__main__': main()
