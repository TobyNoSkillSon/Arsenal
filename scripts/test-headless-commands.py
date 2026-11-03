#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offscreen command acceptance; optional parity with pre-fold helper binaries. No real weights."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arsenal', type=Path, required=True)
    parser.add_argument('--stub-worker', type=Path, required=True)
    parser.add_argument('--legacy-catalog', type=Path)
    parser.add_argument('--legacy-runner-check', type=Path)
    options = parser.parse_args()
    app, stub = options.arsenal.resolve(), options.stub_worker.resolve()
    legacy = {'catalog': options.legacy_catalog, 'runner-check': options.legacy_runner_check}
    folders = sorted(p for p in (ROOT / 'models').iterdir() if p.is_dir() and not p.name.startswith('.'))
    records = []
    with tempfile.TemporaryDirectory(prefix='arsenal-headless-') as temporary:
        root = Path(temporary)
        support = root / 'support'
        support.mkdir()
        lock = support / '.instance.lock'
        lock.write_text('headless acceptance owns the app lock\n')
        original = lock.read_bytes(), lock.stat().st_mtime_ns
        environment = os.environ | {'ARSENAL_SUPPORT_DIR': str(support),
                                    'ARSENAL_CACHE_DIR': str(root / 'cache'),
                                    'ARSENAL_HEADLESS': '1'}
        # Occupied app lock is a regression trap: dispatch after the single-instance claim
        # would exit silently rather than produce the required result. No UI app is launched.
        with lock.open('r+') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def check(command, arguments, *, expected=None, output=None, results=False):
                def invoke(prefix, label):
                    argv = [str(v) for v in arguments]
                    result_dir = root / f'{len(records)}-{label}'
                    if results:
                        argv += ['--results', str(result_dir)]
                    process = subprocess.run([*prefix, *argv], env=environment, cwd=ROOT,
                                             capture_output=True, text=True, timeout=60)
                    stdout = process.stdout.replace(str(result_dir), '<results>')
                    stderr = process.stderr.replace(str(result_dir), '<results>')
                    # Existing schema diagnostics walk Swift dictionaries; their order varies
                    # even between runs of the same legacy executable. Compare their full contents.
                    lines = []
                    for line in stdout.splitlines(keepends=True):
                        if line.startswith('FAIL op ') and '; ' in line:
                            heading, separator, diagnostic = line.rstrip('\n').partition(': ')
                            line = heading + separator + '; '.join(sorted(diagnostic.split('; '))) + '\n'
                        lines.append(line)
                    stdout = ''.join(lines)
                    report_path = result_dir / 'runner-check.json'
                    report = json.loads(report_path.read_text()) if report_path.exists() else None
                    if report:
                        report.pop('date', None)
                    return process.returncode, stdout, stderr, report

                actual = invoke([str(app), command], 'new')
                if legacy[command]:
                    baseline = invoke([str(legacy[command].resolve())], 'old')
                    assert actual == baseline, (command, arguments, actual, baseline)
                if expected is not None:
                    assert actual[0] == expected, (command, arguments, actual)
                if output is not None:
                    assert output in actual[1], (command, arguments, actual)
                assert not (support / 'status.json').exists(), 'command launched the app service'
                assert original == (lock.read_bytes(), lock.stat().st_mtime_ns), 'command touched app lock'
                records.append({'command': command, 'arguments': [str(v) for v in arguments],
                                'exit': actual[0], 'cases': actual[3]['cases'] if actual[3] else None})

            check('catalog', [], expected=0, output='arsenal-catalog fingerprint')
            check('catalog', ['--help'], expected=0, output='arsenal-catalog fingerprint')
            check('catalog', ['unknown'], expected=1)
            check('catalog', ['fingerprint'], expected=1)
            check('catalog', ['validate', '--all'], expected=0, output=f'Valid: {len(folders)} model folders')
            check('catalog', ['verify'], expected=0, output='Verified catalog sequence')
            check('runner-check', [], expected=2, output='Usage: arsenal-runner-check')
            check('runner-check', ['--help'], expected=0, output='Usage: arsenal-runner-check')
            check('runner-check', ['--model', 'models/missing'], expected=1)
            for timeout in ['0', 'nan', '181']:
                check('runner-check', ['python3', '--model', 'Tests/Fixtures/Templates/runners/toy',
                                       '--timeout', timeout], expected=1)
            for folder in folders:
                check('catalog', ['validate', folder], expected=0)
                check('catalog', ['fingerprint', folder], expected=0)
                # Compare full protocol/error paths using the same CPU stub on real declarations.
                # Stub failures are retained; this is entry-point parity, not model qualification.
                kind = next(line.split('=', 1)[1].strip().strip('"') for line in
                            (folder / 'model.toml').read_text().splitlines() if line.startswith('kind ='))
                check('runner-check', [stub, '--kind', kind, '--model', folder,
                                       '--kind-dir', ROOT / 'Resources/kinds', '--timeout', '1'], results=True)
    print(json.dumps({'checks': len(records), 'legacy_parity': bool(all(legacy.values())),
                      'occupied_app_lock_unchanged': True, 'real_weights_loaded': False,
                      'records': records}, indent=2))


if __name__ == '__main__':
    main()
