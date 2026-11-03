#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Agent-only isolated packaged drop-in smoke. No install, real profile or model download."""
import argparse
import hashlib
import importlib.util
import json
import os
import launch_services
from pathlib import Path
import shutil
import select
import struct
import subprocess
import tempfile
import time
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fresh_smoke', ROOT / 'scripts/smoke-fresh-install.py')
fresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fresh)


def run(command, **kwargs):
    return subprocess.run([str(s) for s in command], check=True, timeout=180, **kwargs)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(folder):
    files = sorted(p for p in (folder / 'runners/swift').rglob('*') if p.is_file() and p.name not in ('build-receipt.json', 'build.sh'))
    (folder / 'runners/swift/build-receipt.json').write_text(json.dumps({
        'command': 'fixture build.sh release arm64; codesign; fixture only; no MLX or model weights',
        'toolchain': '/Library/Developer/CommandLineTools/usr/bin/swift',
        'sha256': {str(p.relative_to(folder)): digest(p) for p in files},
    }))


def build_swift(folder, scratch):
    environment = os.environ.copy()
    environment['ARSENAL_FIXTURE_BUILD'] = str(scratch)
    run(['/bin/bash', folder / 'runners/swift/build.sh'], env=environment)
    receipt(folder)


def evidence(folder, runner, catalog, actual, seconds):
    check_files(actual)
    directory = folder / 'results/fix31'
    directory.mkdir(parents=True, exist_ok=True)
    retained = directory / 'actual.txt'
    retained.write_text(json.dumps(actual, sort_keys=True))
    identity = json.loads(run([catalog, 'catalog', 'fingerprint', folder], cwd=folder.parent, capture_output=True, text=True).stdout)
    result = {
        'schema': 1, 'model': folder.name, 'kind': 'local.swift-layers',
        'chip': 'fixture', 'os_build': 'packaged-smoke', 'app_build': '3.1.0', 'runner': runner,
        'worker_digest': 'a' * 64, 'precision': 'bf16', 'mode': 'exact', 'suite': {'id': 'fixture', 'version': 1, 'sha256': 'a' * 64},
        'profile': 'quick', 'date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stub': False,
        'gate_passed': True, 'conditions': {'measurement_lock': False, 'contaminated': False, 'contended': False, 'cpu_limit_pct': 250, 'gpu_limit_pct': 10, 'sample_interval_s': 3, 'energy': 'not_measured', 'idle_samples': [], 'run_samples': []},
        'metrics': {'seconds': seconds}, 'cases': [],
        'checks': {'protocol': 'pass', 'parity': 'pass', 'task': 'pass', 'speed': 'pass', 'stale': False,
                   'reference': {'identity': 'fixed-one-pixel-RGBA-fixture', 'independent': True},
                   **identity, 'cases': {'outputs': 4}, 'evidence': ['results/fix31/actual.txt'],
                   'evidence_hashes': {'results/fix31/actual.txt': digest(retained)}},
    }
    (directory / 'result.json').write_text(json.dumps(result))


def candidate_result(command, folder, donor, support, environment):
    """Exercise the edited fixture before approving it, against the fixed PNG reference."""
    output = support / 'jobs/candidate-check'
    output.mkdir(parents=True)
    process = subprocess.Popen([str(s) for s in command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env={**environment, 'ARSENAL_SUPPORT_DIR': str(support),
                                                            'ARSENAL_CACHE_DIR': str(support / 'cache')})
    deadline = time.monotonic() + 30
    def exact(count):
        data = b''
        while len(data) < count:
            remaining = deadline - time.monotonic()
            assert remaining > 0 and select.select([process.stdout], [], [], remaining)[0], 'candidate response deadline'
            chunk = os.read(process.stdout.fileno(), count - len(data))
            assert chunk, 'candidate closed before replying'
            data += chunk
        return data
    def response(identifier):
        while True:
            size = struct.unpack('>I', exact(4))[0]
            assert 0 < size <= 16 * 1024 * 1024
            frame = exact(size)
            assert frame[0] == 1
            value = json.loads(frame[1:])
            if value.get('id') == identifier and 'event' not in value:
                assert value.get('ok') is True, value
                return value
    def call(identifier, operation, **params):
        frame = b'\x01' + json.dumps({'id': identifier, 'op': operation, **params}).encode()
        process.stdin.write(struct.pack('>I', len(frame)) + frame)
        process.stdin.flush()
        return response(identifier)
    try:
        call(1, 'load', model_dir=str(folder), weights_dir=str(donor), dtype='bf16', fast=False, config={}, assets={})
        start = time.monotonic()
        actual = call(2, 'separate', layers=4, out_dir=str(output))
        seconds = time.monotonic() - start
        check_files(actual)
        assert actual['metadata']['revision'] == 2, actual
        call(3, 'unload')
        assert process.wait(timeout=10) == 0
        return actual, seconds
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()


def check_files(result):
    files = result['files']
    assert len(files) == 4 and [f['id'] for f in files] == [f'layer-{n}' for n in range(4)], result
    for n, item in enumerate(files):
        assert item['width'] == item['height'] == 1
        assert item['depth_min'] == n / 4 and item['depth_max'] == (n + 1) / 4
        path = Path(item['path'])
        data = path.read_bytes()
        assert digest(path) == item['sha256'] and len(data) == item['bytes']
        assert data[:8] == b'\x89PNG\r\n\x1a\n'
        width, height, depth, color = struct.unpack('>IIBB', data[16:26])
        assert (width, height, depth, color) == (1, 1, 8, 6)
        # Independent PNG framing/deflate check against the fixed RGBA fixture.
        offset = 8
        compressed = b''
        while offset < len(data):
            length = struct.unpack('>I', data[offset:offset + 4])[0]
            kind = data[offset + 4:offset + 8]
            payload = data[offset + 8:offset + 8 + length]
            if kind == b'IDAT':
                compressed += payload
            offset += 12 + length
        pixels = zlib.decompress(compressed)
        assert pixels[0] in range(5) and pixels[1:] == bytes([n * 20, 80, 200, 255]), pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('app', type=Path)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    source_app = args.app.resolve()
    args.evidence.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='arsenal-fix31-smoke-', dir='/private/tmp'))
    app = scratch / 'Arsenal.app'
    support = scratch / 'support'
    support.mkdir()
    donor = scratch / 'donor/fixture/layers'
    donor.mkdir(parents=True)
    (donor / 'base.bin').write_bytes(bytes([12, 34, 56]))
    (support / 'config.json').write_text(json.dumps({'table': [], 'get': {'localSources': [str(scratch / 'donor')]}, 'migrations': {'tableV2': True}}))
    environment = {**os.environ, 'ARSENAL_SUPPORT_DIR': str(support), 'ARSENAL_CACHE_DIR': str(scratch / 'cache'), 'ARSENAL_PLAYBACK_MUTED': '1', 'ARSENAL_HEADLESS': '1'}
    for key in ('ARSENAL_CATALOG', 'ARSENAL_HELPERS', 'ARSENAL_STUB', 'ARSENAL_API_PORT', 'HF_TOKEN'):
        environment.pop(key, None)
    # No Hugging Face token: the fixture's repository does not exist, and with the machine's own token Hugging Face says so
    # (a final 404) instead of asking for one, so approval would be refused on any Mac that has a token.
    environment['HF_HOME'] = str(scratch / 'hf-home')
    process = client = None
    logs = (args.evidence / 'app.log').open('w')
    report = {'app': str(source_app), 'launched_app': str(app), 'bundle_executable_sha256': digest(source_app / 'Contents/MacOS/Arsenal'), 'fixtures': [], 'machine_in_use': True}
    try:
        shutil.copytree(source_app, app, symlinks=True)
        process = subprocess.Popen([str(app / 'Contents/MacOS/Arsenal')], env=environment, stdout=logs, stderr=logs)
        deadline = time.monotonic() + 25
        while not (support / 'status.json').exists():
            assert process.poll() is None and time.monotonic() < deadline, 'candidate did not launch'
            time.sleep(.1)
        report['app_pid'] = process.pid
        client = fresh.Client(support, app / 'Contents/Helpers/arsenal-mcp')
        found = client.catalog()
        rows = found['models']
        signed = json.loads((app / 'Contents/Resources/catalog/catalog.json').read_text())
        expected = len(signed.get('models') or signed.get('entries') or [])
        assert expected and found['matched'] == expected and len(rows) == min(expected, 50), (found['matched'], len(rows), expected)
        report['catalog_models'] = found['matched']
        report['descriptors'] = {}
        for row in rows:
            operations = client.card(row['id'])['operations']
            assert operations, row['id']
            assert all(op['input_schema'] and op['name'] for op in operations), operations
            report['descriptors'][row['id']] = [op['name'] for op in operations]
        for identifier in ('d1-3b', 'flux2-klein-4b'):
            assert any(row['id'] == identifier for row in rows)
        # Assembly must follow the native families selected by the current checkout.
        for family in ('d1', 'flux'):
            source = ROOT / ('Workers/D1' if family == 'd1' else 'Workers/Flux')
            if source.exists():
                link = app / ('Contents/Helpers/arsenal-worker-' + family)
                assert link.is_symlink() and link.resolve() == (app / 'Contents/Helpers/arsenal-worker-native').resolve(), link
                report[family + '_native_link'] = True
        def http(path, body=None):
            req = urllib.request.Request(client.base + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.load(response)
        def refresh():
            client.tool('arsenal_settings', action='refresh')
        def model(identifier):
            return client.card(identifier)
        def approve(identifier):
            current = client.card(identifier)
            assert current['local'] and len(current['contentDigest']) == 64, current
            client.tool('arsenal_settings', action='approve_code', model=identifier, content_digest=current['contentDigest'])
            return current['contentDigest']
        def get(identifier):
            network = client.card(identifier).get('downloadBytes', {}).get('bf16')
            assert network is not None and network < 200_000_000, network
            client.wait(client.tool('arsenal_download', model=identifier, dtype='bf16', allow_download=True))
        for runner, identifier in [('swift', 'fixture-swift')]:
            folder = support / 'local-models' / identifier
            shutil.copytree(ROOT / 'Tests/Fixtures/DropIn/fixture-layers', folder)
            manifest = folder / 'model.toml'
            if runner == 'swift':
                text = manifest.read_text().replace('id = "fixture-layers"', 'id = "fixture-swift"').replace('local.layers', 'local.swift-layers')
                text = text.replace('worker = "stub"', 'entry = "runners/swift/bin/fixture-layers"')
                manifest.write_text(text)
                (folder / 'kinds/local.layers.toml').rename(folder / 'kinds/local.swift-layers.toml')
                (folder / 'kinds/local.layers').rename(folder / 'kinds/local.swift-layers')
                kind = folder / 'kinds/local.swift-layers.toml'
                kind.write_text(kind.read_text().replace('local.layers', 'local.swift-layers'))
                source = folder / 'runners/swift/Sources/FixtureLayers/main.swift'
                source.write_text(source.read_text().replace('local.layers', 'local.swift-layers'))
                build_swift(folder, scratch / 'swift-build')
            script = folder / 'runners/swift/Sources/FixtureLayers/main.swift'
            refresh()
            first = approve(identifier)
            client.tool('arsenal_add', model=identifier)
            get(identifier)
            command = [folder / 'runners/swift/bin/fixture-layers']
            checker = run([app / 'Contents/MacOS/Arsenal', 'runner-check', *command, '--model', folder,
                           '--weights', donor, '--kind-dir', folder / 'kinds', '--results', args.evidence / (identifier + '-checker')],
                          capture_output=True, text=True)
            (args.evidence / (identifier + '-checker.log')).write_text(checker.stdout)
            client.tool('arsenal_load', model=identifier, dtype='bf16')
            start = time.monotonic()
            result = http(f'/arsenal/v1/models/{identifier}/ops/separate', {'layers': 4})
            elapsed = time.monotonic() - start
            check_files(result)
            for item in result['files']:
                with urllib.request.urlopen(item['url'], timeout=10) as response:
                    assert hashlib.sha256(response.read()).hexdigest() == item['sha256']
            mcp = client.wait(client.tool('arsenal_run', model=identifier, operation='separate', input={'layers': 4}, format='json'))
            check_files(mcp)
            evidence(folder, runner, app / 'Contents/MacOS/Arsenal', result, elapsed)
            refresh()
            approve(identifier)
            # The optimized revision changes code and its output schema together.
            text = script.read_text()
            script.write_text(text.replace('REVISION = 1', 'REVISION = 2').replace('let revision = 1', 'let revision = 2'))
            schema = folder / f'kinds/local.swift-layers/ops/separate.output.json'
            output = json.loads(schema.read_text())
            output['properties']['metadata']['properties']['revision'] = {'const': 2}
            schema.write_text(json.dumps(output))
            # New pins reproduce the approve -> Get -> idle swap order.
            build_swift(folder, scratch / 'swift-build')
            changed = bytes([65, 43, 21])
            (donor / 'base.bin').write_bytes(changed)
            manifest.write_text(manifest.read_text().replace('a' * 40, 'b' * 40).replace('651f3a7ba220e024b0d76f3df555f406db4912ae05d63f13af6b02d7dcc0fa53', hashlib.sha256(changed).hexdigest()))
            candidate_actual, candidate_seconds = candidate_result(command, folder, donor, scratch / (identifier + '-candidate'), environment)
            evidence(folder, runner, app / 'Contents/MacOS/Arsenal', candidate_actual, candidate_seconds)
            refresh()
            candidate = approve(identifier)
            assert model(identifier)['revision']['active'] == first
            get(identifier)
            deadline = time.monotonic() + 20
            while True:
                current = model(identifier)
                if current['revision']['active'] == candidate:
                    break
                assert not current['revision'].get('error') and time.monotonic() < deadline, current
                time.sleep(.1)
            updated = http(f'/arsenal/v1/models/{identifier}/ops/separate', {'layers': 4})
            check_files(updated)
            assert updated['metadata']['revision'] == 2
            client.tool('arsenal_settings', action='rollback', model=identifier)
            rolled = http(f'/arsenal/v1/models/{identifier}/ops/separate', {'layers': 4})
            assert rolled['metadata']['revision'] == 1
            client.tool('arsenal_unload', model=identifier)
            assert process.poll() is None and digest(app / 'Contents/MacOS/Arsenal') == report['bundle_executable_sha256']
            report['fixtures'].append({'model': identifier, 'runner': runner, 'http_json': True, 'mcp': True, 'weight_revision_idle_swap': True, 'rollback': True, 'real_model_loaded': False, 'quick_seconds': elapsed})
            # The second model's original weights must be available too.
            (donor / 'base.bin').write_bytes(bytes([12, 34, 56]))
        client.close()
        client = None
        report['success'] = True
    finally:
        try:
            if client:
                client.close()
        finally:
            try:
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                logs.close()
                source_receipt = args.evidence.parent / 'fresh-install-smoke/source.json'
                source_receipt.parent.mkdir(parents=True, exist_ok=True)
                source_receipt.write_text(json.dumps({'source': 'verified-local', 'model': 'deterministic-fixtures', 'network_bytes': 0, 'verification': 'three-byte fixture weights; no real model loaded'}))
                report['app_exit_code'] = process.returncode if process else None
                (args.evidence / 'smoke.json').write_text(json.dumps(report, indent=2) + '\n')
            finally:
                try:
                    launch_services.cleanup_tree(scratch)
                finally:
                    shutil.rmtree(scratch)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
