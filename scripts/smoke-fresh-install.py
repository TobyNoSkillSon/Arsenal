#!/usr/bin/env python3
"""Exercise the assembled app through its bundled stdio MCP bridge."""
import hashlib
import json
import pathlib
import os
import re
import select
import signal
import subprocess
import sys
import time
import urllib.request
import wave


class Client:
    def __init__(self, support, bridge):
        self.support = pathlib.Path(support)
        self.base = 'http://127.0.0.1:' + str(json.loads((self.support / 'status.json').read_text())['port'])
        self.serial = 0
        self.buffer = b''
        environment = os.environ.copy()
        environment['ARSENAL_SUPPORT_DIR'] = str(self.support)
        self.process = subprocess.Popen([str(bridge)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment, start_new_session=False)
        try:
            self.request('initialize', {'protocolVersion': '2025-11-25', 'clientInfo': {'name': 'Arsenal fresh-install smoke', 'version': '3'}, 'capabilities': {}})
            self.request('notifications/initialized', None, notification=True)
        except BaseException:
            self.close()
            raise

    def request(self, method, params, notification=False):
        self.serial += 1
        body = {'jsonrpc': '2.0', 'method': method}
        if not notification:
            body['id'] = self.serial
        if params is not None:
            body['params'] = params
        self.process.stdin.write(json.dumps(body).encode() + b'\n')
        self.process.stdin.flush()
        if notification:
            return None
        deadline = time.monotonic() + 65
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('MCP bridge response deadline')
            while b'\n' not in self.buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise TimeoutError('MCP bridge response deadline')
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError('MCP bridge closed before replying')
                self.buffer += chunk
            line, self.buffer = self.buffer.split(b'\n', 1)
            result = json.loads(line)
            if result.get('id') != self.serial:
                continue
            if 'error' in result:
                raise RuntimeError(result['error'])
            return result['result']

    def tool(self, name, **arguments):
        """Calls a tool; structured data when it returns some (format:"json"), else {'text': ...}. Errors raise."""
        result = self.request('tools/call', {'name': name, 'arguments': arguments})
        text = next((item['text'] for item in result['content'] if item['type'] == 'text'), '')
        if result.get('isError'):
            raise RuntimeError(text)
        value = result.get('structuredContent')
        return value if isinstance(value, dict) else {'text': text}

    def wait(self, value):
        """Follows a job until it ends; arsenal_wait answers after at most 50 s, so this calls again."""
        deadline = time.monotonic() + 165
        started = re.search(r'Job ([0-9a-f]{10})', value.get('text', ''))
        if 'status' not in value and started:
            value = {'status': 'running', 'job': started.group(1)}
        while value.get('status') == 'running':
            assert time.monotonic() < deadline, 'job deadline'
            value = self.tool('arsenal_wait', job=value['job'], format='json')
        assert value.get('status') not in ('failed', 'cancelled'), value
        return value

    def card(self, model):
        """The model's catalog card (arsenal_model with format:"json"): states, sizes, digest, revision, operations."""
        return self.tool('arsenal_model', model=model, format='json')

    def catalog(self):
        """Every catalog model, also ones too large for this Mac."""
        return self.tool('arsenal_find', runs_here=False, limit=50, format='json')

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdout.close()
        assert self.process.returncode == 0, f"MCP bridge exited {self.process.returncode}"



def install_deadline(seconds=175):
    if os.getpgrp() != os.getpid():
        os.setpgrp()
    def expire(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)


def write_source_receipt(support, temporary, model, dtype, plan):
    weights = pathlib.Path(support) / 'weights'
    manifests = list((weights / 'snapshots' / model).glob(f'*/{dtype}/arsenal-snapshot.json'))
    assert len(manifests) == 1, 'expected one installed smoke snapshot'
    manifest = json.loads(manifests[0].read_text())
    assert manifest['model'] == model and manifest['precision'] == dtype, manifest
    installed = []
    for pin in manifest['files']:
        path = (manifests[0].parent / pin['path']).resolve(strict=True)
        assert path.is_relative_to(manifests[0].parent.resolve()) and path.is_file(), pin
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        assert path.stat().st_size == pin['bytes'] and digest.hexdigest() == pin['sha256'], pin
        installed.append(dict(pin))
    assert installed, 'empty installed smoke snapshot'
    # FP32 originals are collected after casting; Get's admission records retain their verified pins and origin.
    source_files = []
    for pin in (manifest.get('derived') or {}).get('sourceFiles', manifest['files']):
        record = json.loads((weights / 'verified' / (pin['sha256'] + '.json')).read_text())
        assert record['sha256'] == pin['sha256'] and record['identity']['bytes'] == pin['bytes'], pin
        source_files.append(dict(pin, source_kind=record.get('sourceKind', 'bundled-or-ready')))
    network_bytes = plan['network_bytes']
    local_copy_bytes = sum(item.get('localCopyBytes', 0) for item in plan['items'])
    receipt = {
        'source': 'network' if network_bytes else 'verified-local' if local_copy_bytes else 'bundled-or-ready',
        'model': model, 'dtype': dtype, 'network_bytes': network_bytes, 'local_copy_bytes': local_copy_bytes,
        'verification': {
            'revision': manifest['revision'], 'source_files': source_files, 'installed_files': installed,
            'source_method': 'Get SHA-256 and byte-size admission records',
            'installed_method': 'independent SHA-256 and byte-size check',
        },
    }
    target = pathlib.Path(temporary) / 'source.json'
    staging = target.with_suffix('.json.tmp')
    staging.write_text(json.dumps(receipt, indent=2) + '\n')
    staging.replace(target)
    return receipt


def main():
    install_deadline()
    mode, support, model, app, temporary = sys.argv[1:6]
    client = Client(support, pathlib.Path(app) / 'Contents/Helpers/arsenal-mcp')
    try:
        if mode == 'bootstrap':
            with urllib.request.urlopen(client.base + '/arsenal/v1/models', timeout=10) as response:
                rows = json.load(response)
            assert (rows.get('models', rows) if isinstance(rows, dict) else rows) == [], rows
            print('table: empty on first launch')
            found = client.catalog()
            ids = {row['id'] for row in found['models']}
            signed = json.loads((pathlib.Path(app) / 'Contents/Resources/catalog/catalog.json').read_text())
            expected = {row['manifest']['id'] for row in signed['models']}
            assert model in ids and ids <= expected and found['matched'] == len(expected), (found['matched'], len(expected))
            config_path = pathlib.Path(support) / 'config.json'
            config = json.loads(config_path.read_text())
            assert config['catalog']['source'] == 'bundle'
            print(f'catalog: all {len(ids)} signed models browsable offline, default source bundle')
            if len(sys.argv) > 6 and sys.argv[6]:
                source = pathlib.Path(sys.argv[6]).resolve(strict=True)
                config.setdefault('get', {})['localSources'] = [str(source)]
                config_path.write_text(json.dumps(config))
                client.tool('arsenal_settings', action='refresh')
                print('get: configured verified local source; shipping recipe unchanged')
            client.tool('arsenal_add', model=model)
            with urllib.request.urlopen(client.base + '/arsenal/v1/models', timeout=10) as response:
                table = json.load(response)
            table = table.get('models', table) if isinstance(table, dict) else table
            assert [row['id'] for row in table] == [model], table
            print(f'add: {model} is the only table model')
        elif mode == 'get':
            card = client.card(model)
            plan = {'network_bytes': card.get('downloadBytes', {}).get('fp16'), 'items': [{'localCopyBytes': card.get('localCopyBytes', {}).get('fp16', 0)}]}
            assert plan['network_bytes'] is not None and plan['network_bytes'] < 2_000_000_000, plan
            result = client.wait(client.tool('arsenal_download', model=model, dtype='fp16', allow_download=True))
            assert result.get('text', '').startswith('Downloaded') or 'is already downloaded' in result.get('text', ''), result
            source = write_source_receipt(support, temporary, model, 'fp16', plan)
            print('Get source: ' + json.dumps(source), flush=True)
            print('get: completed, weights verified')
        else:
            client.tool('arsenal_load', model=model, dtype='fp16')
            try:
                result = client.wait(client.tool('arsenal_run', model=model, operation='speak', input={'text': 'Arsenal is ready.', 'voice': 'af_heart'}, play=False, format='json'))
                with wave.open(result['files'][0]['path']) as audio:
                    assert audio.getnframes() > 0 and audio.getframerate() > 0
                    print(f'speak: real WAV, {audio.getnframes()} frames at {audio.getframerate()} Hz, silent')
            finally:
                client.tool('arsenal_unload', model=model)
    finally:
        client.close()


if __name__ == '__main__':
    main()
