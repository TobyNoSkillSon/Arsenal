#!/usr/bin/env python3
"""Isolated C8 acceptance: localhost, certificate-signed fixture apps, real helper swap/relaunch.
Run via scripts/mlock work v3-update-e2e -- python3 scripts/test-updates.py.
No real models, profile, installation, publication or GPU work. Temporary files/processes removed.
"""
import argparse
import ctypes
import hashlib
import http.server
import json
import os
import pathlib
import plistlib
import re
import uuid
import shutil
import signal
import stat
import zipfile
import subprocess
import tempfile
import threading
import time
import launch_services

REPO = pathlib.Path(__file__).resolve().parent.parent
CLT = pathlib.Path('/Library/Developer/CommandLineTools')
ENV = dict(os.environ, DEVELOPER_DIR=str(CLT))
PROCESSES = []
OWNED_ROOT = None
TEST_KEYCHAIN = None
SIGNING_PROCESSES = []
FIXTURE_ID = 'dev.arsenal.update-fixture'


def run(arguments, timeout=120, **kwargs):
    arguments = list(map(str, arguments))
    process = subprocess.Popen(arguments, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs)
    if pathlib.Path(arguments[0]).name in ['codesign', 'security']:
        SIGNING_PROCESSES.append((process.pid, pathlib.Path(arguments[0]).name))
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill(); process.communicate()
        print(f'Stopped timed-out owned {pathlib.Path(arguments[0]).name} PID {process.pid}', flush=True)
        raise
    if process.returncode:
        raise RuntimeError(stderr + stdout)
    return stdout.strip()


def disposable_signing_keychain(root):
    global TEST_KEYCHAIN
    password = 'arsenal-fixture-disposable-password'
    TEST_KEYCHAIN = root/'fixture-signing.keychain'
    # Modern Security creates private /tmp keychains without adding them to the user's search list.
    # Verify that property, and never use list-keychains -s or default-keychain.
    before = run(['/usr/bin/security', 'list-keychains', '-d', 'user'])
    run(['/usr/bin/security', 'create-keychain', '-p', password, TEST_KEYCHAIN])
    after = run(['/usr/bin/security', 'list-keychains', '-d', 'user'])
    if before != after:
        raise RuntimeError('Creating the private keychain changed the user search list; refusing to sign.')
    run(['/usr/bin/security', 'unlock-keychain', '-p', password, TEST_KEYCHAIN])
    identities = []
    for name in ['Arsenal Update Fixture A', 'Arsenal Update Fixture B']:
        prefix = root/name.replace(' ', '-')
        config = prefix.with_suffix('.cnf')
        config.write_text('[req]\ndistinguished_name=dn\nx509_extensions=code\nprompt=no\n[dn]\nCN='+name+
                          '\n[code]\nbasicConstraints=critical,CA:false\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=codeSigning\n')
        key, certificate, archive = prefix.with_suffix('.key'), prefix.with_suffix('.crt'), prefix.with_suffix('.p12')
        run(['/usr/bin/openssl', 'req', '-new', '-x509', '-newkey', 'rsa:2048', '-nodes', '-sha256', '-days', '1',
             '-config', config, '-keyout', key, '-out', certificate])
        run(['/usr/bin/openssl', 'pkcs12', '-export', '-inkey', key, '-in', certificate, '-name', name,
             '-passout', 'pass:'+password, '-out', archive])
        run(['/usr/bin/security', 'import', archive, '-k', TEST_KEYCHAIN, '-P', password, '-T', '/usr/bin/codesign'])
        identities.append(name)
    run(['/usr/bin/security', 'set-key-partition-list', '-S', 'apple-tool:,apple:,codesign:', '-s', '-k', password, TEST_KEYCHAIN])
    assert before == run(['/usr/bin/security', 'list-keychains', '-d', 'user'])
    print('PASS disposable signing: two real certificates, explicit /tmp keychain, unchanged user search list', flush=True)
    return identities


def await_file(path, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return path.read_text()
        time.sleep(.1)
    diagnostics = []
    for name in ['process.log', 'state', 'helper-finished', 'support/update-failure.txt']:
        file = path.parent/name
        if file.exists(): diagnostics.append(name+': '+file.read_text()[:4000])
    raise AssertionError(f'Timeout waiting for {path}; '+ '; '.join(diagnostics))


def sign(app, identity):
    for executable in (app/'Contents/Helpers').iterdir():
        run(['codesign', '--keychain', TEST_KEYCHAIN, '--force', '--options', 'runtime', '--timestamp=none', '--sign', identity, '--identifier', 'dev.arsenal.helper.'+executable.name, executable])
    run(['codesign', '--keychain', TEST_KEYCHAIN, '--force', '--options', 'runtime', '--timestamp=none', '--sign', identity, '--identifier', FIXTURE_ID, app])
    run(['codesign', '--verify', '--deep', '--strict', app])


def bundle(path, fixture, helper, version, identity):
    (path/'Contents/MacOS').mkdir(parents=True)
    (path/'Contents/Helpers').mkdir()
    shutil.copy2(fixture, path/'Contents/MacOS/Arsenal')
    shutil.copy2(helper, path/'Contents/Helpers/ArsenalUpdateHelper')
    info = dict(CFBundleIdentifier=FIXTURE_ID, CFBundleExecutable='Arsenal', CFBundleName='Arsenal Update Fixture',
                CFBundlePackageType='APPL', CFBundleShortVersionString=version, CFBundleVersion='1', LSUIElement=True)
    (path/'Contents/Info.plist').write_bytes(plistlib.dumps(info))
    sign(path, identity)


def file_digest(root):
    result = {}
    for file in root.rglob('*'):
        if file.is_file():
            result[str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
    return result


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.requests.append(self.path)
        mode = getattr(self.server, 'mode', '')
        if self.path == '/releases/latest' and self.server.requests.count(self.path) > 1 and mode in ['check-race', 'check-failure', 'check-manual-failure']:
            time.sleep(1)
            if mode in ['check-failure', 'check-manual-failure']: self.send_error(503); return
        if self.path == '/app.zip' and mode.startswith('redirect-'):
            location = {'redirect-host': 'https://example.com/forbidden.zip',
                        'redirect-http': 'http://github.com/forbidden.zip',
                        'redirect-local': f'http://127.0.0.1:{self.server.server_port}/redirected.zip'}[mode]
            self.send_response(302); self.send_header('Location', location); self.end_headers(); return
        if self.path == '/app.zip' and mode == 'oversized':
            self.send_response(200); self.send_header('Content-Length', str(256*1024*1024+1)); self.end_headers()
            try: self.wfile.write(b'x'*65536)
            except (BrokenPipeError, ConnectionResetError): pass
            return
        if self.path == '/releases/latest':
            base = f'http://127.0.0.1:{self.server.server_port}'
            body = json.dumps(dict(tag_name='v3.0.0', body='Everything updates together.', draft=False, prerelease=False,
                assets=[dict(name='Arsenal-3.0.0-arm64.zip', size=len(self.server.zip), browser_download_url=base+'/app.zip'),
                        dict(name='SHA256SUMS', size=len(self.server.sums), browser_download_url=base+'/SHA256SUMS')])).encode()
        elif self.path in ['/app.zip', '/redirected.zip']:
            body = self.server.zip
        elif self.path == '/SHA256SUMS':
            body = self.server.sums
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try: self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError): pass


def launch(app, root, mode, server):
    env = dict(ENV, ARSENAL_UPDATE_TEST_ROOT=str(root), ARSENAL_UPDATE_TEST_MODE=mode,
               ARSENAL_SUPPORT_DIR=str(root/'support'), ARSENAL_CACHE_DIR=str(root/'cache'),
               ARSENAL_UPDATE_TEST_API=f'http://127.0.0.1:{server.server_port}/releases/latest')
    log = (root/'process.log').open('w')
    process = subprocess.Popen([str(app/'Contents/MacOS/Arsenal')], env=env, stdout=log, stderr=log)
    log.close()
    PROCESSES.append(process)
    return process


def stop_pid(pid):
    # A stale receipt PID may have been reused. Only signal an executable within this run's install root.
    path = ctypes.create_string_buffer(4096)
    library = ctypes.CDLL('/usr/lib/libproc.dylib')
    if not library.proc_pidpath(pid, path, len(path)) or not path.value.decode().startswith(str(OWNED_ROOT)+'/'):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class RUsage(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8*16)] + [(name, ctypes.c_uint64) for name in
        ['user', 'system', 'idle_wakeups', 'interrupt_wakeups', 'pageins', 'wired', 'resident', 'footprint', 'started', 'exited']]


def idle_sample(pid):
    info = RUsage()
    library = ctypes.CDLL('/usr/lib/libproc.dylib')
    if library.proc_pid_rusage(pid, 0, ctypes.byref(info)):
        raise RuntimeError('Could not sample fixture process idle usage')
    return info.user+info.system, info.idle_wakeups, info.interrupt_wakeups


def assert_idle_unchanged(scratch, old, server):
    processes = []
    server.requests = []; server.mode = ''
    for mode in ['idle-baseline', 'idle-scheduler']:
        root = scratch/mode; root.mkdir(); app = root/'Arsenal.app'; shutil.copytree(old, app)
        process = launch(app, root, mode, server)
        armed = await_file(root/'armed')
        assert armed.endswith('true' if mode == 'idle-scheduler' else 'false'), armed
        processes.append((mode, process))
    time.sleep(2)  # Let launch registration settle before the idle sample.
    before = [idle_sample(p.pid) for _, p in processes]
    start = time.monotonic(); time.sleep(8); elapsed = time.monotonic()-start
    after = [idle_sample(p.pid) for _, p in processes]
    readings = []
    for (mode, process), first, last in zip(processes, before, after):
        delta = [b-a for a, b in zip(first, last)]
        cpu = delta[0]/1e9/elapsed*100
        readings.append((cpu, delta[1], delta[2]))
        print(f'IDLE {mode}: {cpu:.4f}% CPU, {delta[1]} package-idle / {delta[2]} interrupt wakeups over {elapsed:.1f}s', flush=True)
        stop_pid(process.pid); process.wait(timeout=5)
    baseline, armed = readings
    assert armed[0] < .1, readings
    assert armed[1] <= baseline[1] and armed[2] <= baseline[2], readings
    print('PASS scheduler armed: no additional wakeups against the same fixture without a scheduler', flush=True)
    assert server.requests == [], server.requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', help='Comma-separated cases for an affected-path rerun; default runs every case.')
    parser.add_argument('--skip-idle', action='store_true', help='Reuse an existing unchanged scheduler measurement.')
    options = parser.parse_args()
    modes = ['no-repo', 'hash', 'signature', 'unsafe-path', 'unsafe-link', 'oversized', 'redirect-host', 'redirect-http', 'redirect-local',
             'check-race', 'check-failure', 'good', 'restore-failure', 'restore-timeout', 'launch-failure', 'launch-timeout',
             'exit-timeout', 'late-exit', 'death-before-swap', 'death-after-swap', 'rollback-cleanup', 'existing-previous',
             'restore-missing', 'restore-mismatch', 'slow-startup', 'launch-callback-timeout', 'check-manual-failure', 'launch-unconfirmed', 'restore-slow', 'restore-row-failure']
    if options.cases:
        selected = options.cases.split(',')
        if set(selected)-set(modes): parser.error('Unknown case: '+','.join(set(selected)-set(modes)))
        modes = selected
    global OWNED_ROOT, FIXTURE_ID
    scratch = pathlib.Path(tempfile.mkdtemp(prefix='arsenal-update-e2e-', dir='/tmp')).resolve()
    OWNED_ROOT = scratch
    FIXTURE_ID = 'dev.arsenal.update-fixture.' + uuid.uuid4().hex
    server = Server(('127.0.0.1', 0), Handler)
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        identity, other = disposable_signing_keychain(scratch)
        fixture = scratch/'fixture'
        # The expected identity alone is substituted in task-owned source copies. Production sources stay untouched.
        # The same verifier, installer and helper entry point run against a non-production LaunchServices identity.
        source_root = scratch/'sources'; source_root.mkdir()
        sources = []
        for source in sorted((REPO/'Sources/ArsenalCore/Update').glob('*.swift')):
            copy = source_root/source.name
            copy.write_text(source.read_text().replace('dev.arsenal.app', FIXTURE_ID))
            sources.append(copy)
        helper_source = source_root/'main.swift'
        helper_source.write_text((REPO/'Sources/ArsenalUpdateHelper/main.swift').read_text().replace('import ArsenalCore\n', ''))
        helper = scratch/'ArsenalUpdateHelper'
        run([CLT/'usr/bin/swiftc', '-parse-as-library', '-swift-version', '6', '-target', 'arm64-apple-macos26.0',
             '-sdk', CLT/'SDKs/MacOSX.sdk', '-O', *sources, helper_source, '-o', helper])
        # Exercise the real production applicationShouldTerminate through AppKit, with fixture shutdown work.
        delegate_source = (REPO/'Sources/Arsenal/App/AppDelegate.swift').read_text()
        begin = delegate_source.index('    func applicationShouldTerminate(')
        end = delegate_source.index('\n    ///', begin)
        method = delegate_source[begin:end]
        quit_declaration = re.search(r'static let quitLimit: Duration = (.+)', delegate_source).group(1)
        check_source = scratch/'quit-limit.swift'
        check_source.write_text('import Foundation\n@main struct LimitCheck { static func main() { let quit: Duration = '+quit_declaration+
                                '; precondition(Duration.seconds(UpdateInstaller.Limits().exit) > quit) } }\n')
        check_binary = scratch/'quit-limit'
        run([CLT/'usr/bin/swiftc', '-parse-as-library', '-swift-version', '6', '-target', 'arm64-apple-macos26.0',
             '-sdk', CLT/'SDKs/MacOSX.sdk', *sources, check_source, '-o', check_binary])
        run([check_binary])
        print('PASS production helper exit allowance exceeds actual AppDelegate.quitLimit ('+quit_declaration+')', flush=True)
        fixture_source = scratch/'update-app.swift'
        fixture_source.write_text((REPO/'scripts/fixtures/update-app.swift').read_text().replace(
            '    // applicationShouldTerminate is inserted verbatim from the production AppDelegate by the test builder.', method))
        run([CLT/'usr/bin/swiftc', '-parse-as-library', '-swift-version', '6', '-target', 'arm64-apple-macos26.0',
             '-sdk', CLT/'SDKs/MacOSX.sdk', '-O', *sources, fixture_source, '-o', fixture])
        fixture_helper = scratch/'fixture-helper'
        run([CLT/'usr/bin/swiftc', '-parse-as-library', '-swift-version', '6', '-target', 'arm64-apple-macos26.0',
             '-sdk', CLT/'SDKs/MacOSX.sdk', '-O', *sources, REPO/'scripts/fixtures/update-helper.swift', '-o', fixture_helper])
        old = scratch/'old/Arsenal.app'; new = scratch/'new/Arsenal.app' 
        bundle(old, fixture, helper, '2.2.0', identity)
        bundle(new, fixture, helper, '3.0.0', identity)
        archive = scratch/'good.zip'
        run(['/usr/bin/ditto', '-c', '-k', '--keepParent', '--norsrc', new, archive])
        good = archive.read_bytes()
        ancient = scratch/'ancient/Arsenal.app'
        bundle(ancient, fixture, helper, '1.0.0', identity)
        wrong_bytes = None
        if 'signature' in modes:
            wrong = scratch/'wrong/Arsenal.app'; shutil.copytree(new, wrong)
            sign(wrong, other)
            wrong_zip = scratch/'wrong.zip'; run(['/usr/bin/ditto', '-c', '-k', '--keepParent', '--norsrc', wrong, wrong_zip])
            wrong_bytes = wrong_zip.read_bytes()
        unsafe_bytes = {}
        for kind in ['unsafe-path', 'unsafe-link']:
            malicious = scratch/(kind+'.zip')
            shutil.copy2(archive, malicious)
            with zipfile.ZipFile(malicious, 'a') as output:
                if kind == 'unsafe-path':
                    output.writestr('../escaped', b'blocked before extraction')
                else:
                    link = zipfile.ZipInfo('Arsenal.app/Contents/escape')
                    link.create_system = 3; link.external_attr = (stat.S_IFLNK | 0o777) << 16
                    output.writestr(link, '/tmp')
            unsafe_bytes[kind] = malicious.read_bytes()
        for mode in modes:
            root = scratch/mode; root.mkdir(); app = root/'Arsenal.app'; shutil.copytree(old, app)
            if mode in ['restore-timeout', 'launch-failure', 'launch-timeout', 'exit-timeout', 'late-exit', 'death-before-swap', 'death-after-swap', 'rollback-cleanup', 'slow-startup', 'launch-callback-timeout', 'launch-unconfirmed', 'restore-slow', 'restore-row-failure']:
                shutil.copy2(fixture_helper, app/'Contents/Helpers/ArsenalUpdateHelper'); sign(app, identity)
            previous = pathlib.Path(str(app)+'.previous')
            older_before = None
            if mode in ['death-before-swap', 'death-after-swap', 'existing-previous', 'rollback-cleanup', 'launch-unconfirmed']:
                shutil.copytree(ancient, previous); older_before = file_digest(previous)
                # Also exercise cleanup of an incomplete copy left by a dead helper.
                orphan = root/('.Arsenal.app.incoming-'+str(uuid.uuid4()))
                orphan.mkdir(); (orphan/'partial').write_text('incomplete update copy')
            before = file_digest(app)
            server.zip = unsafe_bytes.get(mode, wrong_bytes if mode == 'signature' else good)
            digest = '0'*64 if mode == 'hash' else hashlib.sha256(server.zip).hexdigest()
            server.sums = f'{digest}  Arsenal-3.0.0-arm64.zip\n'.encode()
            server.requests = []; server.mode = mode
            process = launch(app, root, mode, server)
            if mode == 'no-repo':
                assert await_file(root/'no-repo') == 'false false'
                process.wait(timeout=5)
                assert server.requests == [], server.requests
                assert file_digest(app) == before
                print('PASS no repo: no scheduler, no network, no bundle changes', flush=True)
                continue
            await_file(root/'available')
            if mode in ['hash', 'signature', 'unsafe-path', 'unsafe-link', 'oversized', 'redirect-host', 'redirect-http']:
                reason = await_file(root/'failure')
                stop_pid(process.pid); process.wait(timeout=5)
                expected = {'hash': 'SHA256', 'signature': 'signature', 'unsafe-path': 'unsafe paths', 'unsafe-link': 'symbolic link', 'oversized': 'too large', 'redirect-host': 'release asset', 'redirect-http': 'release asset'}[mode]
                assert expected in reason, reason
                assert file_digest(app) == before
                assert not pathlib.Path(str(app)+'.previous').exists()
                print(f'PASS {mode}: rejected before swap; installed bundle unchanged', flush=True)
                continue
            if mode in ['check-race', 'check-failure', 'check-manual-failure']:
                await_file(root/'check-start')
                state = await_file(root/'check-state')
                assert state.startswith('failed' if mode == 'check-manual-failure' else 'available'), state
                if mode == 'check-race':
                    assert await_file(root/'race-state').startswith('available')
                    assert not (root/'waiting').exists() and server.requests.count('/app.zip') == 0
                    print('PASS two clicks during in-flight check: no install, idle acquisition or download', flush=True)
                else:
                    assert 'Could not check' in await_file(root/'check-error')
                    print('PASS '+('manual check visibly fails' if mode == 'check-manual-failure' else 'background check failure retains available offer'), flush=True)
                assert file_digest(app) == before
                stop_pid(process.pid); process.wait(timeout=5)
                continue
            await_file(root/'waiting')
            assert file_digest(app) == before
            if older_before is None: assert not previous.exists()
            else: assert file_digest(previous) == older_before
            assert server.requests.count('/releases/latest') == 1, server.requests
            time.sleep(.3)
            (root/'idle').touch()
            assert await_file(root/'returned').startswith('installing')
            helper_pid = int(await_file(root/'helper-pid')) if (mode in ['restore-timeout', 'launch-failure', 'launch-timeout', 'exit-timeout', 'late-exit', 'death-before-swap', 'death-after-swap', 'rollback-cleanup', 'slow-startup', 'launch-callback-timeout', 'launch-unconfirmed', 'restore-slow', 'restore-row-failure']) else None
            if helper_pid: PROCESSES.append(helper_pid)
            if mode == 'exit-timeout':
                reason = await_file(root/'failure', timeout=20)
                assert 'did not quit in time' in reason, reason
                await_file(root/'released')
                assert file_digest(app) == before
                assert not (root/'support/update-restore.json').exists()
                stop_pid(process.pid); process.wait(timeout=5)
                print('PASS cancelled quit: bounded exit + late-exit wait, unchanged bundle, admission released', flush=True)
                continue
            process.wait(timeout=15)
            if mode in ['restore-slow', 'restore-row-failure']:
                pid = int(await_file(root/'native-launch-complete')); PROCESSES.append(pid)
                await_file(root/'restore-started')
                if mode == 'restore-slow':
                    # Keep models pending beyond the helper's four-second launch-answer deadline.
                    time.sleep(5)
                    acknowledgements = list((root/'support').glob('update-ack-*'))
                    assert len(acknowledgements) == 1, acknowledgements
                    ack = json.loads(acknowledgements[0].read_text())
                    record = json.loads((root/'support/update-restore.json').read_text())
                    assert ack['phase'] == 'launched' and ack['pid'] == pid and ack['token'] == record['token'], ack
                    assert not (root/'restored').exists() and not (root/'restore-state').exists()
                    assert not (root/'helper-finished').exists(), 'helper rejected a launched app while restoration was pending'
                    assert plistlib.loads((app/'Contents/Info.plist').read_bytes())['CFBundleShortVersionString'] == '3.0.0'
                    assert file_digest(previous) == before
                    os.kill(pid, 0)
                    (root/'finish-restore').touch()
                else:
                    errors = json.loads(await_file(root/'row-failures'))
                    assert set(errors) == {'fixture-chat', 'fixture-voice'} and all(errors.values())
                restored = json.loads(await_file(root/'restored'))
                assert restored['version'] == '3.0.0' and restored['pid'] == pid
                assert restored['models'] == ([] if mode == 'restore-row-failure' else ['fixture-chat', 'fixture-voice'])
                assert await_file(root/'helper-finished') == 'success'
                assert await_file(root/'restore-state') == 'none'
                assert len((root/'launch-pids').read_text().splitlines()) == 1
                assert not (root/'support/update-failure.txt').exists()
                assert not list((root/'support').glob('update-ack-*'))
                assert plistlib.loads((app/'Contents/Info.plist').read_bytes())['CFBundleShortVersionString'] == '3.0.0'
                assert file_digest(previous) == before
                os.kill(pid, 0)
                print(f'PASS {mode}: launch answered independently; model restoration never rolled back the app', flush=True)
                stop_pid(pid)
                continue
            if mode == 'launch-unconfirmed':
                reason = await_file(root/'helper-finished', timeout=15)
                assert 'not' in reason or 'timed out' in reason, reason
                assert await_file(root/'open-count') == '1'
                assert not list(root.glob('.Arsenal.app.previous-*'))
                assert not (root/'launched').exists()
                assert file_digest(previous) == before
                assert plistlib.loads((app/'Contents/Info.plist').read_bytes())['CFBundleShortVersionString'] == '3.0.0'
                assert 'No second instance' in await_file(root/'support/update-failure.txt')
                print('PASS unconfirmed launch timeout: exactly one request, verified app and previous retained, bounded helper', flush=True)
                continue
            if mode.startswith('death-'):
                pid = int(await_file(root/'checkpoint'))
                os.kill(pid, signal.SIGKILL)
                previous = pathlib.Path(str(app)+'.previous')
                installed = plistlib.loads((app/'Contents/Info.plist').read_bytes())['CFBundleShortVersionString']
                expected = '2.2.0' if mode == 'death-before-swap' else '3.0.0'
                assert installed == expected, installed
                if mode == 'death-before-swap': assert file_digest(app) == before
                else: assert file_digest(previous) == before
                run(['codesign', '--verify', '--deep', '--strict', app])
                # Run the exact startup reconciler after helper death. It preserves the correct older bundle.
                run([fixture_helper, '--reconcile', app])
                expected_previous = older_before if mode == 'death-before-swap' else before
                actual_previous = file_digest(previous)
                differences = [name for name in set(actual_previous)|set(expected_previous) if actual_previous.get(name) != expected_previous.get(name)]
                assert actual_previous == expected_previous, (mode, plistlib.loads((previous/'Contents/Info.plist').read_bytes())['CFBundleShortVersionString'], differences[:8], [p.name for p in root.glob('.Arsenal.app.*')])
                assert not list(root.glob('.Arsenal.app.incoming-*')) and not list(root.glob('.Arsenal.app.previous-*'))
                print(f'PASS {mode}: installed path always present and valid; atomic swap preserves previous', flush=True)
                continue
            if mode in ['restore-failure', 'restore-timeout', 'restore-missing', 'restore-mismatch']:
                pid = int(await_file(root/'launched')); PROCESSES.append(pid)
                if mode == 'restore-timeout':
                    await_file(root/'helper-finished', timeout=15)
                    json.loads(await_file(root/'restored', timeout=15))
                    assert await_file(root/'restore-state') == 'none', await_file(root/'restore-state')
                else:
                    assert await_file(root/'restore-state').startswith('failed'), await_file(root/'restore-state')
                    if mode == 'restore-mismatch': assert 'does not match this launch' in await_file(root/'restore-state')
                # Keep the app alive until the helper consumes its launch/restore receipt. Stopping it at
                # the controller's state marker can otherwise simulate a launch failure before the next poll.
                deadline = time.monotonic()+15
                while time.monotonic() < deadline and list((root/'support').glob('update-ack-*')):
                    time.sleep(.1)
                assert not list((root/'support').glob('update-ack-*')), 'helper did not consume the restore receipt'
                assert not (root/'support/update-failure.txt').exists()
                assert plistlib.loads((app/'Contents/Info.plist').read_bytes())['CFBundleShortVersionString'] == '3.0.0'
                assert file_digest(previous) == before
                os.kill(pid, 0)
                print(f'PASS {mode}: new version kept; no stale failure file; late successful restore stays successful', flush=True)
                stop_pid(pid)
                continue
            restored = json.loads(await_file(root/'restored', timeout=30))
            PROCESSES.append(restored['pid'])
            assert restored['models'] == ['fixture-chat', 'fixture-voice'], restored
            if mode in ['launch-failure', 'launch-timeout', 'late-exit', 'rollback-cleanup']:
                assert restored['version'] == '2.2.0', restored
                assert file_digest(app) == before
                await_file(root/'helper-finished')
                if mode == 'rollback-cleanup':
                    assert previous.exists()
                    assert previous.stat().st_flags & stat.UF_IMMUTABLE, 'cleanup fault did not make the failed bundle immutable'
                    os.chflags(previous, 0)
                    run([fixture_helper, '--reconcile', app])
                    assert file_digest(previous) == older_before
                    assert not list(root.glob('support/update-ack-*'))
                assert not list((root/'support').glob('update-ack-*'))
                assert not (root/'support/update-restore.json').exists()
                print(f'PASS {mode}: previous app relaunched and restored; deadlines bounded; ack cleaned', flush=True)
            else:
                assert restored['version'] == '3.0.0', restored
                assert file_digest(pathlib.Path(str(app)+'.previous')) == before
                if helper_pid: await_file(root/'helper-finished', timeout=15)
                if mode == 'launch-callback-timeout':
                    assert len((root/'launch-pids').read_text().splitlines()) == 1
                # The app writes its restore marker just before the helper observes its final receipt.
                deadline = time.monotonic()+15
                while time.monotonic() < deadline and (list(root.glob('.Arsenal.app.incoming-*')) or list(root.glob('.Arsenal.app.previous-*'))):
                    time.sleep(.1)
                assert not list(root.glob('.Arsenal.app.incoming-*')) and not list(root.glob('.Arsenal.app.previous-*'))
                print(f'PASS {mode}: real AppKit .terminateLater → atomic swap → relaunch → restore, previous retained', flush=True)
            stop_pid(restored['pid'])
        if not options.skip_idle: assert_idle_unchanged(scratch, old, server)
        print('PASS '+('selected' if options.cases else 'full')+' local release acceptance complete (no real models loaded): '+','.join(modes), flush=True)
    finally:
        cleanup_errors = []
        def cleanup(action):
            try:
                action()
            except Exception as error:
                cleanup_errors.append(str(error))
        cleanup(server.shutdown)
        cleanup(server.server_close)
        for process in PROCESSES:
            if isinstance(process, int):
                cleanup(lambda process=process: stop_pid(process))
            elif process.poll() is None:
                cleanup(lambda process=process: stop_pid(process.pid))
                cleanup(lambda process=process: process.wait(timeout=5))
        # Catch an unacknowledged relaunch; only task-owned exact executable paths.
        def stop_unacknowledged():
            listing = run(['/bin/ps', '-axo', 'pid=,command='])
            for line in listing.splitlines():
                if str(scratch) in line and '/Contents/MacOS/Arsenal' in line:
                    stop_pid(int(line.strip().split(maxsplit=1)[0]))
        cleanup(stop_unacknowledged)
        # Unique per-run identity also covers updater staging directories already deleted by production cleanup.
        cleanup(lambda: launch_services.unregister(r['path'] for r in launch_services.dump_registrations() if r['identifier'] == FIXTURE_ID))
        def clean_stage_records():
            for root in scratch.iterdir():
                if root.is_dir():
                    stage_record = root/'stage-dir'
                    if stage_record.exists():
                        stage = pathlib.Path(stage_record.read_text())
                        if stage.name.startswith('arsenal-update-') and stage.parent == pathlib.Path(tempfile.gettempdir()).resolve():
                            cleanup(lambda stage=stage: launch_services.cleanup_tree(stage))
                            cleanup(lambda stage=stage: shutil.rmtree(stage, ignore_errors=True))
                    cleanup(lambda root=root: subprocess.run(['/usr/bin/defaults', 'delete', f'dev.arsenal.update-test.{scratch.name}.{root.name}'], capture_output=True, timeout=5))
            for previous in scratch.glob('*/Arsenal.app.previous'):
                cleanup(lambda previous=previous: os.chflags(previous, 0))
        cleanup(clean_stage_records)
        cleanup(lambda: launch_services.cleanup_tree(scratch))
        if TEST_KEYCHAIN is not None and TEST_KEYCHAIN.exists():
            cleanup(lambda: run(['/usr/bin/security', 'delete-keychain', TEST_KEYCHAIN]))
            if TEST_KEYCHAIN.exists():
                cleanup_errors.append('Disposable keychain still exists: ' + str(TEST_KEYCHAIN))
            else:
                print('PASS disposable keychain deleted; signing/security child PIDs: '+str(SIGNING_PROCESSES), flush=True)
        cleanup(lambda: shutil.rmtree(scratch))
        if cleanup_errors:
            raise RuntimeError('Fixture cleanup failed: ' + '; '.join(cleanup_errors))



if __name__ == '__main__':
    os.chdir(REPO)
    main()
