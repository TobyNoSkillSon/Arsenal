#!/usr/bin/env python3
"""Deterministic regression checks for incremental invalidation and generated output reuse."""
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import shutil
import unittest
from unittest.mock import patch
from incremental import consumers, prepare

spec = importlib.util.spec_from_file_location('build_cache', Path(__file__).with_name('build-cache.py'))
cache = importlib.util.module_from_spec(spec); spec.loader.exec_module(cache)


# A small fixed graph for routing/invalidation fixtures; production reads Package.swift.
DEPENDENCIES = {
    'ArsenalWire': [], 'ArsenalCore': ['ArsenalWire'],
    'ArsenalBench': ['ArsenalCore'], 'ArsenalCommands': ['ArsenalCore', 'ArsenalBench', 'ArsenalWire'],
    'ArsenalAPI': ['ArsenalCore', 'ArsenalBench'], 'Arsenal': ['ArsenalCore', 'ArsenalAPI', 'ArsenalCommands'],
    'ArsenalUpdateHelper': ['ArsenalCore'], 'arsenal-bench': ['ArsenalBench'],
    'arsenal-stats': ['ArsenalCore'], 'arsenal-stub-worker': ['ArsenalWire'], 'arsenal-testserver': ['ArsenalAPI'],
    'ArsenalRunnerCheckTests': ['ArsenalCommands'], 'ArsenalCoreTests': ['ArsenalCore'],
    'ArsenalAPITests': ['ArsenalAPI'], 'ArsenalBenchTests': ['ArsenalBench'], 'ArsenalAppTests': ['Arsenal', 'ArsenalCore'],
    'ArsenalPackageTests': ['ArsenalRunnerCheckTests', 'ArsenalCoreTests', 'ArsenalAPITests', 'ArsenalBenchTests', 'ArsenalAppTests'],
}

class BuildInfrastructureTests(unittest.TestCase):
    def test_core_and_api_consumers_are_transitive(self):
        self.assertIn('ArsenalAppTests', consumers({'ArsenalCore'}, DEPENDENCIES))
        self.assertIn('ArsenalAPITests', consumers({'ArsenalCore'}, DEPENDENCIES))
        self.assertNotIn('ArsenalCore', consumers({'ArsenalCore'}, DEPENDENCIES))
        future = {**DEPENDENCIES, 'NewModelTests': ['ArsenalAPI']}
        self.assertIn('NewModelTests', consumers({'ArsenalCore'}, future))
        self.assertEqual(consumers({'ArsenalAPI'}, DEPENDENCIES), {'Arsenal', 'arsenal-testserver', 'ArsenalAPITests', 'ArsenalAppTests', 'ArsenalPackageTests'})

    def test_changed_core_retains_dependencies_and_refreshes_consumers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['Package.swift', 'Packages/ArsenalWire/Package.swift', 'Sources/ArsenalCore/A.swift', 'Sources/ArsenalAPI/A.swift']:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('before')
            scratch = root / '.build'; output = scratch / 'arm64-apple-macosx/debug'
            for target in ['ArsenalCore', 'ArsenalAPI', 'ArsenalAppTests', 'arsenal_testserver', 'MCP']:
                folder = output / (target + '.build'); folder.mkdir(parents=True); (folder / 'A.o').write_text('compiled'); (folder / 'output-file-map.json').write_text('{}')
            metadata = {'key': 'fixture', 'dependencies': DEPENDENCIES, 'folders': {name: 'Sources/' + name for name in DEPENDENCIES}}
            with patch('incremental.package_graph', return_value=metadata):
                initial = prepare(root, scratch, 'debug')
            (scratch / 'arsenal-abi-debug.json').write_text(json.dumps(initial))
            for target in ['ArsenalCore', 'ArsenalAPI', 'ArsenalAppTests', 'arsenal_testserver']:
                folder = output / (target + '.build'); folder.mkdir(exist_ok=True); (folder / 'A.o').write_text('compiled'); (folder / 'output-file-map.json').write_text('{}')
            (root / 'Sources/ArsenalCore/A.swift').write_text('ABI changed')
            with patch('incremental.package_graph', return_value=metadata):
                prepare(root, scratch, 'debug')
            self.assertTrue((output / 'ArsenalCore.build/A.o').exists())
            self.assertTrue((output / 'MCP.build/A.o').exists())
            self.assertFalse((output / 'ArsenalAPI.build/A.o').exists())
            self.assertTrue((output / 'ArsenalAPI.build/output-file-map.json').exists())
            self.assertFalse((output / 'arsenal_testserver.build/A.o').exists())
            self.assertFalse((output / 'ArsenalAppTests.build/A.o').exists())

    def test_receipt_rejects_modified_or_missing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'output'; path.write_text('signed output')
            receipt = Path(directory) / 'receipt.json'
            command = [sys.executable, str(Path(__file__).with_name('build-cache.py'))]
            args = ['--key', 'source', '--receipt', str(receipt), str(path)]
            subprocess.run(command + ['record'] + args, check=True)
            self.assertEqual(subprocess.run(command + ['check'] + args).returncode, 0)
            path.write_text('tampered')
            self.assertEqual(subprocess.run(command + ['check'] + args).returncode, 1)
            path.unlink()
            self.assertEqual(subprocess.run(command + ['check'] + args).returncode, 1)

    def test_work_lock_serializes_and_allows_nested_commands(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(dir=root / 'tmp') as directory:
            lock_script = Path(directory) / 'scripts/mlock'; lock_script.parent.mkdir()
            shutil.copy2(root / 'scripts/mlock', lock_script)
            ready = Path(directory) / 'ready'; release = Path(directory) / 'release'
            owner_code = 'import pathlib,time,sys; p=pathlib.Path(sys.argv[1]); p.write_text("ready"); r=pathlib.Path(sys.argv[2]); end=time.monotonic()+5;\nwhile not r.exists() and time.monotonic()<end: time.sleep(.02)'
            owner = subprocess.Popen([str(lock_script), 'work', 'lock-owner', '--', sys.executable, '-c', owner_code, str(ready), str(release)])
            try:
                import time
                end = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < end: time.sleep(.02)
                self.assertTrue(ready.exists())
                waiter = subprocess.Popen([str(lock_script), 'work', 'lock-waiter', '--', sys.executable, '-c', 'pass'])
                try:
                    with self.assertRaises(subprocess.TimeoutExpired): waiter.wait(timeout=.1)
                    release.touch(); self.assertEqual(owner.wait(timeout=5), 0)
                    self.assertEqual(waiter.wait(timeout=5), 0)
                finally:
                    if waiter.poll() is None: waiter.kill(); waiter.wait()
            finally:
                release.touch()
                if owner.poll() is None: owner.kill(); owner.wait()
            subprocess.run([str(lock_script), 'work', 'outer', '--', str(lock_script), 'work', 'inner', '--', sys.executable, '-c', 'pass'], check=True, timeout=5)

    def test_entry_scripts_take_the_lock_once_from_a_symlinked_checkout(self):
        import os
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(dir=root / 'tmp') as directory:
            real = Path(directory) / 'real'; (real / 'scripts').mkdir(parents=True)
            for name in ['mlock', 'check.sh', 'ready.sh']: shutil.copy2(root / 'scripts' / name, real / 'scripts' / name)
            (real / 'scripts/dev-lib.sh').write_text('entry_log() { :; }\nentry_jobs() { :; }\ntimed() { shift; "$@"; }\n')
            for name in ['test.sh', 'check-identity.sh', 'build.sh']:
                stub = real / 'scripts' / name; stub.write_text('#!/bin/sh\necho "ran $0 $ARSENAL_WORK_LOCKED_ROOT"\n'); stub.chmod(0o755)
            link = Path(directory) / 'link'; link.symlink_to(real)
            env = {k: v for k, v in os.environ.items() if not k.startswith('ARSENAL_')}
            for name in ['check.sh', 'ready.sh']:
                result = subprocess.run(['/bin/bash', str(link / 'scripts' / name)], env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('test.sh ' + str(real.resolve()), result.stdout)

    def test_zero_test_targets_fail_even_when_the_runner_exits_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(Path(__file__).with_name('test-targets.py').resolve()), '--', sys.executable, '-c', 'print("Test run with 0 tests passed")']
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            receipt = json.loads((Path(directory) / '.build/test-results/last.json').read_text())
            self.assertEqual({item['target'] for item in receipt['results']}, set(__import__('incremental').package_graph(Path(__file__).resolve().parent.parent)['tests']))
            self.assertTrue(all(item['tests'] == 0 and not item['passed'] for item in receipt['results']))

    def test_changed_selector_keeps_all_transitive_target_suites(self):
        selector_spec = importlib.util.spec_from_file_location('test_targets', Path(__file__).with_name('test-targets.py'))
        selector = importlib.util.module_from_spec(selector_spec); selector_spec.loader.exec_module(selector)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            for name in ['Sources/ArsenalAPI/A.swift', 'Sources/Arsenal/A.swift', 'Sources/ArsenalCore/A.swift']:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('before')
            subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Arsenal tests', '-c', 'user.email=tests@example.invalid', 'commit', '-qm', 'fixture'], check=True)
            for path, expected in [('Sources/ArsenalAPI/A.swift', ['ArsenalAPITests', 'ArsenalAppTests']), ('Sources/Arsenal/A.swift', ['ArsenalAppTests']), ('Sources/ArsenalCore/A.swift', selector.TARGETS)]:
                file = root / path; file.write_text('after')
                self.assertEqual(selector.changed_targets(root, targets=selector.TARGETS, dependencies=DEPENDENCIES), expected)
                file.write_text('before')
            reader = root / 'Tests/ArsenalCoreTests/UpdateTests.swift'
            reader.parent.mkdir(parents=True)
            reader.write_text('let source = repo.appendingPathComponent("Sources/Arsenal/A.swift")')
            subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Arsenal tests', '-c', 'user.email=tests@example.invalid', 'commit', '-qm', 'reader'], check=True)
            (root / 'Sources/Arsenal/A.swift').write_text('after')
            self.assertEqual(selector.changed_targets(root, targets=selector.TARGETS, dependencies=DEPENDENCIES), ['ArsenalCoreTests', 'ArsenalAppTests'])
            (root / 'Sources/Arsenal/A.swift').write_text('before')
            (root / 'new-resource.json').write_text('{}')
            self.assertEqual(selector.changed_targets(root, targets=selector.TARGETS, dependencies=DEPENDENCIES), selector.TARGETS)

    def test_raw_log_bytes_do_not_hide_successful_test_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(Path(__file__).with_name('test-targets.py').resolve()), '--', sys.executable, '-c', 'import sys; sys.stdout.buffer.write(b"\\xff\\nTest run with 1 tests passed\\n")']
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0)
            receipt = json.loads((Path(directory) / '.build/test-results/last.json').read_text())
            self.assertTrue(all(item['tests'] == 1 and item['passed'] for item in receipt['results']))
            self.assertIn(b'\xff', (Path(directory) / '.build/test-results/ArsenalCoreTests.log').read_bytes())

    def test_single_test_summary_passes_the_filtered_count_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(Path(__file__).with_name('test-targets.py').resolve()), '--filtered', '--', sys.executable, '-c', 'print("Test run with 1 test in 1 suite passed")']
            result = subprocess.run(command, cwd=directory, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((Path(directory) / '.build/test-results/filtered.json').read_text())
            self.assertEqual(receipt['results'][0]['tests'], 1)

    def test_key_tracks_content_and_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input'; path.write_text('before')
            initial = cache.key([path], 'toolchain one')
            self.assertNotEqual(initial, cache.key([path], 'toolchain two'))
            path.write_text('after')
            self.assertNotEqual(initial, cache.key([path], 'toolchain one'))
            with self.assertRaises(FileNotFoundError): cache.hashes([path.with_name('missing')])

    def test_candidate_tracks_ignored_checkout_content_and_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / '.gitignore').write_text('.build/\nWorkers/.build/\n')
            subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Tests', '-c', 'user.email=tests@example.invalid', 'commit', '-qm', 'fixture'], check=True)
            for location in ['.build/checkouts/dep', 'Workers/.build/checkouts/dep']:
                checkout = root / location; checkout.mkdir(parents=True)
                subprocess.run(['git', 'init', '-q', str(checkout)], check=True)
                source = checkout / 'Dependency.swift'; source.write_text('before')
                subprocess.run(['git', '-C', str(checkout), 'add', '.'], check=True)
                subprocess.run(['git', '-C', str(checkout), '-c', 'user.name=Tests', '-c', 'user.email=tests@example.invalid', 'commit', '-qm', 'dependency'], check=True)
                initial = cache.candidate_key(root)
                source.write_text('after')
                self.assertNotEqual(initial, cache.candidate_key(root))
                source.write_text('before')
                subprocess.run(['git', '-C', str(checkout), '-c', 'user.name=Tests', '-c', 'user.email=tests@example.invalid', 'commit', '--allow-empty', '-qm', 'new revision'], check=True)
                self.assertNotEqual(initial, cache.candidate_key(root))

    def test_fallback_requires_positive_counts_and_preserves_failure(self):
        checker = Path(__file__).with_name('test-run-checked.py')
        for output, code, expected in [('Test run with 0 tests passed', 0, 1), ('Test run with 2 tests passed', 0, 0), ('Executed 1 test, with 0 failures', 0, 0), ('Test run with 2 tests failed', 7, 7), ('', 0, 1)]:
            result = subprocess.run([sys.executable, str(checker), '--', sys.executable, '-c', 'import sys; print(sys.argv[1]); sys.exit(int(sys.argv[2]))', output, str(code)], capture_output=True, timeout=10)
            self.assertEqual(result.returncode, expected, result.stderr)
        for flag in ['--help', '-h', 'list', '--list-tests']:
            result = subprocess.run([sys.executable, str(checker), '--inspect', '--', sys.executable, '-c', 'pass', flag], capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, str(checker), '--', sys.executable, '-c', 'pass', '--filter', 'list'], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 1)

    def test_shell_fallbacks_gate_counts_and_only_exempt_inspection(self):
        import os
        scripts = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'scripts').mkdir(); (root / '.build').mkdir()
            (root / 'scripts/test.sh').write_text(scripts.joinpath('test.sh').read_text())
            shutil.copy2(scripts / 'test-run-checked.py', root / 'scripts/test-run-checked.py')
            (root / 'scripts/dev-lib.sh').write_text('CLT="$PWD/toolchain"\nFRAMEWORKS="$CLT/frameworks"\nswift_debug() { "$CLT/usr/bin/swift" "$@"; }\ntimed() { shift; "$@"; }\n')
            lint = root / 'scripts/lint.sh'; lint.write_text('#!/bin/sh\nexit 0\n'); lint.chmod(0o755)
            (root / 'scripts/incremental.py').write_text('pass\n')
            swift = root / 'toolchain/usr/bin/swift'; swift.parent.mkdir(parents=True)
            swift.write_text('#!/bin/sh\nprintf "%s\\n" "$FIXTURE_OUTPUT"\nexit "$FIXTURE_EXIT"\n'); swift.chmod(0o755)
            env = {**os.environ, 'ARSENAL_DEV_LOCKED': '1', 'FIXTURE_OUTPUT': 'Test run with 0 tests passed', 'FIXTURE_EXIT': '0'}
            def run(args, expected):
                result = subprocess.run(['/bin/bash', str(root / 'scripts/test.sh'), *args], env=env, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, expected, (args, result.stdout, result.stderr))
            for args in [['--skip-build'], ['--parallel'], ['--no-parallel'], ['--enable-code-coverage'], ['--maximum-repetitions', '2'], ['--repeat-until', 'failure'], ['--package-path', 'somewhere']]:
                run(args + ['--filter', 'DoesNotExist'], 1)
                env['FIXTURE_OUTPUT'] = 'Test run with 1 test passed'; run(args, 0)
                env['FIXTURE_EXIT'] = '7'; run(args, 7)
                env['FIXTURE_EXIT'] = '0'; env['FIXTURE_OUTPUT'] = 'Test run with 0 tests passed'
            for args in [['--help'], ['-h'], ['list'], ['--list-tests']]: run(args, 0)
            run(['--skip-build', '--filter', 'list'], 1)

    def test_abi_receipt_rejects_source_changes_during_compilation(self):
        import incremental
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'Sources/ArsenalCore/A.swift'
            source.parent.mkdir(parents=True); source.write_text('before')
            scratch = root / '.build'
            metadata = {'key': 'fixture', 'dependencies': DEPENDENCIES, 'folders': {'ArsenalCore': 'Sources/ArsenalCore'}}
            with patch('incremental.package_graph', return_value=metadata):
                prepare(root, scratch, 'debug')
                incremental.record(root, scratch, 'debug')
                prepare(root, scratch, 'debug')
                source.write_text('changed while compiling')
                with self.assertRaisesRegex(RuntimeError, 'changed during compilation'):
                    incremental.record(root, scratch, 'debug')
                self.assertFalse((scratch / 'arsenal-abi-debug.json').exists())

    def test_metal_identity_tracks_compiler_and_sdk_before_candidate_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sdk = root / 'SDK'; sdk.mkdir()
            settings = sdk / 'SDKSettings.json'; settings.write_text('SDK one')
            metal = root / 'metal'; metal.write_text('compiler one')
            metallib = root / 'metallib'; metallib.write_text('linker one')
            version = 'Metal one'
            def xcrun(command, **kwargs):
                self.assertEqual(kwargs['env']['DEVELOPER_DIR'], str(root))
                return {('--show-sdk-path',): str(sdk), ('--show-sdk-version',): '26', ('--find', 'metal'): str(metal), ('--find', 'metallib'): str(metallib), ('metal', '--version'): version}[tuple(command[1:])]
            with patch.object(cache.subprocess, 'check_output', side_effect=xcrun):
                initial = cache.metal_identity(root)
                for file in [settings, metal, metallib]:
                    old = file.read_text(); file.write_text('changed')
                    self.assertNotEqual(initial, cache.metal_identity(root)); file.write_text(old)
                version = 'Metal two'; self.assertNotEqual(initial, cache.metal_identity(root))
                version = 'Metal one'
            build = Path(__file__).with_name('build.sh').read_text()
            self.assertIn('scripts/build-cache.py metal', build)
            self.assertLess(build.index('scripts/build-cache.py metal'), build.index('scripts/build-cache.py check --key "$candidate_key"'))
            self.assertLess(build.index('rev-parse HEAD)" == "$MLX_SWIFT_PIN"'), build.index('scripts/build-cache.py check --key "$candidate_key"'))
            self.assertIn('$source_key|$metal_identity|', build)

    def test_app_shards_run_every_listed_test_once_each_under_its_own_name(self):
        """Shards split by suite; the last takes the rest by --skip; their counts must equal the list."""
        selector_spec = importlib.util.spec_from_file_location('test_targets', Path(__file__).with_name('test-targets.py'))
        selector = importlib.util.module_from_spec(selector_spec); selector_spec.loader.exec_module(selector)
        listing = ['ArsenalAppTests.Big/a()', 'ArsenalAppTests.Big/b(x:)', 'ArsenalAppTests.Small/a()', 'ArsenalAppTests.Mid/a()',
                   'ArsenalAppTests.Mid/b()', 'ArsenalAppTests.MidExtra/a()', 'ArsenalAppTests.freeFunction()', 'ArsenalCoreTests.Core/a()']
        suites = selector.listed_suites(listing, 'ArsenalAppTests')
        self.assertEqual(suites, {'Big': 2, 'Small': 1, 'Mid': 2, 'MidExtra': 1})
        groups = selector.shard_suites(suites, {'Big': 50.0, 'Small': 1.0}, 3)
        self.assertEqual(sorted(name for group in groups for name in group), sorted(suites))
        self.assertEqual(groups[0], ['Big'])
        # A test runner stand-in: swift-testing's --filter/--skip regex search over the listed IDs.
        runner = 'import os,re,sys\nids=open(os.environ["FAKE_LIST"]).read().split()\nargs=sys.argv[1:]\n' \
                 'inc=[args[i+1] for i,a in enumerate(args) if a=="--filter"]\nexc=[args[i+1] for i,a in enumerate(args) if a=="--skip"]\n' \
                 'ran=[t for t in ids if any(re.search(p,t) for p in inc) and not any(re.search(p,t) for p in exc) and t not in os.environ.get("FAKE_DROP","").split()]\n' \
                 'open(os.environ["FAKE_RAN"],"a").write("".join(os.path.basename(sys.argv[0])+" "+t+"\\n" for t in ran))\n' \
                 'print("Test run with %d tests passed" % len(ran))\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / '.build').mkdir()
            graph = __import__('incremental').package_graph(Path(__file__).resolve().parent.parent)
            listing += [target + '.Fixture/a()' for target in graph['tests'] if target != 'ArsenalAppTests']
            (root / '.build/test-list.txt').write_text('\n'.join(listing) + '\n')
            (root / 'list').write_text('\n'.join(listing))
            script = root / 'runner.py'; script.write_text('#!' + sys.executable + '\n' + runner); script.chmod(0o755)
            def run(drop=''):
                ran = root / 'ran'; ran.write_text('')
                env = {**__import__('os').environ, 'FAKE_LIST': str(root / 'list'), 'FAKE_RAN': str(ran), 'FAKE_DROP': drop, 'ARSENAL_APP_SHARDS': '3'}
                result = subprocess.run([sys.executable, str(Path(__file__).with_name('test-targets.py').resolve()), '--', str(script)],
                                        cwd=directory, capture_output=True, text=True, timeout=30, env=env)
                return result, [line.split(' ', 1) for line in ran.read_text().splitlines()]
            result, ran = run()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            app = [(name, test) for name, test in ran if test.startswith('ArsenalAppTests.')]
            self.assertEqual(sorted(test for _, test in app), sorted(t for t in listing if t.startswith('ArsenalAppTests.')))
            self.assertEqual({name for name, _ in app}, {'arsenal-app-tests-1', 'arsenal-app-tests-2', 'arsenal-app-tests-3'})
            receipt = json.loads((root / '.build/test-results/last.json').read_text())
            entry = next(item for item in receipt['results'] if item['target'] == 'ArsenalAppTests')
            self.assertEqual((entry['tests'], entry['listed'], len(entry['shards'])), (7, 7, 3))
            # A test the shards never reach fails the run even though every shard passed on its own.
            result, _ = run(drop='ArsenalAppTests.Mid/b()')
            self.assertEqual(result.returncode, 1)
            self.assertIn('shards ran 6 tests, the list has 7', result.stderr)

    def test_release_layout_edits_rebuild_root_and_native_consumers(self):
        """Both production configurations use WMO, whose dependents must track layout edits."""
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(name, text):
                file = root / name; file.parent.mkdir(parents=True, exist_ok=True); file.write_text(text)
            write('Package.swift', '''// swift-tools-version: 6.0
import PackageDescription
let package = Package(name: "RootProbe", products: [.library(name: "Producer", targets: ["Producer"]), .executable(name: "root-probe", targets: ["root-probe"])], targets: [.target(name: "Wire"), .target(name: "Producer", dependencies: ["Wire"]), .executableTarget(name: "root-probe", dependencies: ["Producer"])])
''')
            write('Sources/Wire/Wire.swift', 'public let wireValue = 11\n')
            original = 'import Wire\npublic struct Value { public let x:Int; public init() { x = wireValue } }\n'
            padded = 'import Wire\npublic struct Value { public let padding:Int; public let x:Int; public init() { padding = 99; x = wireValue } }\n'
            write('Sources/Producer/Value.swift', original)
            write('Sources/root-probe/main.swift', 'import Producer\nprint(Value().x)\n')
            write('Workers/Package.swift', '''// swift-tools-version: 6.0
import PackageDescription
let package = Package(name: "NativeProbe", dependencies: [.package(name: "RootProbe", path: "..")], targets: [.target(name: "Worker", dependencies: [.product(name: "Producer", package: "RootProbe")], path: "Native/Sources/Worker"), .executableTarget(name: "native-probe", dependencies: ["Worker"], path: "Native/Sources/native-probe")])
''')
            write('Workers/Native/Sources/Worker/Worker.swift', 'import Producer\npublic func result() -> Int { Value().x }\n')
            write('Workers/Native/Sources/native-probe/main.swift', 'import Worker\nprint(result())\n')
            env = {**os.environ, 'DEVELOPER_DIR': '/Library/Developer/CommandLineTools'}
            for package, executable, flags in [(root, 'root-probe', ['-Xswiftc', '-Osize', '-Xlinker', '-no_exported_symbols']), (root / 'Workers', 'native-probe', ['-Xlinker', '-no_exported_symbols'])]:
                write('Sources/Producer/Value.swift', original)
                command = ['/Library/Developer/CommandLineTools/usr/bin/swift', 'build', '--package-path', str(package), '-c', 'release', '--build-system', 'native', '--jobs', '2', *flags]
                for source in [original, padded]:
                    write('Sources/Producer/Value.swift', source)
                    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    output = subprocess.check_output([str(package / '.build/release' / executable)], text=True, timeout=10)
                    self.assertEqual(output.strip(), '11')
                    wire = next((package / '.build').glob('*/release/Wire.build/*.o'))
                    if source == original: stamp = wire.stat().st_mtime_ns
                    else: self.assertEqual(stamp, wire.stat().st_mtime_ns)
                print('release layout probe: ' + executable + ' retained Wire; output 11 before/after', flush=True)


if __name__ == '__main__': unittest.main()
