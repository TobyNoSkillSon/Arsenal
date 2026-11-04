#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Parser, cleanup boundary and real LaunchServices fixture lifecycle checks; no app launch."""
import pathlib
import importlib.util
import plistlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import launch_services as ls


def record(path, identifier=ls.PRODUCTION_ID):
    return f'bundle id: fixture (0x1)\npath: {path} (0x123)\nidentifier: {identifier}\n'


class LaunchServicesTests(unittest.TestCase):
    def test_only_installed_is_allowed(self):
        self.assertEqual(ls.check(record(ls.INSTALLED)), [str(ls.INSTALLED)])

    def test_homebrew_install_is_installed_too(self):
        cask = '/Applications/Arsenal.app'
        self.assertEqual(ls.check(record(cask) + record(ls.INSTALLED)), [cask, str(ls.INSTALLED)])
        with patch.object(ls.subprocess, 'run') as run, self.assertRaisesRegex(RuntimeError, 'installed Arsenal'):
            ls.unregister([pathlib.Path(cask)])
        run.assert_not_called()

    def test_intermediates_and_rollback_fail(self):
        for path in ['/lab/.build/pkg/stage/Arsenal.app', '/lab/dist/Arsenal.app', str(ls.INSTALLED)+'.previous', '/Applications/Arsenal.app.previous', '/lab with spaces/Arsenal.app']:
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, 'Unexpected production'):
                ls.check(record(ls.INSTALLED) + record(path))

    def test_identifiers_are_exact(self):
        dump = record('/tmp/Arsenal.app', 'dev.arsenal.app.fixture') + record('/tmp/Other.app', 'dev.arsenal.rollback-test')
        self.assertEqual(ls.check(dump), [])

    def test_parser_retains_spaces_and_drops_numeric_suffix(self):
        self.assertEqual(list(ls.registrations(record('/a path/Arsenal.app'))),
                         [{'path': '/a path/Arsenal.app', 'identifier': ls.PRODUCTION_ID}])

    def test_installed_cannot_be_unregistered(self):
        with patch.object(ls.subprocess, 'run') as run, self.assertRaisesRegex(RuntimeError, 'installed Arsenal'):
            ls.unregister([ls.INSTALLED])
        run.assert_not_called()

    def test_scan_error_is_accepted_only_if_registration_is_gone(self):
        result = subprocess.CompletedProcess([], 1, 'failed to scan: -10814', '')
        with patch.object(ls.subprocess, 'run', return_value=result), patch.object(ls, 'dump_registrations', return_value=[]):
            ls.unregister(['/tmp/deleted-fixture/Arsenal.app'])
        rows = [{'path': '/tmp/deleted-fixture/Arsenal.app', 'identifier': 'dev.arsenal.fixture'}]
        with patch.object(ls.subprocess, 'run', return_value=result), patch.object(ls, 'dump_registrations', return_value=rows), patch.object(ls.time, 'sleep'), self.assertRaisesRegex(RuntimeError, 'registrations remain'):
            ls.unregister(['/tmp/deleted-fixture/Arsenal.app'])

    def test_outstanding_registration_is_retried_with_a_bound(self):
        path = '/tmp/owned-fixture/Arsenal.app'
        rows = [{'path': path, 'identifier': 'dev.arsenal.fixture'}]
        result = subprocess.CompletedProcess([], 0, '', '')
        with patch.object(ls.subprocess, 'run', return_value=result) as run, \
                patch.object(ls, 'dump_registrations', side_effect=[rows, []]), patch.object(ls.time, 'sleep'):
            ls.unregister([path])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0], [ls.LSREGISTER, '-u', path])

    def test_cleanup_includes_deleted_paths_and_renamed_bundles_only_inside_root(self):
        with tempfile.TemporaryDirectory(prefix='arsenal-ls-parser-') as directory:
            root = pathlib.Path(directory).resolve()
            info = root/'Arsenal.app.previous/Contents/Info.plist'
            info.parent.mkdir(parents=True); info.write_bytes(b'fixture')
            deleted = root/'gone/Arsenal.app'
            rows = [{'path': str(deleted), 'identifier': 'dev.arsenal.fixture'},
                    {'path': str(root)+'-other/Arsenal.app', 'identifier': ls.PRODUCTION_ID},
                    {'path': str(ls.INSTALLED), 'identifier': ls.PRODUCTION_ID}]
            with patch.object(ls, 'dump_registrations', return_value=rows), patch.object(ls, 'unregister') as unregister:
                ls.cleanup_tree(root)
            self.assertEqual(unregister.call_args.args[0], {info.parent.parent, deleted})

    def test_real_fixture_registration_is_removed_after_rename(self):
        with tempfile.TemporaryDirectory(prefix='arsenal-ls-fixture-') as directory:
            root = pathlib.Path(directory).resolve()
            app = root/'Arsenal.app'; contents = app/'Contents'
            contents.mkdir(parents=True)
            fixture_id = 'dev.arsenal.launchservices-fixture'
            (contents/'Info.plist').write_bytes(plistlib.dumps({
                'CFBundleIdentifier': fixture_id, 'CFBundlePackageType': 'APPL', 'CFBundleName': 'Arsenal Registration Fixture'}))
            try:
                subprocess.run([ls.LSREGISTER, '-f', str(app)], check=True, capture_output=True, timeout=15)
                self.assertTrue(any(r['identifier'] == fixture_id and r['path'] == str(app) for r in ls.dump_registrations()))
                app.rename(root/'Arsenal.app.previous')
            finally:
                ls.cleanup_tree(root)
            self.assertFalse(any(root in pathlib.Path(r['path']).parents for r in ls.dump_registrations()))

    def test_updater_cleanup_continues_after_unreadable_stage_record(self):
        spec = importlib.util.spec_from_file_location('update_fixtures', pathlib.Path(__file__).with_name('test-updates.py'))
        updates = importlib.util.module_from_spec(spec); spec.loader.exec_module(updates)
        owned = []
        deleted_keychains = []
        def fail_before_build(root):
            owned.append(root)
            updates.TEST_KEYCHAIN = root/'fixture-signing.keychain'
            updates.TEST_KEYCHAIN.write_bytes(b'fixture')
            stage = root/'case/stage-dir'; stage.parent.mkdir()
            stage.write_bytes(b'\xff')  # Decoding fails during teardown, before keychain/tree cleanup.
            raise RuntimeError('Injected failure before compilation')
        def fake_run(arguments, **kwargs):
            if list(map(str, arguments))[:2] == ['/usr/bin/security', 'delete-keychain']:
                deleted_keychains.append(arguments[2])
                arguments[2].unlink()
            return ''
        with patch.object(updates, 'disposable_signing_keychain', side_effect=fail_before_build), \
                patch.object(updates, 'run', side_effect=fake_run), \
                patch.object(ls, 'dump_registrations', return_value=[]), \
                patch('sys.argv', ['test-updates', '--cases', 'good', '--skip-idle']), \
                self.assertRaisesRegex(RuntimeError, 'Fixture cleanup failed'):
            updates.main()
        self.assertEqual(deleted_keychains, [updates.TEST_KEYCHAIN])
        self.assertFalse(owned[0].exists())


if __name__ == '__main__':
    unittest.main()
