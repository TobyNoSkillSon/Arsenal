#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deterministic package-closure and single-runtime bundle regressions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('native_layout', Path(__file__).with_name('native-layout.py'))
layout = importlib.util.module_from_spec(spec); spec.loader.exec_module(layout)


class NativeLayoutTests(unittest.TestCase):
    def test_resources_follow_linked_modules_and_ignore_stale_test_bundles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = {}
            for module in ['Engine', 'TestFixtures', 'Cmlx']:
                accessor = root / (module + '.build/DerivedSources/resource_bundle_accessor.swift')
                accessor.parent.mkdir(parents=True)
                bundle = 'mlx-swift_Cmlx.bundle' if module == 'Cmlx' else 'Family_' + module + '.bundle'
                accessor.write_text(f'let mainPath = Bundle.main.bundleURL.appendingPathComponent("{bundle}").path')
                (root / bundle).mkdir()
                commands[module] = {'moduleName': module, 'sources': [str(accessor)]}
            (root / 'description.json').write_text(json.dumps({
                'targetDependencyMap': {'arsenal_worker_native': ['Engine', 'Cmlx']}, 'swiftCommands': commands}))
            self.assertEqual(layout.resource_bundles(root), [root / 'Family_Engine.bundle'])
            (root / 'Family_Engine.bundle').rmdir()
            with self.assertRaisesRegex(ValueError, 'Missing linked resource'):
                layout.resource_bundles(root)

    def test_unknown_build_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'description.json').write_text('{"targetDependencyMap":{},"swiftCommands":{}}')
            with self.assertRaisesRegex(ValueError, 'Missing SwiftPM'):
                layout.resource_bundles(root)

    def test_compaction_preserves_token_strings_and_exact_numeric_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'tokenizer.json'
            original = b' { "token": "two words \\" \\n \\u0020", "id": 9007199254740993, "score": 1.234567890123456789e-12 }\n'
            path.write_bytes(original)
            # Include escaped quotes, backslashes, whitespace tokens, Unicode and long numbers.
            before = json.loads(original)
            saved = layout.compact_resources(root)
            self.assertGreater(saved, 0)
            self.assertEqual(json.loads(path.read_bytes()), before)
            self.assertIn(b'1.234567890123456789e-12', path.read_bytes())
            self.assertIn(b'9007199254740993', path.read_bytes())
            self.assertEqual(layout.compact_resources(root), 0)

    def test_layout_rejects_separate_workers_and_duplicate_metal(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / 'Arsenal.app'; helpers = app / 'Contents/Helpers'; helpers.mkdir(parents=True)
            (helpers / 'arsenal-worker-native').write_bytes(b'native')
            metal = helpers / 'mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib'
            metal.parent.mkdir(parents=True); metal.write_bytes(b'metal')
            for family in ['audio', 'perception']:
                (helpers / ('arsenal-worker-' + family)).symlink_to('arsenal-worker-native')
            receipt = layout.inventory(app, ['audio', 'perception'])
            self.assertEqual(receipt['bytes'], 11)
            self.assertEqual(receipt['components_bytes']['native'], 6)
            self.assertEqual(receipt['budget_bytes'], 45 * 1048576)
            (helpers / 'arsenal-worker-audio').unlink(); (helpers / 'arsenal-worker-audio').write_bytes(b'copy')
            with self.assertRaisesRegex(ValueError, 'relative link'):
                layout.inventory(app, ['audio', 'perception'])
            (helpers / 'arsenal-worker-audio').unlink(); (helpers / 'arsenal-worker-audio').symlink_to('arsenal-worker-native')
            duplicate = helpers / 'other/default.metallib'; duplicate.parent.mkdir(); duplicate.write_bytes(b'copy')
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                layout.inventory(app, ['audio', 'perception'])


if __name__ == '__main__':
    unittest.main()
