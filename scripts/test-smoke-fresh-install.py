#!/usr/bin/env python3
"""CPU-only checks for the fresh-install smoke's source receipt."""
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('smoke', pathlib.Path(__file__).with_name('smoke-fresh-install.py'))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class ReceiptTests(unittest.TestCase):
    def fixture(self, root, kind):
        support = root / 'support'
        snapshot = support / 'weights/snapshots/kokoro-82m/revision/fp16'
        snapshot.mkdir(parents=True)
        original = b'original source'
        derived = b'derived fp16'
        source_pin = {'path': 'original.safetensors', 'bytes': len(original), 'sha256': hashlib.sha256(original).hexdigest()}
        installed_pin = {'path': 'model.safetensors', 'bytes': len(derived), 'sha256': hashlib.sha256(derived).hexdigest()}
        (snapshot / installed_pin['path']).write_bytes(derived)
        (snapshot / 'arsenal-snapshot.json').write_text(json.dumps({
            'model': 'kokoro-82m', 'precision': 'fp16', 'revision': 'revision',
            'files': [installed_pin], 'derived': {'sourceFiles': [source_pin]},
        }))
        verified = support / 'weights/verified'
        verified.mkdir()
        record = verified / (source_pin['sha256'] + '.json')
        record.write_text(json.dumps({'sha256': source_pin['sha256'], 'identity': {'bytes': len(original)}, 'sourceKind': kind}))
        return support, snapshot, record

    def test_preserves_labels_and_records_verified_source_and_installed_hashes(self):
        for label, network, local, kind in [('network', 15, 0, 'download'), ('verified-local', 0, 15, 'localCopy'), ('bundled-or-ready', 0, 0, 'localCopy')]:
            with self.subTest(source=label), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                support, _, _ = self.fixture(root, kind)
                receipt = smoke.write_source_receipt(support, root, 'kokoro-82m', 'fp16', {'network_bytes': network, 'items': [{'localCopyBytes': local}]})
                self.assertEqual(receipt['source'], label)
                self.assertEqual((receipt['model'], receipt['dtype']), ('kokoro-82m', 'fp16'))
                self.assertEqual(receipt['verification']['source_files'][0]['source_kind'], kind)
                self.assertEqual(len(receipt['verification']['installed_files'][0]['sha256']), 64)
                self.assertEqual(json.loads((root / 'source.json').read_text()), receipt)
                self.assertFalse((root / 'source.json.tmp').exists())

    def test_bad_installed_hash_or_source_admission_prevents_receipt(self):
        for failure in ['installed_hash', 'source_admission']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                support, snapshot, record = self.fixture(root, 'download')
                if failure == 'installed_hash':
                    (snapshot / 'model.safetensors').write_bytes(b'corrupt fp16!')
                else:
                    data = json.loads(record.read_text())
                    data['identity']['bytes'] += 1
                    record.write_text(json.dumps(data))
                with self.assertRaises(AssertionError):
                    smoke.write_source_receipt(support, root, 'kokoro-82m', 'fp16', {'network_bytes': 15, 'items': []})
                self.assertFalse((root / 'source.json').exists())

    def test_failed_get_does_not_write_receipt(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(smoke, 'install_deadline'), patch.object(smoke, 'Client') as client_type:
            client = client_type.return_value
            client.card.return_value = {'downloadBytes': {'fp16': 15}, 'localCopyBytes': {}}
            client.tool.return_value = {'status': 'running', 'job': '0123456789'}
            client.wait.return_value = {'status': 'failed'}
            with patch.object(smoke.sys, 'argv', ['smoke', 'get', directory, 'kokoro-82m', '/lab/Arsenal.app', directory]), self.assertRaises(AssertionError):
                smoke.main()
            client.close.assert_called_once()
            self.assertFalse((pathlib.Path(directory) / 'source.json').exists())


if __name__ == '__main__':
    unittest.main()
