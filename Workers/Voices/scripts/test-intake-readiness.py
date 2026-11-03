#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('readiness', Path(__file__).with_name('intake-readiness.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReadinessTests(unittest.TestCase):
    def test_all_models_have_exactly_three_blocked_tiers_and_no_pretend_measurements(self):
        for model in module.MODELS:
            rows = module.blocked_rows(model)
            self.assertEqual([r['precision'] for r in rows], ['fp16', 'int8', 'int4'])
            self.assertTrue(all(r['state'] == 'blocked' and r['quality_binding'] is None for r in rows))
            self.assertTrue(all(r['qualified'] is None and r['peak_gb'] is None and r['warm_seconds'] is None for r in rows))
            self.assertTrue(all('component-capture adapter' in r['implementation_pending'][0] for r in rows))

    def test_header_only_coverage_includes_nested_codec_and_conditioning(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / 'codec/model.safetensors'
            path.parent.mkdir()
            header = json.dumps({'encoder.linear.weight': {'dtype': 'F32', 'shape': [256, 256], 'data_offsets': [0, 262144]},
                                 'encoder.conv.weight': {'dtype': 'F32', 'shape': [256, 3, 256], 'data_offsets': [262144, 1048576]}}).encode()
            path.write_bytes(struct.pack('<Q', len(header)) + header + bytes(1048576))
            result = module.inventory('qwen3-tts-base', root, ['codec/model.safetensors'])
            self.assertEqual(result['packed_matrices'], 0)
            self.assertEqual(len(result['tensors']), 2)
            self.assertEqual(result['components']['codec/encoder']['eligible_matrices'], 1)
            self.assertEqual(result['components']['codec/encoder']['parameters'], 262144)

    def test_truncated_headers_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'bad.safetensors'
            for data in [b'bad', struct.pack('<Q', 100) + b'{}']:
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    module.descriptors(path)

    def test_selection_keeps_only_explicitly_ineligible_tensors(self):
        self.assertIsNone(module.skip_reason('flow_lm.flow_net.weight', [256, 256]))
        for key, shape in [('norm.weight', [256]), ('proj.weight', [1024, 65]), ('small.weight', [4, 64]), ('audio_prompt', [1, 8, 256])]:
            self.assertIsNotNone(module.skip_reason(key, shape))


if __name__ == '__main__':
    unittest.main()
