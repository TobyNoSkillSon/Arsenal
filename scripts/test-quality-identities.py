#!/usr/bin/env python3
"""Identity changes follow source dependencies; unrelated worker changes do not stale all models."""
import importlib.util, pathlib, tempfile
spec = importlib.util.spec_from_file_location('identities', pathlib.Path(__file__).with_name('quality-identities.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as temp:
    root = pathlib.Path(temp)
    def put(path, text):
        file = root / path; file.parent.mkdir(parents=True, exist_ok=True); file.write_text(text)
    put('Workers/Voices/Sources/VireoEngine/Quantization.swift', 'original')
    put('Workers/Sound/Sources/SoundEngine/Quantization.swift', 'original')
    before = m.identities(root)
    put('Workers/Voices/Sources/VireoEngine/Quantization.swift', 'changed')
    after = m.identities(root)
    assert before['voices'] != after['voices'] and before['sound'] == after['sound']
    put('Workers/Voices/.build/Fake.swift', 'ignored')
    assert after == m.identities(root)
    put('Packages/ModelPrecision/Sources/ModelPrecision/StreamingWeights.swift', 'changed')
    assert all(value != m.identities(root)[key] for key, value in after.items())
    before = m.identities(root)
    put('models/birefnet/runners/swift/Sources/MLXBiRefNet/WeightLoading.swift', 'changed')
    after = m.identities(root)
    assert before['birefnet'] != after['birefnet']
    assert all(value == after[key] for key, value in before.items() if key != 'birefnet')
    before = m.identities(root)
    put('Workers/QwenImage/Sources/QwenImageNative/Transformer.swift', 'changed')
    after = m.identities(root)
    assert before['qwenimage'] != after['qwenimage']
    assert all(value == after[key] for key, value in before.items() if key != 'qwenimage')
print('quality identity source-closure checks: 5 passed')
