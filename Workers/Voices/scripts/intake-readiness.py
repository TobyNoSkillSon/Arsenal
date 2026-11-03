#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Agent-side source preflight for the voice rollout; no tensor/model execution or downloads.

Reproduces the 15 Oct 2026 source audit and its recorded reference-search outcome.
Creates a timestamped BLOCKED receipt, never schema-2 table stats or a quality binding.
This is preparation evidence, not an automatic detector of future reference availability.
Hash pinned originals, inspect private APFS clones, hash originals again, then delete only those clones.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import tomllib
import uuid

MODELS = ('breeze-tts-2', 'qwen3-tts-base', 'qwen3-tts-voicedesign', 'pocket-tts')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def descriptors(path):
    with path.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('Truncated archive')
        size = struct.unpack('<Q', raw)[0]
        if size > 64 << 20 or size > path.stat().st_size - 8:
            raise ValueError('Invalid header size')
        header = json.loads(stream.read(size))
    return {k: v for k, v in header.items() if k != '__metadata__'}


def skip_reason(key, shape):
    if not key.endswith('.weight'):
        return 'not a Linear/Embedding weight (buffers, biases, norms stay half)'
    if len(shape) != 2:
        return 'non-matrix layout (convolution/norm stays half)'
    if math.prod(shape) < 65536:
        return 'fewer than 65,536 parameters'
    if shape[1] % 64:
        return 'input width is incompatible with affine group 64'
    return None


def component(model, relative, key):
    if '/' in relative and not relative.startswith('embeddings/'):
        return relative.split('/')[0] + '/' + key.split('.')[0]
    if relative.startswith('embeddings/'):
        return 'preset-conditioning'
    parts = key.split('.')
    if parts[0] in ('talker', 'flow_lm', 'mimi'):
        return '.'.join(parts[:2])
    return parts[0]


def inventory(model, folder, files):
    components = {}
    tensors = []
    for relative in files:
        if not relative.endswith('.safetensors'):
            continue
        for key, item in sorted(descriptors(folder / relative).items()):
            shape, dtype = item['shape'], item['dtype']
            count = math.prod(shape)
            reason = skip_reason(key, shape)
            name = component(model, relative, key)
            entry = components.setdefault(name, dict(tensors=0, parameters=0, source_dtypes={}, eligible_matrices=0, eligible_parameters=0, skipped_reasons={}))
            entry['tensors'] += 1
            entry['parameters'] += count
            entry['source_dtypes'][dtype] = entry['source_dtypes'].get(dtype, 0) + count
            if reason:
                entry['skipped_reasons'][reason] = entry['skipped_reasons'].get(reason, 0) + 1
            else:
                entry['eligible_matrices'] += 1
                entry['eligible_parameters'] += count
            tensors.append(dict(file=relative, key=key, shape=shape, dtype=dtype, component=name, planned='keep-half' if reason else 'affine-g64', reason=reason))
    return dict(method='Header-only inventory on pinned private APFS clones. Eligibility is a conversion plan, not measured coverage or proof of loader compatibility.',
                packed_matrices=0, components=components, tensors=tensors)



def write_inventory(path, data):
    summary = {k: v for k, v in data.items() if k != 'tensors'}
    text = json.dumps(summary, indent=2)[:-2] + ',\n  "tensors": [\n'
    text += ',\n'.join('    ' + json.dumps(t, separators=(',', ':')) for t in data['tensors'])
    path.write_text(text + '\n  ]\n}\n')


def blocked_rows(model):
    cause = ('The pinned speech codec contains FP32 learned parameters. Its original implementation cannot be executed under the no-FP32 rule. '
             'No saved original component tensors/PCM with matched inputs were found in this model gate, the prior Arsenal smoke location, or the inspected Vireo reference locations. '
             'Casting the codec to either half format and comparing it to itself would not qualify the cast. SAME-L acceptance is specific to that decoder and is not voice-codec acceptance.')
    if model == 'pocket-tts':
        cause = ('The default Pocket path forces FlowLM hidden states and flow noise/time inputs through FP32 into learned flow/EOS projections; preset conditioning is stored FP32. '
                 'The BF16 checkpoint alone does not establish a half-arithmetic reference for those paths. '
                 'No retained matched-input half reference was found in this model gate, prior Arsenal smoke location, or inspected Vireo reference locations. '
                 'A half candidate cannot qualify itself.')
    return [dict(precision=tier, state='blocked', qualified=None, max_relative_l2=None, min_cosine=None,
                 end_to_end_metric=None, warm_seconds=None, peak_gb=None, quality_binding=None,
                 blocker=dict(code='half-reference-not-qualified', reason=cause,
                              release_condition='Provide matched-input original captures, an independent GPU half reference, or an explicit accepted cast basis; then finish architecture component adapters, run gates and served-path measurements.'),
                 implementation_pending=['architecture component-capture adapter', 'real-weight stored-pack loader checks', 'format gate', 'quantized component gate', 'served-path sanity and warm/peak measurements', 'pipeline quality bindings'])
            for tier in ('fp16', 'int8', 'int4')]


def run(root, model, cache):
    folder = root / 'models' / model
    manifest = tomllib.loads((folder / 'model.toml').read_text())
    weights = manifest['weights']
    source = cache / ('models--' + weights['repo'].replace('/', '--')) / 'snapshots' / weights['revision']
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')
    run_id = 'intake-' + stamp + '-' + str(uuid.uuid4())
    evidence = folder / 'gate/evidence/intake' / run_id
    evidence.mkdir(parents=True, exist_ok=False)
    result = evidence / 'blocked-result.json'
    result.parent.mkdir(parents=True, exist_ok=True)
    lab_root = root / 'tmp/rollout-voices'
    lab_root.mkdir(parents=True, exist_ok=True)
    lab = Path(tempfile.mkdtemp(prefix=model + '-', dir=lab_root))
    before, after = {}, {}
    try:
        for relative in weights['files']:
            original = source / relative
            expected = weights['sha256'][relative]
            actual = digest(original)
            if actual != expected or original.stat().st_size != weights['bytes'][relative]:
                raise ValueError(f'{model}/{relative}: source pin mismatch')
            before[relative] = actual
            clone = lab / relative
            clone.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['/bin/cp', '-cL', str(original), str(clone)], check=True, timeout=60)
            if digest(clone) != actual:
                raise ValueError('Private clone hash differs')
        coverage = inventory(model, lab, weights['files'])
        for relative in weights['files']:
            after[relative] = digest(source / relative)
        if before != after:
            raise ValueError('Source changed during preflight')
        receipt = dict(model=model, revision=weights['revision'], repository=weights['repo'], source_files=len(before),
                       before_sha256=before, after_sha256=after, unchanged=True, private_clone_verified=True)
        (evidence / 'source-copy-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        write_inventory(evidence / 'coverage.json', coverage)
        document = dict(schema=1, model=model, revision=weights['revision'], run=run_id, classification='blocked_preflight_not_measurement',
                        rows=blocked_rows(model), coverage='gate/evidence/intake/' + run_id + '/coverage.json',
                        network_bytes=0, model_executions=0, quality_bindings_written=0,
                        note='Preparation only. No format/quantization verdict, speed, physical peak, table stats, or independent reference qualification is asserted.')
        result.write_text(json.dumps(document, indent=2) + '\n')
    finally:
        shutil.rmtree(lab)
    assert not lab.exists()
    (evidence / 'receipt.json').write_text(json.dumps(dict(run=run_id, results=str(result.relative_to(folder)), lab_clones_removed=True, model_processes_started=0), indent=2) + '\n')
    print(f'{model}: BLOCKED all 3 tiers; {len(before)} source pins unchanged; {len(coverage["tensors"])} tensors inventoried; {result.relative_to(root)}', flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--cache', type=Path, default=Path.home() / '.cache/huggingface/hub')
    parser.add_argument('--model', choices=MODELS, action='append')
    args = parser.parse_args()
    for selected in args.model or MODELS:
        run(args.root.resolve(), selected, args.cache)
