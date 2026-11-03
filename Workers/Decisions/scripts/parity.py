# SPDX-License-Identifier: MIT
"""Parity and smoke runs of arsenal-worker-decisions against Verdict's own gates (read from Projects/Verdict, never run).

  convert                     derive Von fp16 (from fp32) and int8/int4 views for all five models   (CPU: mlock work)
  laya MODEL DTYPE [--exact]  Verdict's Laya fixture gate: 202 items, each judged alone, wire answers must match the
                              Python oracle exactly (lab/tests/test_laya_regression.py); plus System One decide per item
  von MODEL                   Verdict's Von SDK gate (lab/tests/test_von_helper.py): fp32 Δp ≤ 1e-4 in three modes;
                              fp16 same argmax/side and |Δp| ≤ 0.1; int8/int4 valid answers, max |Δp| reported
  suite                       every model × offered dtype answers Tests/Fixtures/Bench/kinds/decisions/fixtures-v1.json (typed accuracy,
                              latency, peak RAM, load time)                                         (GPU: mlock gpulock)
Writes evidence JSON to models/<id>/gate/. Exit 1 when a gate fails.
"""
import gzip
import json
import statistics
import sys
import time
from decimal import Decimal
from pathlib import Path

from wire import MODELS, WEIGHTS, Worker

VERDICT = Path(__file__).resolve().parents[3].parent / 'Verdict'
FIXTURES = VERDICT / 'lab/fixtures'
REPO = MODELS.parent
SUITE = REPO / 'Tests/Fixtures/Bench/kinds/decisions/fixtures-v1.json'
LAYA = ['laya-english', 'laya-multilingual', 'laya-typed-decisions']
VON = ['von-1.2', 'von-1.1']
DTYPES = ['fp16', 'int8', 'int4']
BITS = {'fp16': 0, 'int8': 8, 'int4': 4}


def save(model, name, data):
    path = MODELS / model / 'gate' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    return path


def convert():
    w = Worker()
    try:
        for model in VON:
            if not (WEIGHTS / model / 'fp16' / 'manifest.json').exists():
                r = w.call('convert', src_dir=str(WEIGHTS / model / 'fp32'), dst_dir=str(WEIGHTS / model / 'fp16'), dtype='fp16',
                           group_size=64, recipe='cast:von-fp16-1', config={'id': model})
                assert r['ok'], r
                print('converted', model, 'fp16', r['convert_s'], 's', sum(f['bytes'] for f in r['files']), 'bytes', flush=True)
        for model in LAYA + VON:
            for dtype in ('int8', 'int4'):
                if (WEIGHTS / model / dtype / 'arsenal-derived.json').exists():
                    continue
                r = w.call('convert', src_dir=str(WEIGHTS / model / 'fp16'), dst_dir=str(WEIGHTS / model / dtype), dtype=dtype,
                           group_size=64, recipe='mlx-affine-g64:verdict-at-load', config={'id': model})
                assert r['ok'], r
                print('derived', model, dtype, r['convert_s'], 's', len(r['files']), 'files', flush=True)
    finally:
        w.close()


def system_one_view(wire_answer, question):
    """What Verdict's SystemOne.answer makes of a /judge answer (for consistency checks)."""
    kind = question['type']
    if kind == 'noul':
        return {'noul': wire_answer['noul'], 'type': 'noul'}
    out = {'confidence': wire_answer['confidence'], 'probabilities': wire_answer['probabilities'], 'type': kind}
    if kind == 'choice':
        out['choice'] = wire_answer['choice']
    else:
        out['score'] = wire_answer['score']
        out['legend'] = {str(i): level for i, level in enumerate(question['criteria'])}
    return out


def laya(model, dtype, fast=True):
    fixture = json.load(gzip.open(FIXTURES / f'{model}-{BITS[dtype]}bit.json.gz'))
    questions = fixture['questions']
    w = Worker()
    try:
        load = w.load(model, dtype, fast=fast)
        assert load['ok'], load
        mismatches, decide_mismatches, changed, max_delta, latencies, goldens = [], [], [], 0.0, [], []
        for entry in fixture['items']:
            body = json.dumps({'items': [entry['input']], 'questions': questions, 'model': model}, ensure_ascii=False)
            start = time.perf_counter()
            r = w.call('judge', body=body)
            latencies.append(time.perf_counter() - start)
            assert r['ok'], r
            got = {k: v for k, v in r['results'][0].items() if k != 'ms'}
            expected = entry['wire']
            if got != expected:
                mismatches.append({'id': entry['id'], 'got': got, 'expected': expected})
            for key, answer in expected.get('answers', {}).items():
                observed = got.get('answers', {}).get(key, {})
                for label, value in answer.get('probabilities', {}).items():
                    max_delta = max(max_delta, abs(value - observed.get('probabilities', {}).get(label, float('nan'))))
                if 'noul' in answer:
                    max_delta = max(max_delta, abs(answer['noul'] - observed.get('noul', float('nan'))))
                if chosen(answer) != chosen(observed):
                    changed.append(f"{entry['id']}/{key}")
            # System One: the same item through decide must be Verdict's System One view of this run's judge answers
            # (identical rows, so identical numbers), and of the oracle where the run itself matches it.
            request = {'model': model, 'state': entry['input'], 'questions': questions}
            d = w.call('decide', body=json.dumps(request, ensure_ascii=False))
            if 'answers' in got:
                want = {qid: system_one_view(a, questions[qid]) for qid, a in got['answers'].items()}
                if not d['ok'] or d['answers'] != want or d['model'] != model:
                    decide_mismatches.append({'id': entry['id'], 'got': d, 'expected': want})
                elif len(goldens) < 12 and got == expected:
                    goldens.append({'request': request, 'status': 200,
                                    'response': {'model': d['model'], 'answers': d['answers'], 'usage': d['usage']}})
            elif d['ok'] or d['error']['status'] != 422:
                decide_mismatches.append({'id': entry['id'], 'got': d, 'expected': '422 over context'})
            elif len(goldens) < 14:
                goldens.append({'request': request, 'status': 422, 'response': d['error']['body']})
        status = w.call('status')
    finally:
        code = w.close()
    result = {'gate': 'verdict laya fixture regression (exact wire match, items judged alone)', 'model': model, 'dtype': dtype,
              'fast': fast, 'fixture': f'Projects/Verdict/lab/fixtures/{model}-{BITS[dtype]}bit.json.gz', 'snapshot': fixture['meta']['snapshot'],
              'items': len(fixture['items']), 'mismatches': len(mismatches), 'changed_answers': len(changed), 'max_delta': max_delta,
              'decide_mismatches': len(decide_mismatches), 'load_s': load['load_s'], 'engine': load.get('engine'),
              'engine_reason': load.get('engine_reason'), 'optimizations': load.get('optimizations'),
              'median_item_ms': round(statistics.median(latencies) * 1000, 2), 'memory_after': status['memory'],
              'exit_code': code, 'examples': (mismatches + decide_mismatches)[:3]}
    if dtype == 'fp16' and not fast:
        golden = REPO / 'Workers/Decisions/Tests/DecisionsTests/Fixtures/systemone' / f'{model}-fp16.json'
        golden.write_text(json.dumps({'source': result['fixture'], 'model': model, 'dtype': dtype, 'cases': goldens},
                                     indent=1, ensure_ascii=False) + '\n')
    # Exact: byte-for-byte the oracle (Verdict's fixture regression). Fast adds the small-M GEMM (not bit-identical);
    # its gate is Verdict's noise ruling (lab/notes/brief-verdict-smallm.md): chosen answer unchanged on >= 99.5 % of
    # items and max |dp| <= 0.01.
    rows = sum(len(e['wire'].get('answers', {})) for e in fixture['items'])
    passed = (not mismatches) if not fast or not mismatches else (len(changed) <= 0.005 * rows and max_delta <= 0.01)
    result['gate_passed'] = passed and not decide_mismatches and code == 0
    save(model, f'parity-{dtype}{"" if fast else "-exact"}.json', result)
    print('LAYA', model, dtype, 'fast' if fast else 'exact', 'PASS' if result['gate_passed'] else 'FAIL', 'items', result['items'], 'mismatches', result['mismatches'], 'changed', len(changed),
          'decide_mismatches', result['decide_mismatches'], 'max_delta', max_delta, 'median_ms', result['median_item_ms'],
          'load_s', result['load_s'], 'footprint_mb', status['memory']['footprint_mb'], flush=True)
    return result['gate_passed']


def chosen(answer):
    if 'choice' in answer:
        return answer['choice']
    if 'probabilities' in answer:
        p = answer['probabilities']; return max(p, key=p.get)
    return answer.get('noul', 0) >= 0.5


def von(model):
    version = model.split('-')[1]
    fixture = json.load(gzip.open(FIXTURES / f'von-{version}-sdk.json.gz'))
    questions = fixture['questions']
    items = [e['item'] for e in fixture['items']]
    summary, ok = {}, True

    def judge(w, its, qs):
        r = w.call('judge', body=json.dumps({'items': its, 'questions': qs, 'model': model}, ensure_ascii=False))
        assert r['ok'], r
        return r['results']

    for dtype in ['fp32'] + DTYPES:
        w = Worker()
        violations, max_p = [], Decimal(0)
        try:
            load = w.load(model, dtype)
            assert load['ok'], load
            start = time.perf_counter()

            def compare(actual, expected, context, strict):
                nonlocal max_p
                for field in ('probabilities', 'noul'):
                    if field not in expected:
                        continue
                    wanted = expected[field]
                    pairs = wanted.items() if isinstance(wanted, dict) else [('true', wanted)]
                    for key, value in pairs:
                        got = actual[field][key] if isinstance(wanted, dict) else actual[field]
                        delta = abs(Decimal(str(got)) - Decimal(str(value)))
                        max_p = max(max_p, delta)
                        if strict and delta > Decimal('0.0001'):
                            violations.append(f'{context}/{field}/{key}: dp={delta}')
                if dtype in ('fp32', 'fp16'):
                    if 'choice' in expected and actual.get('choice') != expected['choice']:
                        violations.append(f'{context}: choice argmax differs')
                    if 'score' in expected:
                        p, ref = actual['probabilities'], expected['probabilities']
                        if max(p, key=p.get) != max(ref, key=ref.get):
                            violations.append(f'{context}: score argmax differs')
                    if 'noul' in expected and (actual['noul'] > 0.5) != (expected['noul'] > 0.5):
                        violations.append(f'{context}: noul side differs')
                if not 0 <= actual.get('confidence', -1) <= 1:
                    violations.append(f'{context}: invalid confidence')

            strict = dtype == 'fp32'
            for entry in fixture['items']:
                one = judge(w, [entry['item']], questions)[0]
                for name, expected in entry['item_batched_answers'].items():
                    compare(one['answers'][name], expected, f'{entry["item"]!r}/{name} three-question', strict)
                    alone = judge(w, [entry['item']], {name: questions[name]})[0]
                    compare(alone['answers'][name], entry['single_question_batched_answers'][name], f'{name} one-question', strict)
            multi = judge(w, items, questions)
            for i, (entry, actual) in enumerate(zip(fixture['items'], multi)):
                for name, expected in entry['batched_answers'].items():
                    compare(actual['answers'][name], expected, f'item={i}/{name} five-item', strict)
            seconds = time.perf_counter() - start
            if dtype == 'fp16' and max_p > Decimal('0.1'):
                violations.append(f'fp16 max |dp| {max_p} > 0.1')
            if dtype in ('int8', 'int4'):
                violations = [v for v in violations if 'confidence' in v]   # lossy by design: Verdict checks validity only
            status = w.call('status')
            summary[dtype] = {'violations': len(violations), 'max_dp': float(max_p), 'load_s': load['load_s'], 'engine': load.get('engine'),
                              'seconds_all_calls': round(seconds, 3), 'memory_after': status['memory'], 'examples': violations[:4]}
        finally:
            summary.setdefault(dtype, {})['exit_code'] = w.close()
        ok = ok and not violations and summary[dtype]['exit_code'] == 0
        print('VON', model, dtype, 'violations', len(violations), 'max_dp', float(max_p), 'load_s', load['load_s'],
              'footprint_mb', status['memory']['footprint_mb'], flush=True)
    save(model, 'parity-von-sdk.json', {'gate': 'verdict von SDK gate: fp32 dp<=1e-4 (three modes); fp16 argmax/side + |dp|<=0.1; '
                                                'int8/int4 valid answers (lossy by design)',
                                        'fixture': f'Projects/Verdict/lab/fixtures/von-{version}-sdk.json.gz', 'model': model, 'results': summary})
    return ok


def suite():
    cases = json.loads(SUITE.read_text())
    ok = True
    for model in LAYA + VON:
        rows = {}
        for dtype in DTYPES:
            w = Worker()
            try:
                load = w.load(model, dtype)
                assert load['ok'], load
                correct, total, latencies = 0, 0, []
                answers = {}
                for case in cases:
                    start = time.perf_counter()
                    r = w.call(case['op'], **case['params'])
                    latencies.append(time.perf_counter() - start)
                    assert r['ok'], r
                    for qid, label in case['labels'].items():
                        a = r['answers'][qid]
                        got = a['noul'] >= 0.5 if a['type'] == 'noul' else a['choice'] if a['type'] == 'choice' else max(a['probabilities'], key=a['probabilities'].get)
                        want = str(label) if a['type'] == 'score' else label
                        correct += got == want; total += 1
                        answers[case['id']] = a
                status = w.call('status')
                rows[dtype] = {'typed_accuracy': f'{correct}/{total}', 'median_case_ms': round(statistics.median(latencies) * 1000, 2),
                               'load_s': load['load_s'], 'wall_load_s': load['wall_s'], 'engine': load.get('engine'),
                               'memory_after': status['memory'], 'answers': answers}
            finally:
                rows.setdefault(dtype, {})['exit_code'] = w.close()
            ok = ok and rows[dtype]['exit_code'] == 0
            print('SUITE', model, dtype, rows[dtype].get('typed_accuracy'), 'median_ms', rows[dtype].get('median_case_ms'), 'load_s',
                  rows[dtype].get('load_s'), 'peak_mb', rows[dtype].get('memory_after', {}).get('mlx_peak_mb'),
                  'footprint_mb', rows[dtype].get('memory_after', {}).get('footprint_mb'), flush=True)
        save(model, 'smoke-suite-v1.json', {'suite': 'Tests/Fixtures/Bench/kinds/decisions/fixtures-v1.json (functional smoke, not a benchmark result)',
                                            'model': model, 'results': rows})
    return ok


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'convert':
        convert(); passed = True
    elif command == 'laya':
        models = LAYA if sys.argv[2] == 'all' else [sys.argv[2]]
        dtypes = DTYPES if sys.argv[3] == 'all' else [sys.argv[3]]
        passed = all([laya(m, d, fast='--exact' not in sys.argv) for m in models for d in dtypes])
    elif command == 'von':
        passed = all([von(m) for m in (VON if sys.argv[2] == 'all' else [sys.argv[2]])])
    elif command == 'suite':
        passed = suite()
    else:
        sys.exit(__doc__)
    sys.exit(0 if passed else 1)
