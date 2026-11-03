#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""No-weight integration checks. Remote checks read exactly the safetensors header, never tensor payloads."""
import argparse, json, os, pathlib, re, select, struct, subprocess, sys, tempfile, time, urllib.request

MAX_HEADER = 64 * 1024 * 1024
# The one error a random-weight case may expect: the operation's own parser rejected the generated output.
PARSER_REJECTION = 'invalid_model_output'

class WorkerError(ValueError):
    """A worker's terminal failure reply, keeping its structured code."""
    def __init__(self, error):
        self.error = error if isinstance(error, dict) else {'code': 'runner', 'message': str(error)}
        self.code = self.error.get('code')
        super().__init__(self.error)

def remote_range(url, start, end):
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(request, timeout=30) as response:
        # Check before reading: many servers ignore Range and send the entire checkpoint.
        if response.status != 206:
            raise ValueError(f'Range request returned {response.status}; refusing to read any response body')
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if not match or [int(match[1]), int(match[2])] != [start, end]:
            raise ValueError('Server returned an unexpected Content-Range')
        expected = end - start + 1
        if response.headers.get('Content-Encoding', 'identity') != 'identity':
            raise ValueError('Compressed ranges are unsupported')
        body = response.read(expected + 1)
        if len(body) != expected:
            raise ValueError('Range response has a different byte count')
        return body, int(match[3])

def header(url):
    length, total = remote_range(url, 0, 7)
    count, = struct.unpack('<Q', length)
    if not 1 <= count <= MAX_HEADER or count + 8 > total:
        raise ValueError('Invalid safetensors header length')
    data, second_total = remote_range(url, 8, 7 + count)
    if second_total != total: raise ValueError('Checkpoint changed between range requests')
    value = json.loads(data)
    if not isinstance(value, dict): raise ValueError('Header must be an object')
    for key, tensor in value.items():
        if key == '__metadata__': continue
        if not isinstance(tensor, dict) or not isinstance(tensor.get('dtype'), str) or not isinstance(tensor.get('shape'), list):
            raise ValueError(f'Invalid tensor {key}')
        if not all(type(v) is int and v >= 0 for v in tensor['shape']): raise ValueError(f'Invalid shape {key}')
        offsets = tensor.get('data_offsets')
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(v) is int for v in offsets) or not 0 <= offsets[0] <= offsets[1] <= total - count - 8:
            raise ValueError(f'Invalid offsets {key}')
    return {k: v for k, v in value.items() if k != '__metadata__'}, count + 8

DTYPES = {'float16': 'F16', 'bfloat16': 'BF16', 'float32': 'F32', 'float64': 'F64', 'int64': 'I64', 'int32': 'I32', 'uint32': 'U32', 'int8': 'I8', 'uint8': 'U8'}
def compare(actual, expected, birefnet=False, mapping=None, planned_casts=None):
    mapping = mapping or {}
    planned_casts = planned_casts if planned_casts is not None else []
    casts = mapping.get('casts', {})
    if not isinstance(casts, dict) or any(not isinstance(value,dict) or value not in ({'from':'F32','to':'F16','at':'Get'}, {'from':'BF16','to':'F16','at':'Get'}) for value in casts.values()):
        raise ValueError('Only explicit per-tensor F32/BF16 to F16 casts at Get are supported')
    if set(casts) - set(actual): raise ValueError('Planned cast names missing from checkpoint')
    if set(casts) & set(mapping.get('ignore', [])): raise ValueError('Planned casts cannot be ignored')
    mapped = {}
    for name, tensor in actual.items():
        if birefnet and (name.endswith('.relative_position_index') or name.endswith('.num_batches_tracked')): continue
        if name in mapping.get('ignore', []): continue
        shape = tensor['shape']
        rule = mapping.get('tensors', {}).get(name, {})
        if 'permutation' in rule:
            axes = rule['permutation']
            if sorted(axes) != list(range(len(shape))): raise ValueError(f'Invalid axis permutation for {name}')
            shape = [shape[axis] for axis in axes]
        if name in casts:
            if tensor['dtype'] != casts[name]['from']: raise ValueError(f'Planned cast source is not {casts[name]["from"]}: {name}')
            planned_casts.append({'source':name,'target':rule.get('name',name),'from':casts[name]['from'],'to':'F16','at':'Get'})
        source_name = name
        name = rule.get('name', name)
        if birefnet:
            name = name.replace('conv_out1.0.', 'conv_out1.')
            if len(shape) == 4: shape = [shape[0], shape[2], shape[3], shape[1]]
        if name in mapped: raise ValueError(f'Duplicate mapped tensor {name}')
        mapped[name] = {'shape': shape, 'dtype': 'F16' if source_name in casts else tensor['dtype']}
    errors = []
    for name in sorted(set(mapped) | set(expected)):
        if name not in mapped: errors.append(f'missing {name}'); continue
        if name not in expected: errors.append(f'unexpected {name}'); continue
        a, e = mapped[name], expected[name]
        if a['shape'] != e['shape']: errors.append(f'{name}: shape {a["shape"]} != {e["shape"]}')
        if a['dtype'] != DTYPES.get(e['dtype'], e['dtype']): errors.append(f'{name}: dtype {a["dtype"]} != {e["dtype"]}')
    return errors

class Worker:
    def __init__(self, command, support):
        vm = subprocess.check_output(['/usr/bin/vm_stat'], text=True, timeout=5)
        page = int(re.search(r'page size of (\d+) bytes', vm)[1])
        counts = {key: int(value) for key,value in re.findall(r'^([^:]+):\s+(\d+)\.',vm,re.M)}
        headroom = sum(counts.get(key,0) for key in ('Pages free','Pages speculative','Pages inactive')) * page
        limit = min(8 * 1024**3, headroom // 2)
        if limit < 256 * 1024**2: raise ValueError('Insufficient free memory for a random-weight smoke')
        self.deadline = time.monotonic() + 590
        env = {**os.environ, 'ARSENAL_LAB_MAX_BYTES':str(limit), 'ARSENAL_SUPPORT_DIR': str(support), 'ARSENAL_CACHE_DIR': str(support / 'cache'), 'ARSENAL_NATIVE_LAB': '1'}
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=(support/'worker.stderr').open('wb'), env=env)
        self.sequence = 0
        try:
            hello = self.frame()
            if hello.get('event') != 'hello': raise ValueError('Worker did not send hello')
        except BaseException:
            self.close()
            raise
    def exact(self, size, deadline):
        out = bytearray()
        while len(out) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], max(0, remaining))[0]: raise TimeoutError('Worker deadline exceeded')
            chunk = os.read(self.process.stdout.fileno(), size-len(out))
            if not chunk: raise ValueError(f'Worker exited {self.process.poll()}')
            out.extend(chunk)
        return bytes(out)
    def frame(self):
        deadline = min(self.deadline, time.monotonic() + 90)
        size, = struct.unpack('>I', self.exact(4, deadline))
        if not 1 <= size <= 16777216: raise ValueError('Invalid wire frame size')
        frame = self.exact(size, deadline)
        if frame[0] != 1: return {'event': 'binary'}
        return json.loads(frame[1:])
    def call(self, op, **params):
        self.sequence += 1
        data = json.dumps({'id': self.sequence, 'op': op, **params}).encode()
        self.process.stdin.write(struct.pack('>I',len(data)+1) + b'\x01' + data); self.process.stdin.flush()
        while True:
            value = self.frame()
            if value.get('id') == self.sequence and 'ok' in value:
                if not value['ok']: raise WorkerError(value.get('error'))
                return {k:v for k,v in value.items() if k not in ('id','ok','memory','peak_bytes')}
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=5)
        self.process.stdin.close(); self.process.stdout.close()

def smoke_contracts(root, model, declaration):
    """Resolve exactly the operations exposed by ModelManifest, including shared/local kinds."""
    import tomllib
    kinds = model/'kinds' if declaration.get('kind_file') else root/'Resources/kinds'
    explicit = declaration.get('op')
    if explicit is None:
        kind = tomllib.loads((kinds/(declaration['kind']+'.toml')).read_text())
        specs = [spec for spec in kind.get('op', []) if spec.get('default', False)]
        folder = kinds/declaration['kind']
    else:
        specs, folder = explicit, model
    def schema(path):
        base, relative = (kinds, path[6:]) if path.startswith('kinds/') else (folder, path)
        target = (base/relative).resolve()
        if not target.is_relative_to(base.resolve()) or not target.is_file():
            raise ValueError(f'Invalid operation schema path: {path}')
        return target
    return {spec['name']: (spec, schema(spec['input']), schema(spec['output_schema']) if spec.get('output_schema') else None) for spec in specs}

def base_precision(declaration):
    native = any(r.get('worker') in ('language','embed','audio','music','restoration','perception','geometry','qwenimage') for r in declaration.get('runner',[]))
    if native or 'precision' in declaration: return declaration.get('precision',{}).get('dtype','fp16')
    derives = declaration.get('derive',{})
    for half in ('fp16','bf16'):
        if derives.get(half,'').startswith('cast:'): return half
    return 'fp16' if declaration['weights']['dtype'] == 'fp32' else declaration['weights']['dtype']

def serving_tiers(declaration):
    tiers = list(dict.fromkeys([base_precision(declaration), *declaration.get('derive', {}).keys()]))
    if not tiers or any(t not in ('fp16','bf16','int8','int4') for t in tiers): raise ValueError('Unsupported declared smoke precision')
    return tiers

def zero_matrix_policy(declaration):
    policy = declaration.get('quantization')
    if policy is None: return False
    if type(policy.get('eligible_matrices')) is not int or policy['eligible_matrices'] != 0 or not isinstance(policy.get('reason'),str) or not policy['reason'].strip():
        raise ValueError('Zero-matrix declaration requires eligible_matrices = 0 and a reason')
    if not {'int8','int4'} <= set(declaration.get('derive',{})): raise ValueError('Zero-matrix models must declare all tiers')
    return True

def validate_inventory(inventory, precision=None, zero_eligible=False, half_dtype=None):
    if not isinstance(inventory, dict) or not inventory: raise ValueError('Parameter inventory must be nonempty')
    count = 0
    for name,tensor in inventory.items():
        if not isinstance(tensor,dict) or not isinstance(tensor.get('shape'),list) or not all(type(v) is int and v>=0 for v in tensor['shape']): raise ValueError(f'Invalid inventory shape: {name}')
        if tensor.get('dtype') not in ('float16','bfloat16','uint32','F16','BF16','U32'): raise ValueError(f'Learned parameter is not half or packed: {name}')
        elements = 1
        for dim in tensor['shape']: elements *= dim
        count += elements
    if count == 0: raise ValueError('Parameter inventory has zero tensor elements')
    if zero_eligible:
        if half_dtype and any(DTYPES.get(tensor['dtype'],tensor['dtype']) != {'fp16':'F16','bf16':'BF16'}[half_dtype] for tensor in inventory.values()): raise ValueError('Zero-matrix tier must retain the declared half dtype')
        if any(tensor['dtype'] in ('uint32','U32') or (len(tensor['shape']) == 2 and tensor['shape'][1] > 0 and tensor['shape'][1] % 64 == 0) for tensor in inventory.values()):
            raise ValueError('Zero-matrix declaration conflicts with packed or potentially eligible matrix inventory')
    if precision in ('int8','int4') and not zero_eligible and not any(tensor['dtype'] in ('uint32','U32') for tensor in inventory.values()):
        raise ValueError('Quantized smoke tier has no packed learned weights')

def validate_output(validator, support, operation, output, schema, result, job):
    request = support/'validate-output.json'
    request.write_text(json.dumps({'operation':operation,'output':output,'schema':json.loads(schema.read_text()) if schema else None,'result':result,'directory':str(job)}))
    reply=subprocess.run([str(pathlib.Path(validator).absolute()),'runner-check','--validate-output',str(request)],timeout=30,capture_output=True,text=True)
    if reply.returncode: raise ValueError('Production output validation failed: '+(reply.stdout+reply.stderr).strip())

# Lab tokenizer fixtures (pinned copies of upstream files, used only when a no-weights gate names them in `assets`).
PERCEPTION_FIXTURES = pathlib.Path(__file__).resolve().parents[1]/'Workers/Perception/Tests/PerceptionTests/Fixtures'

def substitute(value, support, job):
    if isinstance(value,str): return value.replace('$INBOX', str(support/'inbox')).replace('$OUTPUT', str(job)).replace('$PERCEPTION_FIXTURES', str(PERCEPTION_FIXTURES))
    if isinstance(value,dict): return {k:substitute(v, support, job) for k,v in value.items()}
    if isinstance(value,list): return [substitute(v, support, job) for v in value]
    return value

def new_job(support):
    """One output directory per job, as production gives each request its own (a second job never reuses the first's output)."""
    jobs = support/'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    job = jobs/f'job-{len(list(jobs.iterdir())) + 1}'
    job.mkdir()
    return job

def expected_error(case):
    """A random-weight case passes on a parser rejection only when the fixture names exactly that rejection."""
    expected = case.get('expect_error')
    if expected is not None and expected != PARSER_REJECTION:
        raise ValueError(f'Only {PARSER_REJECTION} may be expected from a random-weight case, not {expected!r}')
    return expected

def run_case(worker, case, contracts, support, validator):
    """One fixture operation: a schema-valid output, or exactly the declared parser rejection."""
    op = case['operation']
    if op not in contracts: raise ValueError(f'Fixture names undeclared operation: {op}')
    spec, input_schema, output_schema = contracts[op]
    import jsonschema
    job = new_job(support)
    params = substitute(case['input'], support, job)
    jsonschema.validate(params, json.loads(input_schema.read_text()))
    if spec['output'] in ('audio','image','file'): params['out_dir'] = str(job)
    expected = expected_error(case)
    try:
        result = worker.call(op, **params)
    except WorkerError as error:
        if expected and error.code == expected: return 'parser_rejection'
        raise ValueError(f'{op}: worker failed with {error.error}' + (f'; the fixture expects {expected}' if expected else ''))
    if expected: raise ValueError(f'{op}: the fixture expects {expected}, but the worker returned a result')
    validate_output(validator, support, op, spec['output'], output_schema, result, job)
    return 'output'

def parser_fixtures(model, fixture, contracts):
    """Maker-documented outputs: required for every operation whose random-weight case expects a parser rejection."""
    path = model/'gate/parser-fixtures.json'
    cases = json.loads(path.read_text()).get('cases', []) if path.exists() else []
    for case in cases:
        if case.get('operation') not in contracts: raise ValueError(f'Parser fixture names undeclared operation: {case.get("operation")}')
        if not isinstance(case.get('raw'), str) or not isinstance(case.get('source'), str) or not case['source'].strip():
            raise ValueError('Parser fixture needs raw model output text and its maker source')
        if ('expected' in case) == ('expect_error' in case): raise ValueError('Parser fixture needs exactly one of expected or expect_error')
        if 'expect_error' in case: expected_error(case)
    rejected = {case['operation'] for case in fixture['operations'] if case.get('expect_error')}
    positive = {case['operation'] for case in cases if 'expected' in case}
    if rejected - positive:
        raise ValueError(f'Operations checked only by a parser rejection need a positive parser fixture: {sorted(rejected - positive)}')
    return cases

def run_parser_fixture(worker, case, contracts, support, validator):
    spec, _, output_schema = contracts[case['operation']]
    expected = expected_error(case)
    job = new_job(support)
    try:
        result = worker.call('parse_output', operation=case['operation'], input=substitute(case.get('input', {}), support, job), raw=case['raw'])
    except WorkerError as error:
        if expected and error.code == expected: return
        raise ValueError(f'Parser fixture for {case["operation"]} failed with {error.error}')
    if expected: raise ValueError(f'Parser fixture for {case["operation"]} expects {expected}, but the parser accepted it')
    if result != substitute(case['expected'], support, job): raise ValueError(f'Parser fixture for {case["operation"]} parsed to {json.dumps(result)[:400]}')
    validate_output(validator, support, case['operation'], spec['output'], output_schema, result, job)

def smoke(args):
    root = pathlib.Path(__file__).resolve().parents[1]
    model = pathlib.Path(args.model).resolve()
    import tomllib
    declaration = tomllib.loads((model/'model.toml').read_text())
    fixture = json.loads((model/'gate/no-weights.json').read_text())
    contracts = smoke_contracts(root, model, declaration)
    if not contracts or not fixture.get('operations'): raise ValueError('Smoke needs positive declared operation and fixture counts')
    for case in fixture['operations']: expected_error(case)
    parsers = parser_fixtures(model, fixture, contracts)
    base = base_precision(declaration)
    tiers = serving_tiers(declaration)
    zero_eligible = zero_matrix_policy(declaration)
    for precision in tiers:
        with tempfile.TemporaryDirectory(prefix='arsenal-no-weights-') as temp:
            support = pathlib.Path(temp); (support/'jobs').mkdir(parents=True); (support/'inbox').mkdir()
            # Fixture files come from the model's checked-in fixture directory, never real user data.
            for name, encoded in fixture.get('files', {}).items():
                import base64
                if pathlib.Path(name).name != name: raise ValueError('Fixture filename must be a basename')
                (support/'inbox'/name).write_bytes(base64.b64decode(encoded, validate=True))
            worker = Worker([str(pathlib.Path(args.worker).absolute())], support)
            try:
                config = {**fixture.get('config', {}), 'random_weights': True}
                if declaration.get('architecture'): config['architecture'] = declaration['architecture']
                # Family dispatch reads model_type from config.json; keep this synthetic checkpoint private.
                weights = support/'weights'; weights.mkdir()
                (weights/'config.json').write_text(json.dumps(fixture.get('config', {})))
                worker.call('load', model_dir=str(model), weights_dir=str(weights), dtype=precision, config=config, assets=substitute(fixture.get('assets', {}), support, support))
                inventory = worker.call('parameter_inventory')['parameters']
                validate_inventory(inventory, precision, zero_eligible, base)
                full = [k for k,v in inventory.items() if v['dtype'] in ('float32','float64','F32','F64')]
                if full: raise ValueError(f'Full-precision learned parameters: {full[:8]}')
                if args.inventory and precision == base: pathlib.Path(args.inventory).write_text(json.dumps(inventory, indent=2)+'\n')
                checked = set(); outcomes = {'output': 0, 'parser_rejection': 0}
                for case in fixture['operations']:
                    checked.add(case['operation'])
                    outcomes[run_case(worker, case, contracts, support, args.validator)] += 1
                # The parser does not depend on the tier: maker examples run once, on the base tier.
                if precision == base:
                    for case in parsers: run_parser_fixture(worker, case, contracts, support, args.validator)
                declared = set(contracts)
                if declared - checked: raise ValueError(f'Operations missing from smoke fixture: {sorted(declared-checked)}')
                worker.call('unload')
                if worker.process.wait(timeout=5) != 0: raise ValueError('Worker unload did not exit cleanly')
                print(json.dumps({'status':'passed','operations':sorted(checked),'outputs':outcomes['output'],'parser_rejections':outcomes['parser_rejection'],
                                  'parser_fixtures':len(parsers) if precision == base else 0,'parameters':len(inventory),'precision':precision,'weights_downloaded':0,'effective_dtype':base if zero_eligible else precision,'eligible_matrices':0 if zero_eligible else None,'compression':'none' if zero_eligible else 'model-policy'}))
            finally: worker.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    h = sub.add_parser('header'); h.add_argument('url',nargs='+');h.add_argument('--mapping');h.add_argument('--parameters',required=True);h.add_argument('--birefnet',action='store_true');h.add_argument('--output')
    s = sub.add_parser('smoke');s.add_argument('--model',required=True);s.add_argument('--worker',required=True);s.add_argument('--inventory');s.add_argument('--validator',default=str(pathlib.Path(__file__).resolve().parents[1]/'.build/debug/Arsenal'))
    args=parser.parse_args()
    if args.action == 'smoke': smoke(args); return
    actual,read={},0
    for url in args.url:
        shard,size=header(url)
        if set(actual).intersection(shard): raise ValueError('Duplicate tensor names across checkpoint shards')
        actual.update(shard);read+=size
        if read>MAX_HEADER: raise ValueError('Combined checkpoint headers exceed 64 MiB')
    expected=json.loads(pathlib.Path(args.parameters).read_text())
    mapping=json.loads(pathlib.Path(args.mapping).read_text()) if args.mapping else None
    planned_casts=[]
    errors=compare(actual,expected,args.birefnet,mapping,planned_casts)
    report={'status':'failed' if errors else 'passed','remote_tensors':len(actual),'module_parameters':len(expected),'bytes_read':read,'errors':errors,'planned_casts':planned_casts}
    if args.output: pathlib.Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));sys.exit(bool(errors))
if __name__=='__main__':
    try: main()
    except Exception as error: print(f'model-check: {error}',file=sys.stderr);sys.exit(1)
