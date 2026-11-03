#!/usr/bin/env python3
"""Isolated perception worker checks. No trained weights or downloads. Run through the repo's conformance Python."""
import argparse, base64, importlib.util, json, pathlib, struct, tempfile, tomllib

ROOT = pathlib.Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('model_check', ROOT/'scripts/model-check.py')
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)

def send(worker, operation, **params):
    worker.sequence += 1
    payload = json.dumps({'id': worker.sequence, 'op': operation, **params}).encode()
    worker.process.stdin.write(struct.pack('>I', len(payload)+1)+b'\x01'+payload)
    worker.process.stdin.flush()
    return worker.sequence

def invalid(worker, operation, **params):
    try: worker.call(operation, **params)
    except ValueError as error:
        if not isinstance(error.args[0], dict) or error.args[0].get('code') != 'invalid_request': raise
    else: raise AssertionError('Malformed request succeeded')

def run(args, folder):
    fixture = json.loads((folder/'gate/no-weights.json').read_text())
    declaration = tomllib.loads((folder/'model.toml').read_text())
    contracts = CHECK.smoke_contracts(ROOT, folder, declaration)
    base = None
    reports = []
    for tier in CHECK.serving_tiers(declaration):
        with tempfile.TemporaryDirectory(prefix='arsenal-perception-') as temporary:
            support = pathlib.Path(temporary)
            for subdir in ['weights','jobs/output','inbox']: (support/subdir).mkdir(parents=True, exist_ok=True)
            for name, encoded in fixture['files'].items():
                assert pathlib.Path(name).name == name
                (support/'inbox'/name).write_bytes(base64.b64decode(encoded, validate=True))
            (support/'weights/config.json').write_text(json.dumps(fixture['config']))
            def substitute(value):
                if isinstance(value,str): return value.replace('$INBOX',str(support/'inbox')).replace('$OUTPUT',str(support/'jobs/output')).replace('$PERCEPTION_FIXTURES',str(CHECK.PERCEPTION_FIXTURES))
                if isinstance(value,list): return [substitute(v) for v in value]
                if isinstance(value,dict): return {k:substitute(v) for k,v in value.items()}
                return value
            worker = CHECK.Worker([str(args.worker)], support)
            try:
                worker.call('load', model_dir=str(folder), weights_dir=str(support/'weights'), dtype=tier, config={**fixture['config'],'architecture':declaration['architecture'],'random_weights':True}, assets=substitute(fixture.get('assets', {})))
                inventory = worker.call('parameter_inventory')['parameters']
                CHECK.validate_inventory(inventory,tier)
                if tier == 'fp16': base = inventory
                eligible = {k:v for k,v in base.items() if k.endswith('.weight') and len(v['shape']) == 2 and v['shape'][1] % 64 == 0}
                packed = {k:v for k,v in inventory.items() if v['dtype'] == 'uint32'}
                if tier != 'fp16':
                    assert set(eligible) == set(packed), (sorted(set(eligible)-set(packed)), sorted(set(packed)-set(eligible)))
                    bits = 8 if tier == 'int8' else 4
                    for key, original in eligible.items():
                        shape = original['shape']; owner=key[:-7]
                        assert packed[key]['shape'] == [shape[0],shape[1]*bits//32]
                        assert inventory[owner+'.scales']['shape'] == [shape[0],shape[1]//64]
                        assert inventory[owner+'.biases']['shape'] == [shape[0],shape[1]//64]
                checked=set(); calls=0; rejected=0
                for case in fixture['operations']:
                    op=case['operation']; params=substitute(case['input'])
                    transport,input_schema,output_schema=contracts[op]
                    CHECK.validate_output(args.validator,support,op+"-input","json",input_schema,params)
                    if transport['output']=='file': params['out_dir']=str(support/'jobs/output')
                    try: result=worker.call(op,**params)
                    except ValueError as error:
                        if declaration['id'] != 'locateanything-3b': raise
                        detail=error.args[0]
                        assert isinstance(detail,dict) and detail.get('code')=='runner' and ('token limit' in detail.get('message','') or 'malformed grounding output' in detail.get('message','')), detail
                        rejected+=1; checked.add(op); calls+=1
                        continue
                    CHECK.validate_output(args.validator,support,op,transport['output'],output_schema,result)
                    checked.add(op); calls+=1
                assert checked == set(contracts)
                first=fixture['operations'][0]['operation']
                invalid(worker,first)
                sessions=0
                if declaration['id']=='sam3.1':
                    image={'path':str(support/'inbox/fixture.png')}
                    new=worker.call('segment_session',action='create',image=image)['session_id']
                    result=worker.call('segment',session_id=new,points=[{'x':0.5,'y':0.5}],out_dir=str(support/'jobs/output'))
                    CHECK.validate_output(args.validator,support,'segment','file',contracts['segment'][2],result)
                    worker.call('segment_session',action='close',session_id=new)
                    invalid(worker,'segment',session_id=new,points=[{'x':0.5,'y':0.5}],out_dir=str(support/'jobs/output'))
                    video=worker.call('video_session',action='create',video={'path':str(support/'inbox/fixture.mp4')})['session_id']
                    result=worker.call('track_video',session_id=video,max_frames=2,prompt_updates=[{'frame_index':0,'object_id':7,'action':'add','points':[{'x':0.5,'y':0.5}]},{'frame_index':0,'object_id':8,'action':'add','points':[{'x':0.3,'y':0.3}]},{'frame_index':1,'object_id':8,'action':'remove'},{'frame_index':1,'object_id':7,'action':'update','boxes':[{'xmin':0.1,'ymin':0.1,'xmax':0.9,'ymax':0.9}]}],out_dir=str(support/'jobs/output'))
                    CHECK.validate_output(args.validator,support,'track_video','file',contracts['track_video'][2],result)
                    assert len(result['files']) == 3, 'Two objects on frame zero, one after removal on frame one'
                    worker.call('video_session',action='close',session_id=video)
                    invalid(worker,'video_session',action='close',session_id=video)
                    sessions=2
                # Reader receives cancellation while the serial engine owns one real operation.
                if first=='embed': params={'input':['cancel this']*256}
                else:
                    params=substitute(fixture['operations'][0]['input']);params['out_dir']=str(support/'jobs/output')
                    if declaration['id']=='locateanything-3b': params['max_tokens']=4096
                target=send(worker,first,**params); cancel=send(worker,'cancel',target=target)
                terminal={}
                while len(terminal)<2:
                    reply=worker.frame()
                    if reply.get('id') in (target,cancel) and 'ok' in reply:
                        assert reply['id'] not in terminal
                        terminal[reply['id']]=reply
                assert terminal[cancel]['ok']
                if not terminal[target]['ok']:
                    error=terminal[target]['error']
                    assert error['code']=='cancelled'
                worker.call('ping')
                worker.call('unload');assert worker.process.wait(timeout=5)==0
                report={'model':declaration['id'],'precision':tier,'operations':sorted(checked),'calls':calls,'successful_outputs':calls-rejected,'rejected_incomplete_or_malformed':rejected,'parameters':len(inventory),'eligible_matrices':len(eligible),'packed_matrices':len(packed),'malformed':'passed','sessions':sessions,'cancellation':terminal[target].get('error',{}).get('code','completed-before-cancel'),'unload_exit':0,'weights_downloaded':0}
                print(json.dumps(report),flush=True);reports.append(report)
            except BaseException:
                print((support/'worker.stderr').read_text()[-6000:],flush=True)
                raise
            finally: worker.close()
    return reports

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=pathlib.Path,required=True)
    parser.add_argument('--validator',type=pathlib.Path,default=ROOT/'.build/debug/Arsenal')
    parser.add_argument('--output',type=pathlib.Path)
    parser.add_argument('models',nargs='+',type=pathlib.Path)
    args=parser.parse_args();args.worker=args.worker.resolve();args.validator=args.validator.resolve()
    reports=[]
    for folder in args.models: reports.extend(run(args,folder.resolve()))
    if args.output: args.output.write_text(json.dumps(reports,indent=2)+'\n')
if __name__=='__main__':main()
