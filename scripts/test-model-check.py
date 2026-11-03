#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import importlib.util,json,pathlib,struct,threading,unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
spec=importlib.util.spec_from_file_location('model_check',pathlib.Path(__file__).with_name('model-check.py'));check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)
class RangeTests(unittest.TestCase):
    def test_exact_ranges_and_mapping(self):
        header={'layer.weight':{'dtype':'F16','shape':[2,3],'data_offsets':[0,12]}}
        body=json.dumps(header).encode();checkpoint=struct.pack('<Q',len(body))+body+b'\0'*12;requests=[]
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                a,b=map(int,self.headers['Range'].split('=')[1].split('-'));requests.append((a,b))
                self.send_response(206);self.send_header('Content-Range',f'bytes {a}-{b}/{len(checkpoint)}');self.end_headers();self.wfile.write(checkpoint[a:b+1])
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            result,size=check.header(f'http://127.0.0.1:{server.server_port}/model.safetensors')
            self.assertEqual(requests,[(0,7),(8,7+len(body))]);self.assertEqual(size,len(body)+8)
            self.assertEqual(check.compare(result,{'layer.weight':{'dtype':'float16','shape':[2,3]}}),[])
            self.assertTrue(check.compare(result,{'layer.weight':{'dtype':'float16','shape':[3,2]}}))
        finally:server.shutdown();thread.join();server.server_close()
    def test_refuses_ignored_range(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):self.send_response(200);self.end_headers()
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            with self.assertRaisesRegex(ValueError,'refusing to read'):check.header(f'http://127.0.0.1:{server.server_port}/weights')
        finally:server.shutdown();thread.join();server.server_close()
    def test_model_owned_mapping(self):
        actual={'backbone.conv.weight':{'dtype':'BF16','shape':[3,2,4,5]},'counter':{'dtype':'I64','shape':[]}}
        expected={'encoder.conv.weight':{'dtype':'bfloat16','shape':[3,4,5,2]}}
        mapping={'ignore':['counter'],'tensors':{'backbone.conv.weight':{'name':'encoder.conv.weight','permutation':[0,2,3,1]}}}
        self.assertEqual(check.compare(actual,expected,mapping=mapping),[])
        mapping['tensors']['backbone.conv.weight']['permutation']=[0,0,1,2]
        with self.assertRaisesRegex(ValueError,'axis permutation'):check.compare(actual,expected,mapping=mapping)
    def test_smoke_resolves_shared_local_and_default_contracts(self):
        import tomllib
        root=pathlib.Path(__file__).resolve().parents[1];model=root/'models/birefnet'
        declaration=tomllib.loads((model/'model.toml').read_text())
        own=check.smoke_contracts(root,model,declaration)
        self.assertEqual(set(own),{'remove_background'})
        self.assertTrue(own['remove_background'][1].is_relative_to(model))
        spec={'name':'transcribe','input':'kinds/audio/ops/transcribe.input.json','output_schema':'kinds/audio/ops/transcribe.output.json','output':'json'}
        shared=check.smoke_contracts(root,model,{'kind':'audio','op':[spec]})
        self.assertEqual(shared['transcribe'][1],root/'Resources/kinds/audio/ops/transcribe.input.json')
        self.assertEqual(shared['transcribe'][2],root/'Resources/kinds/audio/ops/transcribe.output.json')
        self.assertEqual(check.smoke_contracts(root,model,{'kind':'audio'}),{}) # New capabilities are opt-in.
        inherited=check.smoke_contracts(root,model,{'kind':'images'})
        self.assertIn('generate',inherited)
        self.assertEqual(check.smoke_contracts(root,model,{'kind':'images','op':[]}),{})
        bad={**spec,'input':'kinds/../../README.md'}
        with self.assertRaisesRegex(ValueError,'Invalid operation schema'):check.smoke_contracts(root,model,{'kind':'audio','op':[bad]})
    def test_empty_operation_smoke_is_rejected(self):
        import tempfile,argparse
        with tempfile.TemporaryDirectory() as temporary:
            folder=pathlib.Path(temporary);(folder/'gate').mkdir()
            (folder/'model.toml').write_text('kind="audio"\nop=[]\n')
            (folder/'gate/no-weights.json').write_text('{"operations":[]}')
            with self.assertRaisesRegex(ValueError,'positive declared operation'):
                check.smoke(argparse.Namespace(model=str(folder)))
    def test_empty_and_full_precision_inventories_fail(self):
        for inventory in [{},{'empty':{'dtype':'float16','shape':[0]}},{'full':{'dtype':'float32','shape':[1]}}]:
            with self.assertRaises(ValueError):check.validate_inventory(inventory)
        check.validate_inventory({'weight':{'dtype':'float16','shape':[2,64]}})
        with self.assertRaisesRegex(ValueError,'no packed learned weights'):
            check.validate_inventory({'weight':{'dtype':'float16','shape':[2,64]}},'int4')
        check.validate_inventory({'weight':{'dtype':'uint32','shape':[2,8]},'scales':{'dtype':'float16','shape':[2,1]}},'int4')
    def test_planned_fp32_cast_is_explicit_and_shape_checks_remain(self):
        actual={'conv.weight':{'dtype':'F32','shape':[4,3,3,3]}}
        expected={'conv.weight':{'dtype':'float16','shape':[4,3,3,3]}}
        self.assertTrue(check.compare(actual,expected))
        mapping={'casts':{'conv.weight':{'from':'F32','to':'F16','at':'Get'}}}
        casts=[]
        self.assertEqual(check.compare(actual,expected,mapping=mapping,planned_casts=casts),[])
        self.assertEqual(casts,[{'source':'conv.weight','target':'conv.weight','from':'F32','to':'F16','at':'Get'}])
        expected['conv.weight']['shape']=[3,3,3,4]
        self.assertTrue(check.compare(actual,expected,mapping=mapping))
        actual['conv.weight']['dtype']='BF16'
        with self.assertRaisesRegex(ValueError,'not F32'):check.compare(actual,expected,mapping=mapping)
        with self.assertRaises(ValueError):check.compare({},expected,mapping=mapping)
        with self.assertRaises(ValueError):check.compare(actual,expected,mapping={'casts':{'conv.weight':{'from':'BF16','to':'F16','at':'Load'}}})
    def test_effective_base_tracks_manifest_cast_policy(self):
        self.assertEqual(check.base_precision({'weights':{'dtype':'fp32'},'derive':{'fp16':'cast:rife'}}),'fp16')
        self.assertEqual(check.base_precision({'weights':{'dtype':'fp32'},'derive':{'bf16':'cast:exception'}}),'bf16')
        self.assertEqual(check.base_precision({'weights':{'dtype':'bf16'},'derive':{'fp16':'cast:default'}}),'fp16')
    def test_native_bf16_source_defaults_to_fp16_and_explicit_exception(self):
        declaration={'weights':{'dtype':'bf16'},'runner':[{'worker':'embed'}],'derive':{'int8':'embed-v1','int4':'embed-v1'}}
        self.assertEqual(check.serving_tiers(declaration),['fp16','int8','int4'])
        declaration['precision']={'dtype':'bf16','reason':'Vendor activation overflow','evidence':'log/source.md'}
        self.assertEqual(check.serving_tiers(declaration),['bf16','int8','int4'])
        actual={'weight':{'dtype':'BF16','shape':[2,64]}}
        expected={'weight':{'dtype':'float16','shape':[2,64]}}
        self.assertTrue(check.compare(actual,expected))
        casts=[]
        self.assertEqual(check.compare(actual,expected,mapping={'casts':{'weight':{'from':'BF16','to':'F16','at':'Get'}}},planned_casts=casts),[])
        self.assertEqual(casts[0]['from'],'BF16')
    def test_fp32_source_never_becomes_a_serving_tier(self):
        for dtype,derives in [('fp32',{'fp16':'cast:rife','int8':'rife','int4':'rife'}),('fp32',{'fp16':'cast:moge','int8':'moge','int4':'moge'})]:
            self.assertEqual(check.serving_tiers({'weights':{'dtype':dtype},'derive':derives}),['fp16','int8','int4'])
        with self.assertRaisesRegex(ValueError,'Unsupported'):check.serving_tiers({'weights':{'dtype':'fp32'},'derive':{'fp32':'invalid'}})
    def test_zero_eligible_tiers_are_explicit_and_truthful(self):
        declaration={'derive':{'int8':'rife','int4':'rife'},'quantization':{'eligible_matrices':0,'reason':'Spatial convolutions only.'}}
        self.assertTrue(check.zero_matrix_policy(declaration))
        inventory={'conv.weight':{'dtype':'float16','shape':[4,3,3,3]}}
        for precision in ['fp16','int8','int4']:check.validate_inventory(inventory,precision,zero_eligible=True)
        with self.assertRaisesRegex(ValueError,'no packed'):check.validate_inventory(inventory,'int8')
        for tensor in [{'dtype':'uint32','shape':[4,8]},{'dtype':'float16','shape':[4,64]}]:
            with self.assertRaisesRegex(ValueError,'conflicts'):check.validate_inventory({'weight':tensor},'int4',zero_eligible=True)
        with self.assertRaisesRegex(ValueError,'declared half dtype'):check.validate_inventory({'weight':{'dtype':'bfloat16','shape':[4,3,3,3]}},'int4',zero_eligible=True,half_dtype='fp16')
        declaration['quantization']['reason']=''
        with self.assertRaises(ValueError):check.zero_matrix_policy(declaration)
    def test_birefnet_layout_and_dtype(self):
        actual={'conv_out1.0.weight':{'dtype':'F16','shape':[1,2,3,4]},'bb.relative_position_index':{'dtype':'I64','shape':[1]}}
        expected={'conv_out1.weight':{'dtype':'float16','shape':[1,3,4,2]}}
        self.assertEqual(check.compare(actual,expected,True),[])
        expected['conv_out1.weight']['dtype']='bfloat16';self.assertTrue(check.compare(actual,expected,True))
    def test_parser_rejection_passes_only_on_exactly_the_declared_error(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root=pathlib.Path(temporary);support=root/'support';(support/'inbox').mkdir(parents=True)
            schema=root/'input.json';schema.write_text('{"type":"object"}')
            contracts={'extract':({'name':'extract','output':'json'},schema,None)}
            class FakeWorker:
                def __init__(self,reply):self.reply=reply;self.calls=[]
                def call(self,op,**params):
                    self.calls.append((op,params))
                    if isinstance(self.reply,dict) and 'code' in self.reply:raise check.WorkerError(self.reply)
                    return self.reply
            case={'operation':'extract','input':{'text':'Ada'},'expect_error':'invalid_model_output'}
            import sys,types
            if 'jsonschema' not in sys.modules:
                try:import jsonschema  # noqa: F401  (the conformance venv supplies it)
                except ImportError:sys.modules['jsonschema']=types.SimpleNamespace(validate=lambda value,schema:None)
            rejection=FakeWorker({'code':'invalid_model_output','message':'not JSON'})
            self.assertEqual(check.run_case(rejection,case,contracts,support,'validator'),'parser_rejection')
            for reply in [{'code':'runner','message':'crash'},{'code':'unsupported_op','message':'no'}]:
                with self.assertRaisesRegex(ValueError,'expects invalid_model_output'):check.run_case(FakeWorker(reply),case,contracts,support,'validator')
            with patch.object(check,'validate_output') as validate:
                with self.assertRaisesRegex(ValueError,'returned a result'):check.run_case(FakeWorker({'data':{}}),case,contracts,support,'validator')
                validate.assert_not_called()
                plain={'operation':'extract','input':{'text':'Ada'}}
                self.assertEqual(check.run_case(FakeWorker({'data':{}}),plain,contracts,support,'validator'),'output');validate.assert_called_once()
                with self.assertRaisesRegex(ValueError,'worker failed'):check.run_case(rejection,plain,contracts,support,'validator')
            with self.assertRaisesRegex(ValueError,'Only invalid_model_output'):check.run_case(rejection,{**case,'expect_error':'runner'},contracts,support,'validator')
    def test_parser_rejections_need_positive_maker_fixtures(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            model=pathlib.Path(temporary);(model/'gate').mkdir();support=model/'support';(support/'inbox').mkdir(parents=True)
            contracts={'extract':({'name':'extract','output':'json'},model/'in.json',None)}
            fixture={'operations':[{'operation':'extract','input':{},'expect_error':'invalid_model_output'}]}
            with self.assertRaisesRegex(ValueError,'need a positive parser fixture'):check.parser_fixtures(model,fixture,contracts)
            case={'operation':'extract','input':{'template':{'name':'string'}},'raw':'{"name":"Ada"}','expected':{'data':{'name':'Ada'}},'source':'model card example'}
            (model/'gate/parser-fixtures.json').write_text(json.dumps({'cases':[case]}))
            self.assertEqual(check.parser_fixtures(model,fixture,contracts),[case])
            for bad in [{**case,'source':''},{k:v for k,v in case.items() if k!='raw'},{**case,'expect_error':'invalid_model_output'},{**case,'operation':'chat'}]:
                (model/'gate/parser-fixtures.json').write_text(json.dumps({'cases':[bad]}))
                with self.assertRaises(ValueError):check.parser_fixtures(model,fixture,contracts)
            class FakeWorker:
                def __init__(self,reply):self.reply=reply;self.calls=[]
                def call(self,op,**params):
                    self.calls.append((op,params))
                    if 'code' in self.reply:raise check.WorkerError(self.reply)
                    return self.reply
            with patch.object(check,'validate_output') as validate:
                worker=FakeWorker({'data':{'name':'Ada'}})
                check.run_parser_fixture(worker,case,contracts,support,'validator')
                self.assertEqual(worker.calls,[('parse_output',{'operation':'extract','input':case['input'],'raw':case['raw']})]);validate.assert_called_once()
                with self.assertRaisesRegex(ValueError,'parsed to'):check.run_parser_fixture(FakeWorker({'data':{'name':'Bob'}}),case,contracts,support,'validator')
                negative={**{k:v for k,v in case.items() if k!='expected'},'raw':'not json','expect_error':'invalid_model_output'}
                check.run_parser_fixture(FakeWorker({'code':'invalid_model_output','message':'x'}),negative,contracts,support,'validator')
                with self.assertRaisesRegex(ValueError,'accepted'):check.run_parser_fixture(FakeWorker({'data':{}}),negative,contracts,support,'validator')
if __name__=='__main__':unittest.main()
