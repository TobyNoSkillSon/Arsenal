# SPDX-License-Identifier: MIT
"""Bounded Layer 3. Invoke one allowed model per scripts/mlock gpulock call; never downloads weights."""
import argparse,array,io,json,math,os,pathlib,signal,subprocess,time,traceback,wave
import httpx
from openai import OpenAI
from elevenlabs import ElevenLabs
ROOT=pathlib.Path(__file__).resolve().parents[2]
ALLOWED={'kokoro-82m','pocket-tts','qwen3-tts-voicedesign','qwen3-tts-base','breeze-tts-2','stable-audio-3-medium','stable-audio-3-small-sfx','re-use','laya-english','laya-multilingual','laya-typed-decisions','von-1.2','von-1.1'}

def main():
    def expired(_signum, _frame): raise TimeoutError("Layer 3 deadline exceeded; terminating the private host.")
    signal.signal(signal.SIGALRM,expired);signal.alarm(170)
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=sorted(ALLOWED));parser.add_argument('--helpers',default='/tmp/arsenal-mb-v2-api/helpers');parser.add_argument('--dtype');parser.add_argument('--reference',type=pathlib.Path);args=parser.parse_args()
    support=pathlib.Path(os.environ['ARSENAL_SUPPORT_DIR']);cache=pathlib.Path(os.environ['ARSENAL_CACHE_DIR']);cache.mkdir(parents=True,exist_ok=True)
    logfile=(cache/(args.model+'-server.log')).open('wb')
    process=subprocess.Popen([pathlib.Path(os.environ.get('ARSENAL_SWIFT_SCRATCH',str(ROOT/'.build')))/'debug/arsenal-testserver','--real',ROOT,args.helpers],env={**os.environ,'ARSENAL_API_PORT':'0'},stdout=logfile,stderr=logfile)
    record={'model':args.model,'service':'RealModelsService','downloaded':False,'classification':'functional_API_conformance_not_benchmark'}
    try:
        base=None
        for _ in range(200):
            if process.poll() is not None: raise RuntimeError('Real testserver exited; see model server log')
            try:
                status=json.loads((support/'status.json').read_text())
                if status['pid']==process.pid and status['port']:base=f"http://127.0.0.1:{status['port']}";break
            except (FileNotFoundError,ValueError,KeyError):pass
            time.sleep(.025)
        assert base
        with httpx.Client(timeout=100) as raw:
            add=raw.post(base+f'/arsenal/v1/models/{args.model}/add',json={});assert add.status_code==200,add.text
            catalog=raw.get(base+'/arsenal/v1/catalog').json();entry=next(x for x in catalog if x['row']['id']==args.model)
            dtype=args.dtype or entry['row']['selectedPrecision']
            fast=args.model not in {'laya-english','laya-multilingual','laya-typed-decisions','von-1.2','von-1.1'} and entry['row'].get('fastAvailable',False)
            load=raw.post(base+'/arsenal/v1/load',json={'id':args.model,'dtype':dtype,'fast':fast});assert load.status_code==200,load.text
            record['load_status']=load.status_code;record['dtype']=dtype;record['fast']=fast;record['helper_binary']=str(pathlib.Path(args.helpers)/('arsenal-worker-decisions' if args.model in {'laya-english','laya-multilingual','laya-typed-decisions','von-1.2','von-1.1'} else 'arsenal-worker-native'))
            if args.model in {'kokoro-82m','pocket-tts','qwen3-tts-voicedesign','qwen3-tts-base','breeze-tts-2'}:
                voice='af_heart' if args.model=='kokoro-82m' else 'alba' if args.model=='pocket-tts' else 'default'
                extra={}
                if args.model in {'qwen3-tts-voicedesign','breeze-tts-2'}:extra['instructions']='A clear friendly English narrator.'
                if args.model=='qwen3-tts-base':
                    assert args.reference and args.reference.is_file(),'Supply --reference WAV from a local permitted voice'
                    extra['reference']={'path':str(args.reference),'text':'Arsenal API test.'}
                client=OpenAI(api_key='local',base_url=base+'/v1',max_retries=0,timeout=40)
                audio=client.audio.speech.create(model=args.model,input='Arsenal API test.',voice=voice,response_format='wav',extra_body=extra).content
                path=cache/(args.model+'.wav');path.write_bytes(audio)
                with wave.open(io.BytesIO(audio)) as wav:
                    assert wav.getsampwidth()==2 and wav.getnchannels()==1 and wav.getframerate()==24000
                    samples=array.array('h',wav.readframes(wav.getnframes()));duration=wav.getnframes()/24000
                rms=math.sqrt(sum(x*x for x in samples)/len(samples))/32768
                assert duration>.1 and .0001<rms<.8,(duration,rms)
                if args.model in {'qwen3-tts-voicedesign','breeze-tts-2'}:
                    design=raw.post(base+f'/arsenal/v1/models/{args.model}/ops/design_voice',json={'description':'A clear friendly English narrator.','preview_text':'Hello.'});assert design.status_code==200,design.text
                    assert design.json()['voice_id'] and pathlib.Path(design.json()['preview']['path']).is_file()
                    record['design_voice']=True
                elif args.model=='qwen3-tts-base':
                    clone=raw.post(base+f'/arsenal/v1/models/{args.model}/ops/clone_voice',json={'audio':{'path':str(args.reference)},'transcript':'Arsenal API test.','name':'API test'});assert clone.status_code==200,clone.text
                    assert clone.json()['voice_id'];record['clone_voice']=True
                else:
                    with client.audio.speech.with_streaming_response.create(model=args.model,input='Hello.',voice=voice,response_format='pcm') as response:
                        assert len(b''.join(response.iter_bytes()))>4800
                    eleven=ElevenLabs(api_key='local',base_url=base+'/providers/elevenlabs',timeout=40)
                    pcm=b''.join(eleven.text_to_speech.stream(voice,model_id=args.model,text='Hello.',output_format='pcm_24000'));assert len(pcm)>4800
                record.update({'wav':str(path),'dtype':dtype,'duration_s':duration,'rms':rms,'profiles':['OpenAI WAV','OpenAI streaming response PCM','ElevenLabs HTTP stream PCM'] if args.model in {'kokoro-82m','pocket-tts'} else ['OpenAI WAV','Universal design_voice' if args.model!='qwen3-tts-base' else 'Universal clone_voice']})
            elif args.model in {'stable-audio-3-medium','stable-audio-3-small-sfx'}:
                op='generate_music' if args.model=='stable-audio-3-medium' else 'generate_sound'
                eleven=ElevenLabs(api_key='local',base_url=base+'/providers/elevenlabs',timeout=100)
                if op=='generate_music':
                    data=b''.join(eleven.music.compose(prompt='Quiet rain and gentle piano',music_length_ms=500,model_id=args.model,output_format='pcm_44100',request_options={'additional_body_parameters':{'steps':2,'seed':1}}))
                else:
                    data=b''.join(eleven.text_to_sound_effects.convert(text='Rain on a roof',duration_seconds=.5,model_id=args.model,output_format='pcm_44100',request_options={'additional_body_parameters':{'steps':2,'seed':1}}))
                assert len(data)>1000
                record.update({'audio_bytes':len(data),'profiles':['ElevenLabs '+op]})
            elif args.model=='re-use':
                source=io.BytesIO()
                with wave.open(source,'wb') as wav:
                    wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
                    samples=array.array('h',(int(3000*math.sin(i*2*math.pi*170/16000)) for i in range(8000)))
                    wav.writeframes(samples.tobytes())
                eleven=ElevenLabs(api_key='local',base_url=base+'/providers/elevenlabs',timeout=100)
                data=b''.join(eleven.audio_isolation.convert(audio=('speech.wav',source.getvalue(),'audio/wav')))
                assert data.startswith(b'RIFF') and len(data)>1000
                record.update({'audio_bytes':len(data),'profiles':['ElevenLabs audio isolation']})
            else:
                ordered={'model':args.model,'state':{'z':'The checkout crashes.','a':'Safari only.'},'questions':{'z':{'type':'choice','criteria':{'z':'bug','a':'other'}},'a':{'type':'noul','instructions':'Is something broken?'}}}
                request_bytes=json.dumps(ordered,ensure_ascii=False,separators=(',',':')).encode()
                standard=raw.post(base+'/v1/systemone',content=request_bytes,headers={'Content-Type':'application/json'})
                generic=raw.post(base+f'/arsenal/v1/models/{args.model}/ops/decide',content=request_bytes,headers={'Content-Type':'application/json'})
                assert standard.status_code==generic.status_code==200,(standard.text,generic.text)
                assert standard.content==generic.content,(standard.text,generic.text)
                assert '_raw_body' not in generic.json() and 'bits' not in generic.json()
                assert generic.headers['x-verdict-bits']==standard.headers['x-verdict-bits']
                record['universal_systemone_byte_parity']=True
                fixture=ROOT/f'Workers/Decisions/Tests/DecisionsTests/Fixtures/systemone/{args.model}-fp16.json'
                if fixture.exists():
                    cases=json.loads(fixture.read_text())['cases'];passed=0
                    for case in cases:
                        body=json.dumps(case['request'],ensure_ascii=False,separators=(',',':')).encode()
                        response=raw.post(base+'/v1/systemone',content=body,headers={'Content-Type':'application/json'})
                        expected=case.get('response',case.get('body'))
                        expected_status=case.get('status',200)
                        assert response.status_code==expected_status,(case.keys(),response.status_code,response.text[:250])
                        assert response.json()==expected,(response.json(),expected)
                        passed+=1
                    record['golden_cases_passed']=passed
                else:
                    body={'model':args.model,'state':'The checkout page crashes on Safari','questions':{'bug':{'type':'noul','instructions':'Is something broken?'},'component':{'type':'choice','criteria':{'frontend':None,'payments':None}},'severity':{'type':'score','criteria':['Cosmetic','Blocked']}}}
                    response=raw.post(base+'/v1/systemone',json=body);assert response.status_code==200,response.text
                    result=response.json();assert 0<=result['answers']['bug']['noul']<=1;assert result['model']==args.model
                    record['systemone']=result
                batch=raw.post(base+'/v1/judge',json={'model':args.model,'items':['The checkout crashes.','Everything works.'],'questions':{'bug':{'type':'noul','instructions':'Is something broken?'}}})
                assert batch.status_code==200,batch.text
                assert len(batch.json()['results'])==2
                record['batch_items']=2
            doc=raw.get(base+'/arsenal/v1/openapi.json');assert doc.status_code==200,doc.text
            from openapi_spec_validator import validate
            validate(doc.json());record['openapi_validated']=True
            unload=raw.post(base+'/arsenal/v1/unload',json={'id':args.model});assert unload.status_code==200,unload.text
            record['unloaded']=True;record['passed']=True
            print(f'Layer 3 {args.model}: PASS',flush=True)
    except Exception as error:
        record.update({'passed':False,'error':str(error),'trace':traceback.format_exc()});print(f'Layer 3 {args.model}: FAIL {str(error)[:200]}',flush=True)
    finally:
        try:
            with httpx.Client(timeout=3) as raw:
                if base:raw.post(base+'/arsenal/v1/unload',json={'id':args.model})
        except Exception:pass
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        logfile.close();(cache/(args.model+'-result.json')).write_text(json.dumps(record,indent=2,ensure_ascii=False))
    return 0 if record.get('passed') else 1
if __name__=='__main__':raise SystemExit(main())
