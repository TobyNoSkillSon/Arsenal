#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded native-worker checks. Invoke real-model mode only through mlock gpulock."""
import argparse, array, hashlib, json, math, os, pathlib, select, struct, subprocess, sys, time, wave
ROOT = pathlib.Path(__file__).resolve().parents[3]
DEV = pathlib.Path.home() / 'Library/Application Support/Arsenal-dev/weights/voices'
class Worker:
    def __init__(self, binary, timeout=180):
        self.log = open('/tmp/arsenal-voices/worker-stderr.log', 'ab')
        self.p = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=dict(os.environ, ARSENAL_SUPPORT_DIR='/tmp/arsenal-voices/support', ARSENAL_CACHE_DIR='/tmp/arsenal-voices/cache'))
        self.timeout = timeout
        self.hello = self.read()[1]
        assert self.hello['event']=='hello' and self.hello['worker']=='voices'
        self.id=0
    def exact(self,n):
        data=b''; deadline=time.monotonic()+self.timeout
        while len(data)<n:
            ready,_,_=select.select([self.p.stdout],[],[],max(0,deadline-time.monotonic()))
            if not ready: raise TimeoutError('worker response deadline')
            b=os.read(self.p.stdout.fileno(),n-len(data))
            if not b: raise RuntimeError('worker exited before reply')
            data+=b
        return data
    def read(self):
        size=struct.unpack('>I',self.exact(4))[0];assert 0<size<=16*1024*1024
        data=self.exact(size)
        return (data[0],json.loads(data[1:]) if data[0]==1 else data[1:])
    def send(self, obj):
        b=b'\x01'+json.dumps(obj).encode();self.p.stdin.write(struct.pack('>I',len(b))+b);self.p.stdin.flush()
    def request(self,op,**params):
        self.id+=1; ident=self.id;self.send(dict(id=ident,op=op,**params));events=[];pcm=b''
        while True:
            kind,value=self.read()
            if kind==2:pcm+=value[4:];continue
            if value.get('id')==ident and 'ok' in value:return value,events,pcm
            events.append(value)
    def close(self):
        if self.p.poll() is None:
            try:self.request('unload');self.p.wait(timeout=5)
            finally:
                if self.p.poll() is None:self.p.kill();self.p.wait(timeout=5)
        self.log.close()
def check_audio(path,stream):
    with wave.open(path,'rb') as f:
        assert f.getsampwidth()==2 and f.getnchannels()==1
        pcm=f.readframes(f.getnframes());rate=f.getframerate();duration=f.getnframes()/rate
    assert pcm==stream, 'file/stream PCM mismatch'
    samples=array.array('h',pcm)
    rms=math.sqrt(sum(float(x*x) for x in samples)/len(samples))/32768
    assert duration>0.1 and 0.0001<rms<0.8, (duration,rms)
    return {'duration_s':duration,'rms':rms,'pcm_sha256':hashlib.sha256(pcm).hexdigest(),'frames':len(samples),'sample_rate':rate}
def smoke(binary,model,dtypes,modes):
    assert os.environ.get("VELLA_LOCK_HELD") == "1", "real smoke requires mlock gpulock"
    out=pathlib.Path('/tmp/arsenal-voices/evidence')/model;out.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((ROOT/'Workers/Voices/metadata/models.json').read_text())
    spec=next(x for x in catalog['models'] if x['id']==model)
    records=[]
    for dtype in dtypes:
      for fast in modes:
        w=Worker(binary)
        try:
            assets={x['id']:str(DEV/'assets'/x['id']) for x in catalog['assets']} if model=='kokoro-82m' else {}
            load,_,_=w.request('load',model_dir=str(ROOT/'models'/model),weights_dir=str(DEV/model/dtype),dtype=dtype,fast=fast,assets=assets)
            assert load['ok'],load
            params={'text':'The quick brown fox jumps over the lazy dog.','stream':True,'out_dir':str(out), 'seed':24301}
            if spec['voices']:params['voice']= 'af_heart' if model=='kokoro-82m' else 'alba'
            if model in ('qwen3-tts-voicedesign','breeze-tts-2'):params['instructions']='A warm adult narrator with a clear, relaxed conversational voice.'
            if model=='qwen3-tts-base':
                ref=pathlib.Path('/tmp/arsenal-voices/reference.wav')
                assert ref.exists(), 'run Kokoro smoke first and save reference.wav'
                params['reference']={'path':str(ref),'text':params['text']}
            result,events,pcm=w.request('speak',**params)
            assert result['ok'],result
            audio=check_audio(result['path'],pcm)
            assert hashlib.sha256(pathlib.Path(result['path']).read_bytes()).hexdigest()==result['sha256']
            status,_,_=w.request('status');assert status['ok']
            records.append({'model':model,'dtype':dtype,'fast':fast,'load':load,'speak':result,'audio':audio,'status':status})
            if model=='kokoro-82m' and dtype=='fp16' and fast:
                import shutil;shutil.copy(result['path'],'/tmp/arsenal-voices/reference.wav')
            print(f'voices: {model} {dtype} {"fast" if fast else "exact"} audio OK',flush=True)
        except Exception as e:
            records.append({'model':model,'dtype':dtype,'fast':fast,'error':str(e)})
            print(f'voices: {model} {dtype} {"fast" if fast else "exact"} FAILED {e}',flush=True)
        finally:
            w.close();assert w.p.returncode==0 or records[-1].get('error')
            (out/'smoke.json').write_text(json.dumps(records,indent=2)+'\n')
    return 1 if any('error' in r for r in records) else 0
def cancellation(binary):
    assert os.environ.get("VELLA_LOCK_HELD") == "1", "real cancellation proof requires mlock gpulock"
    w=Worker(binary, timeout=30)
    try:
        assets={x.name:str(x) for x in (DEV/'assets').iterdir() if x.is_dir()}
        reply,_,_=w.request('load',model_dir=str(ROOT/'models/kokoro-82m'),weights_dir=str(DEV/'kokoro-82m/fp16'),dtype='fp16',fast=True,assets=assets)
        assert reply['ok'],reply
        w.id+=1;target=w.id
        w.send({'id':target,'op':'speak','text':'The quick brown fox jumps over the lazy dog. '*200,'stream':True,'voice':'af_heart'})
        while True:
            kind,event=w.read()
            if kind==1 and event.get('event')=='audio_begin':break
        w.id+=1;cancel=w.id;w.send({'id':cancel,'op':'cancel','target':target})
        replies={}
        while len(replies)<2:
            kind,event=w.read()
            if kind==1 and 'ok' in event and event.get('id') in (target,cancel):replies[event['id']]=event
        assert replies[cancel]['ok'] and replies[cancel]['cancelled']
        assert replies[target]['error']['code']=='cancelled',replies
        pathlib.Path('/tmp/arsenal-voices/cancel-evidence.json').write_text(json.dumps(replies,indent=2))
    finally:w.close()
    assert w.p.returncode==0
    print('voices: active speech cancellation and clean unload passed')
def cpu(binary):
    w=Worker(binary)
    try:
        reply,_,_=w.request('voices');assert reply['ok'] and len(reply['voices'])==36
        reply,_,_=w.request('status');assert reply['ok'] and not reply['loaded']
        reply,_,_=w.request('speak',text='test',stream=True);assert reply['error']['code']=='not_loaded'
        reply,_,_=w.request('unsupported');assert reply['error']['code']=='unsupported_op'
        reply,_,_=w.request('load',model_dir='/tmp/kokoro-82m',weights_dir='/tmp/missing',dtype='bf16',fast=True);assert not reply['ok']
        reply,_,_=w.request('cancel',target=99999);assert reply['ok'] and not reply['cancelled']
        reply,_,_=w.request('ping');assert reply['ok']
    finally:w.close()
    assert w.p.returncode==0
    for payload in [struct.pack('>I',16*1024*1024+1), struct.pack('>I',1)+b'\x03']:
        log=open('/tmp/arsenal-voices/worker-stderr.log','ab')
        p=subprocess.Popen([str(binary)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env=dict(os.environ,ARSENAL_SUPPORT_DIR='/tmp/arsenal-voices/support',ARSENAL_CACHE_DIR='/tmp/arsenal-voices/cache'))
        p.stdin.write(payload);p.stdin.flush();p.wait(timeout=5);assert p.returncode==2;log.close()
    print('voices: CPU wire, voices, error paths, malformed frames and unload passed')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('binary',type=pathlib.Path);ap.add_argument('--model');ap.add_argument('--dtypes',nargs='+');ap.add_argument('--exact-only',action='store_true');ap.add_argument('--fast-only',action='store_true');ap.add_argument('--cancel-test',action='store_true');a=ap.parse_args()
    if a.cancel_test:cancellation(a.binary);sys.exit(0)
    if a.model and a.model not in ('kokoro-82m','pocket-tts'):sys.exit('voices: heavy model runs require a new explicit authorization')
    if a.model:sys.exit(smoke(a.binary,a.model,a.dtypes,[False] if a.exact_only else [True] if a.fast_only else [True,False]))
    cpu(a.binary)
