# SPDX-License-Identifier: MIT
"""Official SDK parse/transport conformance. No provider calls, models, GPU or synthetic UI events."""
from __future__ import annotations
import asyncio, base64, hashlib, io, json, os, pathlib, socket, subprocess, sys, time, traceback, wave
from urllib.parse import urlparse
import httpx
from openai import OpenAI, AsyncOpenAI, BadRequestError
from anthropic import Anthropic, AsyncAnthropic
from elevenlabs import ElevenLabs, AsyncElevenLabs
import replicate
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUPPORT = pathlib.Path(os.environ['ARSENAL_SUPPORT_DIR'])
CACHE = pathlib.Path(os.environ['ARSENAL_CACHE_DIR'])
TEXT = 'Hello from Arsenal. Łódź ✨'
TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
ANTHRO_TOOLS = [{'name': 'get_weather', 'input_schema': TOOLS[0]['function']['parameters']}]
QUESTIONS = {'bug': {'type': 'noul'}, 'kind': {'type': 'choice', 'criteria': {'bug': None, 'other': None}}, 'priority': {'type': 'score', 'criteria': ['Low', 'High']}}

def local_only(request):
    assert request.url.host in {'127.0.0.1', 'localhost'}, f'Unexpected network origin: {request.url.host}'

async def async_local_only(request): local_only(request)

class Server:
    def __init__(self): self.process = None; self.base = ''
    def start(self, fixtures=None):
        env = {**os.environ, 'ARSENAL_API_PORT': '0'}
        env.pop('ARSENAL_TESTSERVER_CATALOG',None);env.pop('ARSENAL_DECISION_FIXTURES',None)
        if fixtures: env['ARSENAL_DECISION_FIXTURES']=str(fixtures)
        self.log = open(CACHE / 'server.log', 'ab')
        self.process = subprocess.Popen([pathlib.Path(os.environ.get('ARSENAL_SWIFT_SCRATCH', str(ROOT/'.build')))/'debug/arsenal-testserver'], env=env, stdout=self.log, stderr=self.log)
        status = SUPPORT / 'status.json'
        for _ in range(100):
            if self.process.poll() is not None: raise RuntimeError('Test server exited; see server.log')
            try:
                obj = json.loads(status.read_text())
                if obj['pid'] == self.process.pid and obj['port']:
                    self.base = f"http://127.0.0.1:{obj['port']}"; return self.base
            except (FileNotFoundError, ValueError, KeyError): pass
            time.sleep(.05)
        raise RuntimeError('Test server failed to become ready in 5 s')
    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=5)
        if hasattr(self, 'log'): self.log.close()

base = ''
client = None

def openai_sync():
    with OpenAI(base_url=base+'/v1', api_key='local', max_retries=0, http_client=httpx.Client(event_hooks={'request': [local_only]}, timeout=10)) as c:
        assert len(c.models.list().data) == 14
        assert c.models.retrieve('stub-chat').id == 'stub-chat'
        result = c.chat.completions.create(model='stub-chat', messages=[{'role':'user','content':'hello'}])
        assert result.choices[0].message.content == TEXT and result.usage.total_tokens == 14
        chunks = list(c.chat.completions.create(model='stub-chat', messages=[{'role':'user','content':'hello'}], stream=True, stream_options={'include_usage':True}))
        assert ''.join(x.choices[0].delta.content or '' for x in chunks if x.choices) == TEXT
        assert chunks[-1].choices == [] and chunks[-1].usage.total_tokens == 14
        assert len({x.id for x in chunks}) == 1
        tool = c.chat.completions.create(model='stub-chat', messages=[{'role':'user','content':'weather'}], tools=TOOLS)
        call = tool.choices[0].message.tool_calls[0]
        assert json.loads(call.function.arguments)['city'] == 'Łódź'
        follow = c.chat.completions.create(model='stub-chat', messages=[{'role':'user','content':'weather'}, tool.choices[0].message.model_dump(exclude_none=True), {'role':'tool','tool_call_id':call.id,'content':'Sunny'}], tools=TOOLS)
        assert follow.choices[0].message.content == TEXT
        streamed = list(c.chat.completions.create(model='stub-chat', messages=[{'role':'user','content':'weather'}], tools=TOOLS, stream=True))
        assert streamed[-1].choices[0].finish_reason == 'tool_calls'
        assert any(x.choices and x.choices[0].delta.tool_calls for x in streamed)
        audio = c.audio.speech.create(model='stub-voices', input='hello', voice='af_heart', response_format='wav').content
        with wave.open(io.BytesIO(audio)) as wav: assert wav.getframerate()==24000 and wav.getnframes()==4800
        with c.audio.speech.with_streaming_response.create(model='stub-voices', input='hello', voice='af_heart', response_format='pcm') as response:
            assert len(b''.join(response.iter_bytes(chunk_size=137))) == 9600
        try: c.audio.speech.create(model='stub-voices', input='hello', voice='af_heart')
        except BadRequestError: pass
        else: raise AssertionError('Unsupported default MP3 was not rejected')
        for fmt in ['b64_json','url']:
            image = c.images.generate(model='stub-images', prompt='cat', size='256x256', response_format=fmt)
            data = base64.b64decode(image.data[0].b64_json) if fmt=='b64_json' else client.get(image.data[0].url).content
            with Image.open(io.BytesIO(data)) as png: assert png.format=='PNG'; png.load()

async def openai_async():
    async with AsyncOpenAI(base_url=base+'/v1', api_key='local', max_retries=0, http_client=httpx.AsyncClient(event_hooks={'request':[async_local_only]},timeout=10)) as c:
        assert (await c.models.retrieve('stub-chat')).id == 'stub-chat'
        assert len((await c.models.list()).data)==14
        assert (await c.chat.completions.create(model='stub-chat',messages=[{'role':'user','content':'hello'}])).choices[0].message.content==TEXT
        stream = await c.chat.completions.create(model='stub-chat',messages=[{'role':'user','content':'hello'}],stream=True)
        chunks = [chunk async for chunk in stream]
        assert ''.join(x.choices[0].delta.content or '' for x in chunks)==TEXT
        async with c.audio.speech.with_streaming_response.create(model='stub-voices',input='hi',voice='af_heart',response_format='wav') as response:
            data = b''.join([chunk async for chunk in response.iter_bytes(chunk_size=73)])
            with wave.open(io.BytesIO(data)) as wav: assert wav.getnframes()==4800
        image = await c.images.generate(model='stub-images',prompt='cat',size='256x256')
        assert base64.b64decode(image.data[0].b64_json).startswith(b'\x89PNG')


def anthropic_sync():
    with Anthropic(base_url=base+'/providers/anthropic',api_key='local',max_retries=0,http_client=httpx.Client(event_hooks={'request':[local_only]},timeout=10)) as c:
        assert c.models.list().data[0].id=='stub-chat'
        assert c.models.retrieve('stub-chat').id=='stub-chat'
        args={'model':'stub-chat','max_tokens':32,'messages':[{'role':'user','content':'hello'}]}
        result=c.messages.create(**args)
        assert result.content[0].text==TEXT and result.stop_reason=='end_turn'
        with c.messages.stream(**args) as stream:
            assert ''.join(stream.text_stream)==TEXT
            assert stream.get_final_message().usage.output_tokens==6
        assert c.messages.count_tokens(model='stub-chat',messages=args['messages']).input_tokens==8
        with c.messages.stream(**{**args,'tools':ANTHRO_TOOLS}) as stream: result=stream.get_final_message()
        assert result.stop_reason=='tool_use' and result.content[0].input=={'city':'Łódź'}
        follow=c.messages.create(**{**args,'tools':ANTHRO_TOOLS,'messages':args['messages']+[{'role':'assistant','content':[x.model_dump() for x in result.content]},{'role':'user','content':[{'type':'tool_result','tool_use_id':result.content[0].id,'content':'Sunny'}]}]})
        assert follow.content[0].text==TEXT

async def anthropic_async():
    async with AsyncAnthropic(base_url=base+'/providers/anthropic',api_key='local',max_retries=0,http_client=httpx.AsyncClient(event_hooks={'request':[async_local_only]},timeout=10)) as c:
        args={'model':'stub-chat','max_tokens':32,'messages':[{'role':'user','content':'hello'}]}
        assert (await c.messages.create(**args)).content[0].text==TEXT
        async with c.messages.stream(**args) as stream:
            assert ''.join([text async for text in stream.text_stream])==TEXT
            assert (await stream.get_final_message()).stop_reason=='end_turn'
        assert (await c.models.list()).data[0].id=='stub-chat'


def elevenlabs_sync():
    c=ElevenLabs(base_url=base+'/providers/elevenlabs',api_key='local',httpx_client=httpx.Client(event_hooks={'request':[local_only]},timeout=10))
    assert c.models.list()[0].model_id=='stub-voices'
    assert c.voices.search().voices[0].voice_id=='af_heart'
    assert c.voices.get('af_heart').name=='Heart'
    for operation in [c.text_to_speech.convert,c.text_to_speech.stream]:
        audio=b''.join(operation('af_heart',model_id='stub-voices',text='hello',output_format='pcm_24000'))
        assert len(audio)==9600

async def elevenlabs_async():
    c=AsyncElevenLabs(base_url=base+'/providers/elevenlabs',api_key='local',httpx_client=httpx.AsyncClient(event_hooks={'request':[async_local_only]},timeout=10))
    assert (await c.models.list())[0].model_id=='stub-voices'
    assert (await c.voices.search()).voices[0].voice_id=='af_heart'
    for operation in [c.text_to_speech.convert,c.text_to_speech.stream]:
        audio=b''.join([chunk async for chunk in operation('af_heart',model_id='stub-voices',text='hello',output_format='pcm_24000')])
        assert len(audio)==9600


def replicate_sync():
    c=replicate.Client(api_token='local',base_url=base+'/providers/replicate',timeout=10)
    c._client.event_hooks={'request':[local_only]}
    prediction=c.predictions.create(version='stub-images',input={'prompt':'cat','width':256,'height':256},wait=2)
    prediction.wait()
    assert prediction.status=='succeeded'
    output=c.run('local/stub-images:'+hashlib.sha256(b'stub-images').hexdigest(),input={'prompt':'cat','width':256,'height':256})
    assert len(output)==1 and client.get(str(output[0])).content.startswith(b'\x89PNG')
    inline=c.run('local/stub-images',input={'prompt':'cat','output_type':'data_uri'})
    assert inline[0].read().startswith(b'\x89PNG')
    official=c.run('local/stub-images',input={'prompt':'cat','width':256,'height':256})
    assert client.get(str(official[0])).content.startswith(b'\x89PNG')
    prediction=c.predictions.create(version='stub-images',input={'prompt':'slow cat'},wait=False)
    prediction.cancel(); prediction.reload()
    assert prediction.status=='canceled'

async def replicate_async():
    c=replicate.Client(api_token='local',base_url=base+'/providers/replicate',timeout=10)
    prediction=await c.predictions.async_create(version='stub-images',input={'prompt':'cat'},wait=2)
    await prediction.async_wait()
    assert prediction.status=='succeeded'
    output=await c.async_run('local/stub-images:'+hashlib.sha256(b'stub-images').hexdigest(),input={'prompt':'cat'})
    assert len(output)==1


def systemone():
    for prefix in ['', '/providers/typesafe']:
        response=client.post(base+prefix+'/v1/systemone',json={'model':'stub-decisions','state':{'title':'Broken'},'questions':QUESTIONS})
        assert response.status_code==200, response.text
        assert response.headers['x-typesafe-request-id'].startswith('req_') and response.headers['x-verdict-bits']=='16'
        body=response.json()
        assert body['answers']['bug']=={'type':'noul','noul':.75}
        assert body['answers']['kind']['choice']=='bug'
        assert body['answers']['priority']['legend']=={'0':'Low','1':'High'}
        assert body['usage']=={'input_tokens':218,'output_tokens':0}
        assert client.post(base+prefix+'/v1/systemone',json={'model':'stub-decisions','state':'s','questions':QUESTIONS,'bits':4}).status_code==409
    batch=client.post(base+'/v1/judge',json={'model':'stub-decisions','items':['a',{'image':'/tmp/no'}],'questions':QUESTIONS}).json()
    assert batch['results'][0]['answers']['bug']=={'noul':.75,'confidence':.75}
    assert batch['results'][1]['ok'] is False
    assert client.get(base+'/providers/typesafe/v1/models').json()['models'][0]['name']=='stub-decisions'


def raw_request(message: bytes) -> bytes:
    address=urlparse(base)
    with socket.create_connection((address.hostname,address.port),timeout=5) as sock:
        sock.sendall(message); chunks=[]
        while True:
            data=sock.recv(8192)
            if not data: break
            chunks.append(data)
        return b''.join(chunks)


def invalid_numeric_and_chat_controls():
    before_jobs = {path.name for path in (SUPPORT/'jobs').iterdir()}
    invalid = [1e100, -1e100, 256.5, '256', True, None, [], {}]
    for path in ['/providers/replicate/v1/predictions', '/providers/replicate/v1/models/local/stub-images/predictions']:
        for key in ['width', 'height', 'num_outputs', 'steps', 'seed']:
            for value in invalid:
                reply = client.post(base+path, json={'version':'stub-images','input':{'prompt':'test',key:value}})
                assert reply.status_code == 422, (path, key, value, reply.text)
    for key in ['n', 'steps', 'seed']:
        for value in invalid:
            reply = client.post(base+'/v1/images/generations', json={'model':'stub-images','prompt':'test',key:value})
            assert reply.status_code == 422, (key,value,reply.text)
    for controls in [{'tool_choice':'required'}, {'tool_choice':{'type':'function','function':{'name':'test'}}}, {'parallel_tool_calls':True}, {'parallel_tool_calls':False}]:
        reply = client.post(base+'/v1/chat/completions', json={'model':'stub-chat','messages':[{'role':'user','content':'test'}],**controls})
        assert reply.status_code == 400, reply.text
    for controls in [{'top_k':10}, {'tool_choice':{'type':'any'}}, {'tool_choice':{'type':'auto','disable_parallel_tool_use':True}}]:
        reply = client.post(base+'/providers/anthropic/v1/messages', json={'model':'stub-chat','max_tokens':20,'messages':[{'role':'user','content':'test'}],**controls})
        assert reply.status_code == 400, reply.text
    # A huge-number request must leave the host alive and serving valid requests.
    assert client.get(base+'/v1/models').status_code == 200
    assert {path.name for path in (SUPPORT/'jobs').iterdir()} == before_jobs, 'Invalid requests allocated output jobs'


def transient_outputs():
    before_jobs = {path.name for path in (SUPPORT/'jobs').iterdir()}
    for _ in range(3):
        reply = client.post(base+'/v1/audio/speech', json={'model':'stub-voices','input':'test','voice':'af_heart','response_format':'wav'})
        assert reply.status_code == 200 and reply.content.startswith(b'RIFF'), reply.text[:100]
        reply = client.post(base+'/providers/elevenlabs/v1/text-to-speech/af_heart?output_format=pcm_24000', json={'model_id':'stub-voices','text':'test'})
        assert reply.status_code == 200 and len(reply.content) == 9600
        reply = client.post(base+'/v1/images/generations', json={'model':'stub-images','prompt':'cat','size':'256x256','response_format':'b64_json'})
        assert reply.status_code == 200, reply.text
    for _ in range(100):
        if {path.name for path in (SUPPORT/'jobs').iterdir()} == before_jobs: break
        time.sleep(.01)
    assert {path.name for path in (SUPPORT/'jobs').iterdir()} == before_jobs, 'Inline responses leaked job directories'


def universal_and_openapi():
    from openapi_spec_validator import validate
    doc=client.get(base+'/arsenal/v1/openapi.json').json()
    validate(doc)
    (CACHE/'openapi.json').write_text(json.dumps(doc,indent=2))
    assert '/arsenal/v1/models/stub-sound/ops/generate_sound' in doc['paths']
    assert '/providers/elevenlabs/v1/audio-isolation' in doc['paths']
    assert 'multipart/form-data' in doc['paths']['/providers/elevenlabs/v1/audio-isolation']['post']['requestBody']['content']
    assert 'multipart/form-data' in doc['paths']['/providers/elevenlabs/v1/voices/add']['post']['requestBody']['content']
    response=client.post(base+'/arsenal/v1/models/stub-chat/ops/chat',json={'messages':[{'role':'user','content':'hello'}]})
    assert response.status_code==200 and response.headers['content-type']=='text/event-stream',response.text
    assert 'event: token' in response.text and 'event: result' in response.text
    for body in [{},{'messages':[]}]:
        response=client.post(base+'/arsenal/v1/models/stub-chat/ops/chat',json=body)
        assert response.status_code==422,response.text
    assert client.post(base+'/arsenal/v1/models/stub-chat/ops/no-such-op',json={}).status_code==404
    response=client.post(base+'/arsenal/v1/models/stub-voices/ops/speak',json={'text':'hi','voice':'af_heart','stream':True})
    assert response.status_code==200 and len(response.content)==9600
    assert response.headers['x-arsenal-encoding']=='pcm_s16le'
    response=client.post(base+'/arsenal/v1/models/stub-voices/ops/speak',json={'text':'hi','voice':'af_heart','stream':False})
    assert response.status_code==200 and response.content.startswith(b'RIFF'),response.text
    response=client.post(base+'/arsenal/v1/models/stub-images/ops/generate',json={'prompt':'cat'})
    assert response.status_code==200 and response.content.startswith(b'\x89PNG')
    multi=client.post(base+'/arsenal/v1/models/stub-images/ops/generate',json={'prompt':'cat','n':2})
    assert multi.status_code==200 and multi.headers['content-type'].startswith('multipart/mixed') and multi.content.count(b'\x89PNG')==2
    response=client.post(base+'/arsenal/v1/models/stub-voices/ops/speak',json={'text':'hi','out_dir':'/tmp/escape'})
    assert response.status_code==422
    catalog=client.get(base+'/arsenal/v1/catalog?kind=sound&q=stub&sort=size&order=desc').json()
    assert len(catalog)==1 and catalog[0]['row']['kind']=='sound'
    assert client.get(base+'/arsenal/v1/catalog?loaded=maybe').status_code==422
    assert client.get(base+'/arsenal/v1/catalog?sort=bogus').status_code==422
    settings=client.get(base+'/arsenal/v1/models/stub-chat/settings').json()
    assert len(settings)==2
    response=client.post(base+'/arsenal/v1/models/stub-chat/settings',json={'values':{'context_limit':8192}})
    assert response.status_code==200 and any(s['pending'] for s in response.json()),response.text
    response=client.post(base+'/arsenal/v1/models/stub-chat/settings',json={'values':{},'apply':True})
    assert response.status_code==200 and not any(s['pending'] for s in response.json())
    assert client.post(base+'/arsenal/v1/models/stub-chat/settings',json={'values':{},'apply':'yes'}).status_code==422
    assert client.post(base+'/arsenal/v1/models/stub-absent/remove',json={'delete_weights':False}).status_code==200
    assert client.post(base+'/arsenal/v1/models/stub-absent/add',json={}).status_code==200
    assert client.post(base+'/arsenal/v1/models/stub-images/remove',json={'delete_weights':False}).status_code==200
    assert not any(row['id']=='stub-images' for row in client.get(base+'/arsenal/v1/models').json())
    assert not any(row['id']=='stub-images' for row in client.get(base+'/v1/models').json()['data'])
    assert client.post(base+'/v1/images/generations',json={'model':'stub-images','prompt':'cat','size':'256x256'}).status_code==404
    assert client.post(base+'/arsenal/v1/models/stub-images/add',json={}).status_code==200
    assert client.post(base+'/v1/images/generations',json={'model':'stub-images','prompt':'cat','size':'256x256'}).status_code==200
    assert any(row['id']=='stub-images' for row in client.get(base+'/arsenal/v1/status').json()['loaded_models'])
    assert client.post(base+'/arsenal/v1/models/stub-images/add',json={}).status_code==200


def approval_and_reasoning():
    rows=client.get(base+'/arsenal/v1/models').json()
    local=next(row for row in rows if row['id']=='stub-local')
    assert local['codeConsentRequired'] and local['contentDigest']=='a'*64
    for action in ['get','load']:
        assert client.post(base+'/arsenal/v1/'+action,json={'id':'stub-local'}).status_code==409
    path=base+'/arsenal/v1/models/stub-local/approve'
    assert client.post(path,json={'content_digest':'b'*64}).status_code==409
    assert client.post(path,json={}).status_code==422
    assert client.post(path,json={'content_digest':local['contentDigest']}).status_code==200
    rows=client.get(base+'/arsenal/v1/models').json()
    assert not next(row for row in rows if row['id']=='stub-local')['codeConsentRequired']
    openai=OpenAI(api_key='local',base_url=base+'/v1',max_retries=0)
    args={'model':'stub-chat','messages':[{'role':'user','content':'Hi'}],'extra_body':{'thinking':True}}
    result=openai.chat.completions.create(**args)
    assert result.choices[0].message.content==TEXT
    assert result.choices[0].message.reasoning_content=='I will answer briefly.'
    chunks=list(openai.chat.completions.create(**args,stream=True))
    assert ''.join(getattr(x.choices[0].delta,'reasoning_content','') or '' for x in chunks)=='I will answer briefly.'
    assert ''.join(x.choices[0].delta.content or '' for x in chunks)==TEXT
    ant=Anthropic(api_key='local',base_url=base+'/providers/anthropic',max_retries=0)
    args={'model':'stub-chat','max_tokens':64,'messages':[{'role':'user','content':'Hi'}],'thinking':{'type':'enabled','budget_tokens':1024}}
    message=ant.messages.create(**args)
    assert message.content[0].type=='thinking' and message.content[0].thinking=='I will answer briefly.'
    assert message.content[1].text==TEXT
    with ant.messages.stream(**args) as stream:
        final=stream.get_final_message()
    assert final.content[0].thinking=='I will answer briefly.' and final.content[1].text==TEXT
    args['thinking']={'type':'disabled'}
    message=ant.messages.create(**args);assert len(message.content)==1 and message.content[0].text==TEXT

def elevenlabs_new_kinds():
    c=ElevenLabs(api_key='local',base_url=base+'/providers/elevenlabs',timeout=10)
    for method,kwargs in [(c.text_to_sound_effects.convert,{'text':'Bird calls','model_id':'stub-sound','duration_seconds':.2}), (c.music.compose,{'prompt':'Quiet piano','music_length_ms':200}), (c.music.stream,{'prompt':'Quiet piano','music_length_ms':200})]:
        audio=b''.join(method(output_format='pcm_44100',**kwargs))
        assert len(audio)==17640,len(audio)
    source=io.BytesIO()
    with wave.open(source,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(b'\0'*9600)
    for method in [c.audio_isolation.convert,c.audio_isolation.stream]:
        result=b''.join(method(audio=('speech.wav',source.getvalue(),'audio/wav')))
        assert result.startswith(b'RIFF')
    preview=c.text_to_voice.design(voice_description='A warm narrator',text='Hello.',output_format='pcm_24000')
    assert len(preview.previews)==3 and len(base64.b64decode(preview.previews[0].audio_base_64))==9600
    saved=c.text_to_voice.create(voice_name='Narrator',voice_description='Warm',generated_voice_id=preview.previews[0].generated_voice_id)
    assert saved.voice_id=='cloned-stub'
    cloned=c.voices.ivc.create(name='Ada',files=[('speech.wav',source.getvalue(),'audio/wav')],labels=json.dumps({'transcript':'Hello'}))
    assert cloned.voice_id=='cloned-stub'

async def elevenlabs_new_kinds_async():
    c=AsyncElevenLabs(api_key='local',base_url=base+'/providers/elevenlabs',timeout=10)
    audio=b''.join([part async for part in c.text_to_sound_effects.convert(text='Bird',model_id='stub-sound',output_format='pcm_44100')])
    assert len(audio)==17640
    audio=b''.join([part async for part in c.music.stream(prompt='Piano',output_format='pcm_44100')])
    assert len(audio)==17640
    audio=b''.join([part async for part in c.audio_isolation.stream(audio=('speech.pcm',b'\0'*9600),file_format='pcm_s16le_16')])
    assert audio.startswith(b'RIFF')
    preview=await c.text_to_voice.design(voice_description='Warm narrator',text='Hi.',output_format='pcm_24000')
    assert len(preview.previews)==3


def replicate_sound():
    c=replicate.Client(api_token='local',base_url=base+'/providers/replicate',timeout=10)
    for op in ['generate_sound','generate_music']:
        output=c.run('local/stub-sound',input={'prompt':'Piano','op':op,'duration':.2})
        assert client.get(str(output[0])).content.startswith(b'RIFF')
    output=c.run('local/stub-sound',input={'prompt':'Bird','output_type':'data_uri'})
    assert output[0].read().startswith(b'RIFF')
    prediction=c.predictions.create(version='stub-sound',input={'prompt':'slow piano'},wait=False)
    prediction.cancel();prediction.reload();assert prediction.status=='canceled'

def adverse_transport():
    host=urlparse(base).netloc
    cases=[
        (403,f'GET /v1/models HTTP/1.1\r\nHost: evil.example\r\n\r\n'.encode()),
        (403,f'GET /v1/models HTTP/1.1\r\nHost: {host}\r\nOrigin: http://evil.example\r\n\r\n'.encode()),
        (400,f'GET /v1/models HTTP/1.1\r\nHost: {host}\r\nHost: {host}\r\n\r\n'.encode()),
        (413,f'POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: 999999999\r\n\r\n'.encode()),
        (400,f'POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: 0\r\nTransfer-Encoding: chunked\r\n\r\n'.encode()),
        (400,f'POST /v1/images/edits HTTP/1.1\r\nHost: {host}\r\nContent-Type: multipart/form-data; boundary=x\r\nContent-Length: 3\r\n\r\nbad'.encode()),
    ]
    for status, request in cases:
        response=raw_request(request)
        assert response.startswith(f'HTTP/1.1 {status} '.encode()),response
    body=json.dumps({'model':'stub-chat','messages':[{'role':'user','content':'hi'}]}).encode()
    fragments=b''.join(f'{len(part):x}\r\n'.encode()+part+b'\r\n' for part in [body[:7],body[7:19],body[19:]])+b'0\r\n\r\n'
    response=raw_request(f'POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n'.encode()+fragments)
    assert response.startswith(b'HTTP/1.1 200 '),response
    for model in ['absent','stub-absent']:
        response=client.post(base+'/v1/chat/completions',json={'model':model,'messages':[{'role':'user','content':'hi'}]})
        assert response.status_code==404
        if model=='stub-absent': assert 'arsenal_download(model: stub-absent)' in response.text
    assert client.post(base+'/v1/chat/completions',content=b'{',headers={'Content-Type':'application/json'}).status_code==400
    assert client.post(base+'/v1/images/generations',json={'model':'stub-chat','prompt':'cat'}).status_code==404
    assert client.get(base+'/arsenal/files/../../status.json').status_code!=200
    for row in client.get(base+'/arsenal/v1/models').json(): assert all(key in row for key in ['downloadBytes','verification','needsReload','codeConsentRequired','contentDigest'])


def disconnect():
    before=client.get(base+'/arsenal/v1/models').json()[0]['metrics']['canceled_runs']
    address=urlparse(base)
    body=json.dumps({'model':'stub-chat','messages':[{'role':'user','content':'slow'}],'stream':True}).encode()
    with socket.create_connection((address.hostname,address.port),timeout=5) as sock:
        sock.sendall(f'POST /v1/chat/completions HTTP/1.1\r\nHost: {address.netloc}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n'.encode()+body)
        assert b'200 OK' in sock.recv(4096)
    for _ in range(40):
        after=client.get(base+'/arsenal/v1/models').json()[0]['metrics']['canceled_runs']
        if after>before: return
        time.sleep(.025)
    raise AssertionError('Client disconnect did not cancel the service run')


def restart_predictions(server):
    global base
    response=client.post(base+'/providers/replicate/v1/predictions',json={'version':'stub-images','input':{'prompt':'cat'}})
    prediction=response.json()
    for _ in range(100):
        prediction=client.get(prediction['urls']['get']).json()
        if prediction['status']=='succeeded': break
        time.sleep(.02)
    completed_id=prediction['id']
    interrupted=client.post(base+'/providers/replicate/v1/predictions',json={'version':'stub-images','input':{'prompt':'restart cat'}}).json()['id']
    server.stop(); base=server.start()
    completed=client.get(base+'/providers/replicate/v1/predictions/'+completed_id).json()
    assert completed['status']=='succeeded' and client.get(completed['output'][0]).content.startswith(b'\x89PNG')
    failed=client.get(base+'/providers/replicate/v1/predictions/'+interrupted).json()
    assert failed['status']=='failed' and 'not regenerated' in failed['error']


def golden_decision_transport(server):
    global base
    fixtures=[]
    for filename in sorted((ROOT/'Workers/Decisions/Tests/DecisionsTests/Fixtures/systemone').glob('*-fp16.json')):
        source=json.loads(filename.read_text())
        for case in source['cases']:
            request=json.dumps(case['request'],ensure_ascii=False,separators=(',',':'))
            response=case['response']
            frame={**response,'ok':True,'id':777,'bits':16} if case['status']==200 else {'ok':False,'id':777,'error':{'code':'validation_error','status':case['status'],'body':response}}
            fixtures.append({'model':source['model'],'requestJSON':request,'frameJSON':json.dumps(frame,ensure_ascii=False,sort_keys=True,separators=(',',':')),'responseJSON':json.dumps(response,ensure_ascii=False,sort_keys=True,separators=(',',':')),'status':case['status']})
    assert len(fixtures)==39
    package=CACHE/'decision-golden-fixtures.json';package.write_text(json.dumps(fixtures,ensure_ascii=False))
    server.stop();base=server.start(fixtures=package)
    for fixture in fixtures:
        reply=client.post(base+'/v1/systemone',content=fixture['requestJSON'].encode(),headers={'Content-Type':'application/json'})
        assert reply.status_code==fixture['status'],reply.text
        assert reply.content==fixture['responseJSON'].encode(),reply.text[:250]


def auto_decision_routing():
    # golden_decision_transport has installed the actual five model IDs in the fixture host.
    models=client.get(base+'/providers/typesafe/v1/models').json()['models']
    assert models[0]['name']=='auto'
    for state,expected in [('The export button crashes.','laya-english'),('Zażółć gęślą jaźń.','laya-multilingual')]:
        response=client.post(base+'/v1/systemone',json={'model':'auto','state':state,'questions':QUESTIONS})
        assert response.status_code==200 and response.json()['model']==expected,response.text
    response=client.post(base+'/v1/judge',json={'model':'auto','items':['Works.','Zażółć.','Broken.','Żółty.'],'questions':QUESTIONS})
    assert response.status_code==200,response.text
    assert [r['model'] for r in response.json()['results']]==['laya-english','laya-multilingual','laya-english','laya-multilingual']
    original='{"model":"laya-english","state":{"z":"first","a":"second"},"questions":{"z":{"type":"choice","criteria":{"z":null,"a":null}},"a":{"type":"noul"}},"raw_order_probe":true}'
    standard=client.post(base+'/v1/systemone',content=original.encode(),headers={'Content-Type':'application/json'})
    universal=client.post(base+'/arsenal/v1/models/laya-english/ops/decide',content=original.encode(),headers={'Content-Type':'application/json'})
    assert standard.status_code==universal.status_code==200,(standard.text,universal.text)
    assert standard.content==universal.content and universal.json()['request_bytes']==original

def main():
    global base,client
    server=Server(); results=[]
    try:
        base=server.start()
        client=httpx.Client(timeout=10,event_hooks={'request':[local_only]})
        tests=[('OpenAI sync',openai_sync),('OpenAI async',lambda:asyncio.run(openai_async())),('Anthropic sync',anthropic_sync),('Anthropic async',lambda:asyncio.run(anthropic_async())),('ElevenLabs sync',elevenlabs_sync),('ElevenLabs async',lambda:asyncio.run(elevenlabs_async())),('Replicate sync',replicate_sync),('Replicate async',lambda:asyncio.run(replicate_async())),('System One + batch',systemone),('Invalid numeric/control',invalid_numeric_and_chat_controls),('Transient outputs',transient_outputs),('Adverse transport',adverse_transport),('Universal + OpenAPI',universal_and_openapi),('Approval + reasoning',approval_and_reasoning),('ElevenLabs new kinds',elevenlabs_new_kinds),('ElevenLabs new async',lambda:asyncio.run(elevenlabs_new_kinds_async())),('Replicate sound',replicate_sound),('Disconnect cancel',disconnect),('Prediction restart',lambda:restart_predictions(server)),('Decision golden bytes',lambda:golden_decision_transport(server)),('Auto decision routing',auto_decision_routing)]
        from capabilities import groups
        tests = groups(base,client) + tests
        for name,test in tests:
            try: test(); results.append({'name':name,'passed':True}); print(f'{name:<22} PASS',flush=True)
            except Exception as error:
                results.append({'name':name,'passed':False,'error':str(error),'trace':traceback.format_exc()}); print(f'{name:<22} FAIL {type(error).__name__}: {str(error)[:200]}',flush=True)
        (CACHE/'results.json').write_text(json.dumps({'sdk_versions':{'openai':'2.6.1','anthropic':'0.72.0','elevenlabs':'2.21.0','replicate':'1.0.7','cohere':'7.2.0','deepl':'1.32.0','mistralai':'1.9.11'},'results':results},indent=2))
        print(f"{sum(x['passed'] for x in results)}/{len(results)} groups passed; evidence {CACHE/'results.json'}")
        return 0 if all(x['passed'] for x in results) else 1
    finally:
        if client: client.close()
        server.stop()

if __name__=='__main__': sys.exit(main())
