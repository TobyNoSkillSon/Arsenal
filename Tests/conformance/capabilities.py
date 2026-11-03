# SPDX-License-Identifier: MIT
"""New SDK wire profiles against deterministic CPU fixtures on the real listener."""
import asyncio,base64,io,json,time,wave
import httpx,cohere,deepl
from mistralai import Mistral
from openai import OpenAI,AsyncOpenAI
from anthropic import Anthropic
from pydantic import TypeAdapter
from openai.types.realtime import RealtimeServerEvent

def groups(base,client):
    def wav():
        out=io.BytesIO()
        with wave.open(out,'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(b'\0'*4800)
        return out.getvalue()
    ai=OpenAI(api_key='local',base_url=base+'/v1',http_client=client,websocket_base_url=base.replace('http:','ws:')+'/v1')
    png='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg=='
    def transcription():
        result=ai.audio.transcriptions.create(model='stub-audio',file=('fixture.wav',wav()),response_format='verbose_json',timestamp_granularities=['word','segment'])
        assert result.text=='Hello world.' and len(result.words)==2
        diarized=ai.audio.transcriptions.create(model='stub-audio',file=('fixture.wav',wav()),response_format='diarized_json')
        assert diarized.segments[0].speaker=='A'
        result=ai.audio.translations.create(model='stub-audio',file=('fixture.wav',wav()))
        assert result.text=='Hello world.'
        events=list(ai.audio.transcriptions.create(model='stub-audio',file=('fixture.wav',wav()),stream=True))
        assert [e.type for e in events]==['transcript.text.delta','transcript.text.delta','transcript.text.done']
        for format,marker in [('srt','00:00:00,000'),('vtt','WEBVTT')]:
            result=ai.audio.transcriptions.create(model='stub-audio',file=('fixture.wav',wav()),response_format=format)
            assert marker in result
    def embedding():
        # Official SDK defaults to base64 and decodes little-endian float32 on receipt.
        result=ai.embeddings.create(model='stub-embeddings',input=['a','b'],dimensions=3)
        assert len(result.data)==2
        assert len(result.data[0].embedding)==3 and abs(result.data[0].embedding[0]-1/3)<1e-6
        result=ai.embeddings.create(model='stub-embeddings',input=['a'],encoding_format='float',dimensions=2)
        assert result.data[0].embedding==[.5,1]
        result=ai.embeddings.create(model='stub-embeddings',input=[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+png}}])
        assert len(result.data)==1
        result=ai.embeddings.create(model='stub-embeddings',input=[{'type':'input_audio','input_audio':{'data':base64.b64encode(wav()).decode(),'format':'wav'}}])
        assert len(result.data)==1
    def cohere_profiles():
        co=cohere.ClientV2(api_key='local',base_url=base+'/providers/cohere',httpx_client=client)
        result=co.rerank(model='stub-embeddings',query='q',documents=['a','b'],top_n=1)
        assert len(result.results)==1 and result.results[0].index==0
        result=co.embed(model='stub-embeddings',texts=['a','b'],input_type='search_document',embedding_types=['float'])
        assert len(result.embeddings.float_)==2
        result=co.embed(model='stub-embeddings',images=['data:image/png;base64,'+png],input_type='search_document',embedding_types=['float'])
        assert len(result.embeddings.float_)==1
        mixed=co.embed(model='stub-embeddings',inputs=[{'content':[{'type':'text','text':'caption'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+png}}]}],input_type='search_document',embedding_types=['float'])
        assert len(mixed.embeddings.float_)==1 and mixed.meta.billed_units.input_tokens==2
        for path in ['/v1/rerank','/v2/rerank']:
            response=client.post(base+path,json={'model':'stub-embeddings','query':'q','documents':['a'],'return_documents':True,'instruction':'Find the answer'})
            assert response.status_code==200,response.text
            assert response.json()['results'][0]['document']['text']=='a'
            assert response.json()['results'][0]['relevance_score']==.8
    def deepl_profile():
        translator=deepl.DeepLClient('local',server_url=base+'/providers/deepl/')
        result=translator.translate_text(['hello','world'],target_lang='PL')
        assert [r.text for r in result]==['Translated: hello','Translated: world']
        languages=translator.get_target_languages()
        assert any(v.code=='PL' for v in languages)
    def mistral_profile():
        with Mistral(api_key='local',server_url=base+'/providers/mistral',client=httpx.Client(timeout=10,event_hooks={'request':[lambda r: (_ for _ in ()).throw(AssertionError('Nonlocal request')) if r.url.host not in ('127.0.0.1','localhost') else None]})) as mistral:
            result=mistral.ocr.process(model='stub-language',document={'type':'image_url','image_url':'data:image/png;base64,'+png})
            assert len(result.pages)==1 and '| A |' in result.pages[0].markdown and '$x^2$' in result.pages[0].markdown
    def multimodal_chat():
        result=ai.chat.completions.create(model='stub-chat',messages=[{'role':'user','content':[{'type':'text','text':'describe'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+png}}]}])
        assert result.choices[0].message.content
        result=ai.chat.completions.create(model='stub-chat',messages=[{'role':'user','content':[{'type':'input_audio','input_audio':{'data':base64.b64encode(wav()).decode(),'format':'wav'}}]}])
        assert result.choices[0].message.content
        result=ai.chat.completions.create(model='stub-chat',messages=[{'role':'user','content':'extract'}],response_format={'type':'json_schema','json_schema':{'name':'fixture','strict':True,'schema':{'type':'object','required':['name'],'properties':{'name':{'type':'string'}},'additionalProperties':False}}})
        assert json.loads(result.choices[0].message.content)=={'name':'fixture'}
        anthropic=Anthropic(api_key='local',base_url=base+'/providers/anthropic',http_client=client)
        result=anthropic.messages.create(model='stub-chat',max_tokens=32,messages=[{'role':'user','content':[{'type':'image','source':{'type':'base64','media_type':'image/png','data':png}},{'type':'text','text':'describe'}]}])
        assert result.content[0].text
    def aliases_jobs():
        upload=client.post(base+'/mcp',json={}) # Authentication still applies; HTTP aliases do not weaken it.
        assert upload.status_code in (401,403)
        result=client.post(base+'/arsenal/v1/text/translate',json={'model':'stub-language','text':['hi'],'target_lang':'PL'})
        assert result.status_code==200,result.text
        result=client.post(base+'/arsenal/v1/models/stub-language/ops/translate',json={'text':['hi'],'target_lang':'PL'},headers={'Prefer':'respond-async'})
        assert result.status_code==202,result.text
        job=result.json()
        for _ in range(20):
            job=client.get(job['urls']['get']).json()
            if job['status'] in ('succeeded','failed','canceled'):break
            time.sleep(.025)
        assert job['status']=='succeeded',job
        assert client.post(job['urls']['cancel']).json()['status']=='succeeded'
        bad=client.post(base+'/arsenal/v1/text/translate',json={'model':'stub-language','text':[],'target_lang':'PL'},headers={'Prefer':'respond-async'})
        assert bad.status_code==422,bad.text
    async def async_profiles():
        async def local_only(request):
            assert request.url.host in ('127.0.0.1','localhost')
        async with httpx.AsyncClient(timeout=10,event_hooks={'request':[local_only]}) as http:
            ai=AsyncOpenAI(api_key='local',base_url=base+'/v1',http_client=http)
            vector=await ai.embeddings.create(model='stub-embeddings',input=['a'],dimensions=2)
            assert vector.data[0].embedding==[.5,1]
            transcript=await ai.audio.transcriptions.create(model='stub-audio',file=('fixture.wav',wav()),response_format='diarized_json')
            assert transcript.segments[0].speaker=='A'
            co=cohere.AsyncClientV2(api_key='local',base_url=base+'/providers/cohere',httpx_client=http)
            ranks=await co.rerank(model='stub-embeddings',query='q',documents=['a'])
            assert ranks.results[0].index==0
            vectors=await co.embed(model='stub-embeddings',texts=['a'],input_type='search_query',embedding_types=['float'])
            assert len(vectors.embeddings.float_)==1
        async with httpx.AsyncClient(timeout=10,event_hooks={'request':[local_only]}) as http:
            async with Mistral(api_key='local',server_url=base+'/providers/mistral',async_client=http) as mistral:
                result=await mistral.ocr.process_async(model='stub-language',document={'type':'image_url','image_url':'data:image/png;base64,'+png})
                assert len(result.pages)==1
    def realtime():
        with ai.realtime.connect(model='stub-audio',extra_query={'intent':'transcription'}) as connection:
            events=TypeAdapter(RealtimeServerEvent)
            def receive():
                event=events.validate_python(connection.recv().model_dump(exclude_unset=True,warnings=False),strict=True)
                assert event.event_id
                return event
            assert receive().type=='session.created'
            connection.session.update(session={'type':'transcription','audio':{'input':{'format':{'type':'audio/pcm','rate':24000},'transcription':{'model':'stub-audio'}}}})
            assert receive().type=='session.updated'
            connection.input_audio_buffer.append(audio=base64.b64encode(b'\0'*4800).decode())
            started=receive();assert started.type=='input_audio_buffer.speech_started'
            connection.input_audio_buffer.commit()
            received=[receive() for _ in range(4)];types=[event.type for event in received]
            assert received[0].item_id==started.item_id==received[-1].item_id
            assert received[-1].usage.type=='duration' and received[-1].usage.seconds==.1
            connection.session.update(session={'type':'transcription','audio':{'input':{'turn_detection':{'type':'semantic_vad','eagerness':'auto','create_response':False,'interrupt_response':False}}}})
            assert receive().type=='session.updated'
            connection.input_audio_buffer.append(audio=base64.b64encode(b'\0'*4800).decode())
            assert receive().type=='input_audio_buffer.speech_started'
            semantic=[receive() for _ in range(4)]
            assert semantic[-1].type=='conversation.item.input_audio_transcription.completed'
            assert types==['input_audio_buffer.speech_stopped','input_audio_buffer.committed','conversation.item.input_audio_transcription.delta','conversation.item.input_audio_transcription.completed'],types
    return [('OpenAI transcription',transcription),('OpenAI embeddings',embedding),('Cohere profiles',cohere_profiles),('DeepL profile',deepl_profile),('Mistral OCR profile',mistral_profile),('Multimodal + schema',multimodal_chat),('Capability jobs',aliases_jobs),('Realtime session',realtime),('Capability async SDKs',lambda:asyncio.run(async_profiles()))]
