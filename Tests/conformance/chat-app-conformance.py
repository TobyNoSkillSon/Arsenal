# SPDX-License-Identifier: MIT
"""Prepared Layer 3 against an existing app. Caller owns the model/memory decision; wrapper holds gpulock."""
import argparse, asyncio, json, os, pathlib, signal
from urllib.parse import urlparse
import httpx
from openai import OpenAI, AsyncOpenAI
from anthropic import Anthropic

MODEL = 'qwen3.8-27b'

def local_only(request):
    if request.url.host not in {'127.0.0.1', 'localhost'}:
        raise RuntimeError('Chat conformance must remain on loopback.')

async def async_local_only(request): local_only(request)

async def async_chat(base):
    async with AsyncOpenAI(api_key='local',base_url=base+'/v1',max_retries=0,timeout=45,
                           http_client=httpx.AsyncClient(timeout=45,event_hooks={'request':[async_local_only]})) as client:
        reply=await client.chat.completions.create(model=MODEL,messages=[{'role':'user','content':'Say hello.'}],max_tokens=8,temperature=0)
        assert reply.model==MODEL and reply.choices[0].message.content.strip(), 'Empty or wrongly identified async answer.'
        return reply.choices[0].message.content

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-url',required=True,help='App origin, e.g. http://127.0.0.1:8798 (not /v1)')
    args=parser.parse_args()
    base=args.base_url.rstrip('/')
    address=urlparse(base)
    if address.scheme!='http' or address.hostname not in {'127.0.0.1','localhost'} or address.path not in {'','/'}:
        parser.error('Use the local app HTTP origin without a path.')
    signal.alarm(170)
    cache=pathlib.Path(os.environ['ARSENAL_CACHE_DIR']);cache.mkdir(parents=True,exist_ok=True)
    evidence={'model':MODEL,'service':'existing app','classification':'functional_API_conformance_not_benchmark'}
    try:
        with httpx.Client(timeout=10,event_hooks={'request':[local_only]}) as raw:
            rows=raw.get(base+'/arsenal/v1/models').raise_for_status().json()
            row=next((row for row in rows if row['id']==MODEL),None)
            if not row: raise RuntimeError('Qwen3.8-27B is not in this app catalog.')
            state=row.get('states',{}).get(row['selectedPrecision'],{}).get('state')
            if state not in {'ready','loaded'}: raise RuntimeError('Get Qwen3.8-27B before this test; conformance never downloads.')
            evidence['precision']=row['selectedPrecision']
        messages=[{'role':'user','content':'Say hello.'}]
        with OpenAI(api_key='local',base_url=base+'/v1',max_retries=0,timeout=45,
                    http_client=httpx.Client(timeout=45,event_hooks={'request':[local_only]})) as client:
            result=client.chat.completions.create(model=MODEL,messages=messages,max_tokens=8,temperature=0)
            assert result.model==MODEL and result.choices[0].message.content.strip()
            stream=client.chat.completions.create(model=MODEL,messages=messages,max_tokens=8,temperature=0,stream=True,stream_options={'include_usage':True})
            chunks=list(stream)
            text=''.join(chunk.choices[0].delta.content or '' for chunk in chunks if chunk.choices)
            assert text.strip() and chunks[-1].usage and chunks[-1].usage.total_tokens>0
            evidence['openai_sync']=result.choices[0].message.content
            evidence['openai_stream']=text
        evidence['openai_async']=asyncio.run(async_chat(base))
        with Anthropic(api_key='local',base_url=base+'/providers/anthropic',max_retries=0,timeout=45,
                       http_client=httpx.Client(timeout=45,event_hooks={'request':[local_only]})) as client:
            reply=client.messages.create(model=MODEL,messages=messages,max_tokens=8,temperature=0)
            assert reply.model==MODEL and any(block.type=='text' and block.text.strip() for block in reply.content)
            evidence['anthropic']=[block.text for block in reply.content if block.type=='text']
        evidence['passed']=True
        print('Qwen3.8-27B app Layer 3: PASS (OpenAI sync/async/SSE + Anthropic).')
    except Exception as error:
        evidence.update(passed=False,error=str(error));print('Qwen3.8-27B app Layer 3: FAIL '+str(error))
    finally:
        (cache/'qwen3.8-27b-app-result.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False))
    return 0 if evidence.get('passed') else 1

if __name__=='__main__': raise SystemExit(main())
