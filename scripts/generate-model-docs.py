#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Render the current declared model catalog. Unknown evidence stays unknown."""
import argparse,json,pathlib,sys,tomllib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sentence(items):
    """Join declared phrases into one sentence; a phrase's own final period is not doubled."""
    return '; '.join(item.rstrip().removesuffix('.') for item in items)+'.'
def tools(m):
    """The windows a model opens in; `operation` (and no declaration) is the generic window built from its operations."""
    names=['the generic window built from its operations' if t=='operation' else t for t in m.get('tools',['operation'])]
    return ', '.join(dict.fromkeys(names))
def bit_ratio(m,dtype):
    """Bits per weight as the app counts them (`ModelManifest.formats`): MLX affine g64 adds half a bit to a derived tier."""
    declared=next((f for f in m.get('format',[]) if f.get('id')==dtype),None)
    if declared and 'bits_per_weight' in declared:return declared['bits_per_weight']
    overhead=0.5 if dtype in m.get('derive',{}) else 0
    return 4+overhead if dtype=='int4' else 8+overhead if dtype=='int8' else 16
def precision_sizes(m,dtype):
    """(download, on disk after Get) for one precision row, as text. A derived tier downloads nothing of its own: it is made
    from the 16-bit download, so that download is not repeated as its size. A disk size the folder does not declare is the
    app's own bit-ratio figure, marked `~`."""
    def total(spec):
        sizes=spec.get('bytes',{})
        return sum(sizes.values()) if len(sizes)==len(spec.get('files',[])) else None
    declared=next((f.get('size_bytes') for f in m.get('format',[]) if f.get('id')==dtype),None)
    vendor=m.get('vendor',{}).get(dtype)
    if vendor is not None:
        size=total(vendor)
        return (str(size) if size is not None else 'unknown', str(declared or size or 'unknown'))
    base=total(m['weights'])
    estimate=None if base is None else int(base*bit_ratio(m,dtype)/(32 if m['weights']['dtype']=='fp32' else 16))
    disk=str(declared) if declared else 'unknown' if estimate is None else str(base) if estimate==base else f'~{estimate}'
    if dtype in ('int8','int4') and dtype in m.get('derive',{}):return ('made from the 16-bit download',disk)
    return (str(base) if base is not None else 'unknown',disk)
# Repositories whose makers gate the download on Hugging Face (Hugging Face's API, 2 Nov 2026): the user accepts the maker's
# terms there and Arsenal downloads with the user's read token. Model id -> the maker who approves.
GATED={'flux3-image-klein':'Black Forest Labs','sam3.1':'Meta','sam-audio-large':'Meta','trellis2-4b':'Meta','idol':'Meta'}
NATIVE_HALF=('language','embed','audio','music','restoration','perception','geometry','qwenimage')
def base_precision(m):
    """`ModelManifest.basePrecision`."""
    if 'precision' in m or any(r.get('worker') in NATIVE_HALF for r in m.get('runner',[])):return m.get('precision',{}).get('dtype','fp16')
    derive=m.get('derive',{})
    for half in ('fp16','bf16'):
        if derive.get(half,'').startswith('cast:'):return half
    return 'fp16' if m['weights']['dtype']=='fp32' else m['weights']['dtype']
def default_precision(m):
    """`ModelManifest.defaultPrecision`: the first listed precision a runner serves (Fish Audio Drama 3 starts on its vendor 4-bit)."""
    base=base_precision(m)
    extra=(set(m.get('derive',{}))|set(m.get('vendor',{}))|{f['id'] for f in m.get('format',[])})-{base}
    names=[r['name'] for r in m.get('runner',[])]
    for dtype in [base]+[d for d in ('int8','int4') if d in extra]+sorted(extra-{'int8','int4'}):
        declared=next((f for f in m.get('format',[]) if f['id']==dtype),None)
        if (declared.get('runners',[]) if declared else names):return dtype
    return base
def download_bytes(m,dtype):
    """`ModelManifest.downloadBytes(precision:)`: the precision's pinned source plus required assets, each file once; None when a pin is missing."""
    sources=[m.get('vendor',{}).get(dtype,m['weights'])]+[a for a in m.get('asset',[]) if 'assets' not in m or a['id'] in m['assets']]
    sizes={}
    for source in sources:
        for name in source['files']:
            digest,size=source.get('sha256',{}).get(name),source.get('bytes',{}).get(name)
            if digest is None or size is None:return None
            sizes[digest]=size
    return sum(sizes.values())
def gigabytes(size):
    """The app's own figure (`formatBytes`): decimal units."""
    return 'unknown' if size is None else f'{size/1e9:.1f} GB' if size>=1e9 else f'{size/1e6:.0f} MB'
def licence(m):
    l=m['licence'];name=l.get('name',l['spdx'])
    return f'[{name}]({l["url"]})' if l.get('url') else name
def render():
    # The shipped catalog decides which folders are models users can get; other folders (chat models a user adds) are not listed.
    shipped={entry['manifest']['id'] for entry in json.loads((ROOT/'Resources/catalog/catalog.json').read_text())['models']}
    models=[m for m in (tomllib.loads(path.read_text()) for path in sorted((ROOT/'models').glob('*/model.toml'))) if m['id'] in shipped]
    sizes={m['id']:download_bytes(m,default_precision(m)) for m in models}
    total=sum(s for s in sizes.values() if s is not None)
    lines=['# Arsenal models','',f'{len(models)} models, {gigabytes(total)} if you get every one. Each downloads on its own, at the precision Arsenal starts it on.','',
        'Generated from model folders by `scripts/generate-model-docs.py`. Figures are declarations/evidence, never estimates from parameter counts. An operation may be integrated before its weights or quality are qualified.','',
        '| Model | What it does | Download | Weights licence |','|---|---|---:|---|']
    for m in models:
        gate=f' · gated: accept {GATED[m["id"]]}\'s terms on Hugging Face' if m['id'] in GATED else ''
        lines+=[f'| [{m["name"]}](../models/{m["id"]}/README.md) | {m.get("purpose",m["summary"])} | {gigabytes(sizes[m["id"]])} | {licence(m)}{gate} |']
    lines+=['']
    for m in models:
        path=ROOT/'models'/m['id']/'model.toml'
        kind=tomllib.loads((ROOT/f'Resources/kinds/{m["kind"]}.toml').read_text())
        ops=m.get('op',[op for op in kind.get('op',[]) if op.get('default',False)])
        lines += [f'## {m["name"]} (`{m["id"]}`)','',m.get('purpose',m['summary']), '',m['summary'],'',f'Good for: {sentence(m.get("good_for",m["tasks"]))}',f'Limits: {sentence(m.get("limits",["Not declared in model.toml; inspect the model README and operation bounds"]))}',f'Parameters: {m.get("params") or "not declared"}.',f'Languages: {", ".join(m["languages"])}.',f'Licence: {licence(m)}' + (f'. {m["licence"]["conditions"].rstrip().removesuffix(".")}' if m['licence'].get('conditions') else '') + (f'. Gated: accept {GATED[m["id"]]}\'s terms on Hugging Face and add a read token in Settings' if m['id'] in GATED else '') + '.',f'Tools: {tools(m)}.','']
        for op in ops:
            lines += [f'- `{op["name"]}`: `POST /arsenal/v1/models/{m["id"]}/ops/{op["name"]}`.']
        lines += ['','| Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |','|---|---:|---:|---:|']
        base=m['weights']['dtype'];cast=[k for k,v in m.get('derive',{}).items() if k in ('fp16','bf16') and v.startswith('cast:')];base=cast[0] if cast else base
        if 'precision' in m or any(r.get('worker') in ('language','embed','audio','music','restoration','perception','geometry','qwenimage') for r in m.get('runner',[])):base=m.get('precision',{}).get('dtype','fp16')
        elif base=='fp32':base='fp16'
        for dtype in [base]+[d for d in ('int8','int4') if d in m.get('derive',{}) or d in m.get('vendor',{})]:
            download,disk=precision_sizes(m,dtype)
            peaks=[]
            for evidence in sorted(path.parent.glob('results/**/*.json')):
                try:r=json.loads(evidence.read_text())
                except (ValueError,OSError):continue
                stats=r.get('stats',{}) if isinstance(r,dict) else {}
                metric=stats.get('metrics',{}).get('peakMemoryGB',{})
                if stats.get('dtype')==dtype and isinstance(metric.get('value'),(int,float)):
                    provenance=metric.get('provenance',{})
                    peaks.append((metric['value'],provenance.get('chip','unknown chip'),provenance.get('date','unknown date'),metric.get('state','unknown')))
            peak='unknown'
            if peaks:
                value,chip,date,state=max(peaks,key=lambda p:p[0])
                peak=f'{value:.3f} ({chip}, {date[:10]}, {state})' 
            lines += [f'| {dtype} | {download} | {disk} | {peak} |']
        lines += ['',f'[Model documentation](../models/{m["id"]}/README.md)','']
    return '\n'.join(lines)+'\n'
def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();destination=ROOT/'docs/MODELS.md';text=render()
    if args.check:
        if not destination.exists() or destination.read_text()!=text:sys.exit('docs/MODELS.md is stale; regenerate it')
        print('MODELS.md matches all model folders')
    else:destination.write_text(text)
if __name__=='__main__':main()
