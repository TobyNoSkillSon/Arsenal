# SPDX-License-Identifier: MIT
"""APFS clone-copy prepared local weights into isolated snapshots. CPU only, run through mlock work.
The real host disables network fetching; this is an API integration proof, not a fresh HF download.
"""
import argparse,hashlib,json,os,pathlib,subprocess,tomllib,uuid
ROOT=pathlib.Path(__file__).resolve().parents[2]
SUPPORT=pathlib.Path(os.environ['ARSENAL_SUPPORT_DIR'])
DEV=pathlib.Path.home()/'Library/Application Support/Arsenal-dev/weights'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def seed(identifier,revision,dtype,source,pins=None,model=None):
    assert source.is_dir(),f'Missing prepared local weights: {source}'
    destination=SUPPORT/f'weights/snapshots/{identifier}/{revision}/{dtype}'
    destination.mkdir(parents=True,exist_ok=True)
    donorfile=source/'arsenal-snapshot.json'
    donor=json.loads(donorfile.read_text()) if donorfile.exists() else None
    if donor:
        assert (donor['model'],donor['revision'],donor['precision'])==(identifier,revision,dtype),'Snapshot donor identity mismatch'
    files=[]
    paths=[source/f['path'] for f in donor['files']] if donor else [source/name for name in pins['files']] if pins else sorted(source.rglob('*'))
    for path in paths:
        relative=path.relative_to(source)
        if not path.is_file() or any(part.startswith('.') for part in relative.parts) or path.name in {'arsenal-derived.json','arsenal-snapshot.json'}:continue
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():subprocess.run(['cp','-cL',str(path),str(target)],check=True,timeout=30)
        hash=digest(target)
        if pins:
            assert target.stat().st_size==pins['bytes'][str(relative)] and hash==pins['sha256'][str(relative)],f'Pinned local source mismatch: {relative}'
        files.append({'path':str(relative),'bytes':target.stat().st_size,'sha256':hash})
    assert files
    snapshot={'model':identifier,'revision':revision,'precision':dtype,'files':files}
    if donor and donor.get('derived'):
        derived=donor['derived']
        assert model and derived['recipe']==model['derive'][dtype] and derived['sourceRevision']==model['weights']['revision'],'Derived donor recipe mismatch'
        assert derived['sourceHashes']==model['weights']['sha256'],'Derived donor source hashes do not match accepted pins'
        assert sorted(derived['files'],key=lambda f:f['path'])==sorted(files,key=lambda f:f['path']),'Derived donor output hashes changed'
        # Upgrade old, verified QA metadata only after exact source/output hash
        # checks. Never manufacture a derive from a raw weight directory.
        derived['sourceRepository']=model['weights']['repo']
        derived['sourceFiles']=[{'path':name,'bytes':model['weights']['bytes'][name],'sha256':model['weights']['sha256'][name]} for name in model['weights']['files']]
        snapshot['derived']=derived
        (destination/'arsenal-derived.json').write_text(json.dumps(derived))
    elif model and dtype!=model['weights']['dtype'] and dtype not in model.get('vendor',{}):
        raise AssertionError('Use a verified derived snapshot; raw seeding cannot invent derive provenance')
    (destination/'arsenal-snapshot.json').write_text(json.dumps(snapshot))
    return sum(f['bytes'] for f in files)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('model',nargs='?',default='kokoro-82m');parser.add_argument('--dtype');parser.add_argument('--source',type=pathlib.Path);parser.add_argument('--pinned-only',action='store_true');parser.add_argument('--fresh',action='store_true')
    args=parser.parse_args();manifest=tomllib.loads((ROOT/f'models/{args.model}/model.toml').read_text())
    dtype=args.dtype or ('fp16' if manifest['weights']['dtype']=='fp32' else manifest['weights']['dtype'])
    family='voices' if manifest['kind']=='voices' else 'decisions' if manifest['kind']=='decisions' else 'mb-v2-sound'
    source=args.source or DEV/family/args.model/dtype
    spec=manifest.get('vendor',{}).get(dtype,manifest['weights'])
    if args.pinned_only:assert dtype==spec['dtype'],'Pinned-only seeding cannot pretend a cast is already derived'
    if args.fresh:
        destination=SUPPORT/f"weights/snapshots/{args.model}/{spec['revision']}/{dtype}"
        if destination.exists():
            previous=destination.with_name(dtype+'.previous-'+uuid.uuid4().hex)
            destination.rename(previous);print(f'Preserved previous QA snapshot at {previous}',flush=True)
    total=seed(args.model,spec['revision'],dtype,source,spec if args.pinned_only else None,manifest)
    configfile=SUPPORT/'config.json';config=json.loads(configfile.read_text()) if configfile.exists() else {}
    config.setdefault('selections',{})[args.model]={'precision':dtype,'fast':True,'keepHot':False}
    for asset in manifest.get('asset',[]):
        if manifest.get('assets') is not None and asset['id'] not in manifest['assets']:continue
        total+=seed('asset-'+asset['id'],asset['revision'],'fp16',DEV/family/'assets'/asset['id'])
    configfile.write_text(json.dumps(config))
    print(f'Cloned and hashed {args.model} {dtype}: {total} bytes, 0 network bytes',flush=True)
if __name__=='__main__':main()
