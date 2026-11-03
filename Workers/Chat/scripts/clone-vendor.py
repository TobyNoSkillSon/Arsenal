#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Clone and hash only the explicit Qwen3.8 vendor files; never downloads or writes into the source."""
import argparse,hashlib,json,pathlib,subprocess,time,tomllib
p=argparse.ArgumentParser();p.add_argument('--manifest',default='models/qwen3.8-27b/model.toml');p.add_argument('--evidence',required=True);a=p.parse_args()
m=tomllib.loads(pathlib.Path(a.manifest).read_text());assert m['id']=='qwen3.8-27b'
source=pathlib.Path.home()/'.omlx/models/Qwen/Qwen3.8-27B'
dest=pathlib.Path.home()/'Library/Application Support/Arsenal-dev/weights/chat/qwen3.8-27b/bf16';dest.mkdir(parents=True,exist_ok=True)
rows=[];start=time.monotonic()
for name in m['weights']['files']:
 src=source/name;out=dest/name;before=src.stat()
 assert src.is_file() and not src.is_symlink() and before.st_size==m['weights']['bytes'][name],name
 if not out.exists():subprocess.run(['/bin/cp','-c',str(src),str(out)],check=True,timeout=20)
 assert out.stat().st_ino!=before.st_ino,name
 h=hashlib.sha256()
 with out.open('rb') as f:
  while chunk:=f.read(8*1024*1024):h.update(chunk)
 assert h.hexdigest()==m['weights']['sha256'][name],'vendor hash mismatch: '+name
 after=src.stat();assert (before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns),name
 out.chmod(0o444)
 rows.append(dict(path=name,bytes=before.st_size,sha256=h.hexdigest()))
report=dict(repo=m['weights']['repo'],revision=m['weights']['revision'],source=str(source),clone=str(dest),method='APFS clone cp -c; SHA256 compared with HF pins; source stat unchanged',files=rows,bytes=sum(x['bytes'] for x in rows),verify_s=time.monotonic()-start)
pathlib.Path(a.evidence).write_text(json.dumps(report,indent=2)+'\n')
print(f'Qwen3.8 vendor clone verified: {len(rows)} files, {report["bytes"]} bytes')
