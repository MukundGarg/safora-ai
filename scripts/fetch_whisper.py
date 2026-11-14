"""Explicit download provisioning, not called by the runtime."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args()
meta=json.loads((Path(__file__).resolve().parents[1]/'models/asset-manifest.json').read_text())['whisper']
root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
for name,expected in meta['sha256'].items():
    target=root/name;temporary=target.with_suffix(target.suffix+'.part')
    url=f"https://huggingface.co/{meta['repo']}/resolve/{meta['revision']}/{name}"
    digest=hashlib.sha256();size=0
    try:
        with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as output:
            while block:=response.read(1024*1024):
                size+=len(block)
                if size>100*1024*1024:raise ValueError('Oversized asset')
                output.write(block);digest.update(block)
        if digest.hexdigest()!=expected:raise ValueError('Model hash mismatch: '+name)
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)
    print(name,size)
