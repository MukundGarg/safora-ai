"""Provision checksum-pinned public model assets explicitly, before APK build.
Runtime remains offline. Downloads are cached in work/model-downloads by default.
"""
import argparse,hashlib,json,shutil,sys,urllib.request,zipfile,subprocess
from safora.asr.assets import validate_archive
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=root/'work/model-downloads');p.add_argument('--offline',action='store_true',help='Fail instead of fetching missing or corrupt cached models');a=p.parse_args()
a.cache.mkdir(parents=True,exist_ok=True)
meta=json.loads((root/'models/android-assets.json').read_text())
assets=root/'android/app/src/main/assets';assets.mkdir(parents=True,exist_ok=True)
def publish(source,dest):
    # Idempotent atomic publication; provision before Gradle, never during a build.
    if dest.is_file() and sha(source)==sha(dest):return
    temporary=dest.with_name(dest.name+'.partial')
    try:shutil.copyfile(source,temporary);temporary.replace(dest)
    finally:temporary.unlink(missing_ok=True)

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):digest.update(block)
    return digest.hexdigest()
yam=a.cache/'yamnet.tflite'
if not yam.is_file() or sha(yam)!=meta['yamnet']['sha256']:
    if a.offline:raise ValueError('Offline YAMNet asset unavailable/corrupt')
    subprocess.run([sys.executable,str(root/'scripts/fetch_yamnet.py'),str(yam)],check=True)
publish(yam,assets/'yamnet.tflite')
for lang,spec in meta['vosk'].items():
    path=a.cache/spec['asset']
    if not path.is_file() or sha(path)!=spec['sha256']:
        if a.offline:raise ValueError('Offline Vosk asset unavailable/corrupt')
        temporary=path.with_suffix('.part')
        try:
            with urllib.request.urlopen(spec['url'],timeout=60) as source,temporary.open('wb') as dest:
                size=0
                while block:=source.read(1024*1024):
                    size+=len(block)
                    if size>spec['bytes']:raise ValueError('Oversized download')
                    dest.write(block)
            if sha(temporary)!=spec['sha256']:raise ValueError('Model SHA256 mismatch')
            temporary.replace(path)
        finally:temporary.unlink(missing_ok=True)
    validate_archive(path,spec)
    publish(path,assets/spec['asset'])
    print(f'{lang}: {spec["root"]} ({spec["bytes"]} bytes); {spec["license"]}')
publish(root/'LICENSE',assets/'MODEL-LICENSE-APACHE-2.0.txt')
publish(root/'models/android-model-notices.md',assets/'MODEL-NOTICES.txt')
publish(root/'models/android-runtime-notices.md',assets/'RUNTIME-NOTICES.txt')
publish(root/'models/android-assets.json',assets/'model-assets.json')
print('Prepared offline Android assets at',assets)
