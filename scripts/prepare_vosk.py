"""Explicit development provisioning; inference itself never fetches assets."""
import argparse,json,sys,urllib.request
from pathlib import Path
from safora.asr.assets import prepare_model,validate_archive,validate_model_dir


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--language',required=True,choices=['en','hi'])
    p.add_argument('--archive',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--download',action='store_true',help='Explicitly download the pinned archive if absent')
    p.add_argument('--validate-only',action='store_true',help='Verify a prepared receipt and required-file hashes')
    a=p.parse_args();spec=json.loads((root/'models/android-assets.json').read_text())['vosk'][a.language]
    if a.validate_only:
        receipt=json.loads((a.output/'.safora-model.json').read_text())
        if receipt['archive_sha256']!=spec['sha256']:raise ValueError('Prepared model provenance mismatch')
        validate_model_dir(a.output,receipt['file_sha256'])
        print('Prepared local model validated:',a.language);return
    if not a.archive.exists():
        if not a.download:raise FileNotFoundError('Archive absent; pass --download explicitly to provision')
        a.archive.parent.mkdir(parents=True,exist_ok=True)
        temporary=a.archive.with_suffix('.part')
        try:
            with urllib.request.urlopen(spec['url'],timeout=60) as src,temporary.open('wb') as dst:
                size=0
                while block:=src.read(1024*1024):
                    size+=len(block)
                    if size>spec['bytes']:raise ValueError('Oversized asset download')
                    dst.write(block)
            validate_archive(temporary,spec);temporary.replace(a.archive)
        finally:temporary.unlink(missing_ok=True)
    prepare_model(a.archive,a.output,spec)
    print('Prepared checksum-verified local model:',a.language)


if __name__=='__main__':main()
