"""Explicit asset provisioning. Model not redistributed in source package."""
import argparse,hashlib,io,tarfile,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args()
url='https://www.kaggle.com/api/v1/models/google/yamnet/tflite/classification-tflite/1/download'
expected='10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de'
with urllib.request.urlopen(url,timeout=60) as response:data=response.read(32*1024*1024+1)
if len(data)>32*1024*1024:raise ValueError('Oversized download')
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
    file=archive.extractfile('1.tflite')
    if file is None:raise ValueError('Missing model member')
    weights=file.read(16*1024*1024+1)
if hashlib.sha256(weights).hexdigest()!=expected:raise ValueError('Model hash mismatch; inspect upstream revision')
path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(weights)
print(f'Wrote {len(weights)} bytes; SHA256 {expected}. Review upstream model license before distribution.')
