"""Run each model in a fresh process; Linux ru_maxrss is KiB, Android use PSS."""
import argparse,json,time,resource,wave
from pathlib import Path
import numpy as np
from safora.audio.stream import waveform
from safora.asr.offline import WhisperASR,VoskASR
from safora.evaluation.metrics import error_rates
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('manifest');p.add_argument('--backend',choices=['whisper','vosk'],default='whisper');a=p.parse_args()
m=(WhisperASR if a.backend=='whisper' else VoskASR)(a.model);results=[]
for row in json.loads(Path(a.manifest).read_text()):
    with wave.open(row['wav'],'rb') as f:
        if f.getsampwidth()!=2:raise ValueError('PCM16 required')
        x=waveform(np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').reshape(-1,f.getnchannels()),f.getframerate())
    start=time.perf_counter();out=m.transcribe(x);seconds=time.perf_counter()-start
    results.append({'id':row['id'],'language':row['language'],'seconds':seconds,'real_time_factor':seconds/max(len(x)/16000,.001),**error_rates(row['reference'],out.text),'uncertain':out.uncertain})
print(json.dumps({'model_bytes':sum(p.stat().st_size for p in Path(a.model).rglob('*') if p.is_file()),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'results':results},indent=2))
