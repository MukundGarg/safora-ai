import argparse,json,wave
import numpy as np
from safora.audio.stream import waveform,WindowStream,Yamnet,TemporalDetector
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('wav');p.add_argument('--threshold',type=float,default=.3);a=p.parse_args()
with wave.open(a.wav,'rb') as f:
    if f.getsampwidth()!=2:raise ValueError('PCM16 WAV required')
    audio=waveform(np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').reshape(-1,f.getnchannels()),f.getframerate())
m=Yamnet(a.model);d=TemporalDetector(threshold=a.threshold)
for t,x in WindowStream().feed(audio):
    scores=m.predict(x);event=d.update(t+.975,scores['screaming'])
    print(json.dumps({'time':t+.975,'scores':scores,'event':bool(event)}))
