"""Create integration-only TTS fixtures; these are not safety/ASR evaluation data.
Requires local espeak-ng executable. Does not use or imitate a person's recording.
"""
import argparse,subprocess,tempfile,wave
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
p=argparse.ArgumentParser();p.add_argument('--espeak',default='espeak-ng');a=p.parse_args()
root=Path(__file__).resolve().parents[1]
dest=root/'android/app/src/androidTest/assets';dest.mkdir(parents=True,exist_ok=True)
for name,text in [('refusal','I said no. Stop touching me.'),('coffee','Let us go somewhere for coffee.')]:
    with tempfile.TemporaryDirectory() as temp:
        wav=Path(temp)/'tts.wav'
        subprocess.run([a.espeak,'-v','en-us','-s','140','-w',str(wav),text],check=True)
        with wave.open(str(wav),'rb') as f:
            samples=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768
            rate=f.getframerate()
        from math import gcd
        divisor=gcd(rate,16000);audio=resample_poly(samples,16000//divisor,rate//divisor)
        padded=np.zeros(8*16000,dtype='<i2')
        n=min(len(padded),len(audio));padded[:n]=(np.clip(audio[:n],-1,1)*32767).astype('<i2')
        (dest/f'synthetic-{name}.pcm16').write_bytes(padded.tobytes())
        print(name,len(padded),'PCM16 16kHz mono samples, authored synthetic speech')
