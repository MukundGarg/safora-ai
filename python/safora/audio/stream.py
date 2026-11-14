from collections import deque
from dataclasses import dataclass
from math import gcd
import numpy as np
from scipy.signal import resample_poly

CLASSES = {"screaming":11,"shout":6,"yell":9,"crying":19,"whimper":21,"wail":22}

def waveform(samples, rate=16000):
    x = np.asarray(samples)
    if x.ndim not in (1,2) or rate <= 0: raise ValueError("Invalid audio shape/rate")
    if np.issubdtype(x.dtype, np.integer):
        if x.dtype != np.int16: raise ValueError("Only signed PCM16 integers supported")
        x = x.astype(np.float32) / 32768
    else: x = x.astype(np.float32)
    if x.ndim == 2: x = x.mean(axis=1)
    if not np.isfinite(x).all() or (len(x) and np.max(np.abs(x)) > 1.001):
        raise ValueError("Audio must be finite, normalized PCM")
    if rate != 16000:
        d = gcd(rate,16000); x = resample_poly(x,16000//d,rate//d)
    return np.ascontiguousarray(x,dtype=np.float32)

class WindowStream:
    """Feed contiguous 16 kHz blocks; timestamps use sample count, not wall time."""
    def __init__(self, window=15600, hop=8000):
        if not 0 < hop <= window: raise ValueError("Invalid hop")
        self.window,self.hop=window,hop
        self.buffer=np.empty(window,dtype=np.float32); self.size=0; self.start=0
    def feed(self, samples):
        x=waveform(samples)
        for offset in range(0,len(x),self.hop):
            block=x[offset:offset+self.hop]; pos=0
            while pos<len(block):
                n=min(self.window-self.size,len(block)-pos)
                self.buffer[self.size:self.size+n]=block[pos:pos+n]; self.size+=n; pos+=n
                if self.size==self.window:
                    yield self.start/16000, self.buffer.copy()
                    keep=self.window-self.hop
                    self.buffer[:keep]=self.buffer[self.hop:];self.size=keep;self.start+=self.hop
    def clear(self):
        self.buffer.fill(0);self.size=0;self.start=0

class Yamnet:
    def __init__(self,path,threads=2,interpreter=None):
        if interpreter is None:
            from ai_edge_litert.interpreter import Interpreter
            interpreter=Interpreter(model_path=str(path),num_threads=threads)
        self.model=interpreter;self.model.allocate_tensors()
        self.i=self.model.get_input_details()[0];self.o=self.model.get_output_details()[0]
        if tuple(self.i['shape'])!=(15600,) or self.i['dtype']!=np.float32:
            raise ValueError("Expected classification YAMNet float32[15600]")
        if tuple(self.o['shape'])!=(1,521) or self.o['dtype']!=np.float32:
            raise ValueError("Expected float32[1,521] output")
    def predict(self,x):
        x=waveform(x)
        if x.shape!=(15600,): raise ValueError("Expected 15600 samples")
        self.model.set_tensor(self.i['index'],x);self.model.invoke()
        y=self.model.get_tensor(self.o['index'])[0]
        if not np.isfinite(y).all(): raise ValueError("Invalid model output")
        return {k:float(y[v]) for k,v in CLASSES.items()}

@dataclass
class AcousticEvent:
    time: float
    score: float
    reason: str = "temporally_confirmed_scream"

class TemporalDetector:
    def __init__(self,threshold=.30,hits=2,windows=3,cooldown=10):
        if not 0<=threshold<=1 or not 1<=hits<=windows or cooldown<0: raise ValueError("Invalid detector config")
        self.threshold,self.hits,self.cooldown=threshold,hits,cooldown
        self.history=deque(maxlen=windows);self.last=-float('inf');self.clock=-float('inf')
    def update(self,time,score):
        if time<self.clock or not np.isfinite([time,score]).all() or not 0<=score<=1: raise ValueError("Invalid time/score")
        self.clock=time;self.history.append(score>=self.threshold)
        if sum(self.history)>=self.hits and time-self.last>=self.cooldown:
            self.last=time;self.history.clear();return AcousticEvent(time,score)
