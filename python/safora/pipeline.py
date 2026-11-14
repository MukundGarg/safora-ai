import asyncio
from collections import deque
import numpy as np
from safora.audio.stream import WindowStream,TemporalDetector,waveform
from safora.asr.offline import EnergyVAD
class Pipeline:
    """Legacy asyncio adapter with independent acoustic/speech locks and bounded queues.
    Caller must feed normalized 16 kHz blocks no longer than one second from its
    event loop. For capture callbacks use Task 1 RealtimeScreamMonitor plus the
    independent Task 2 StreamingASR, not this legacy asyncio.submit method.
    """
    def __init__(self,yamnet,asr,classifier,engine,on_event,on_status=lambda s:None):
        self.yamnet,self.asr,self.classifier,self.engine=yamnet,asr,classifier,engine
        self.on_event,self.on_status=on_event,on_status
        self.audio=asyncio.Queue(maxsize=8);self.speech=asyncio.Queue(maxsize=2)
        self.windows=WindowStream();self.detector=TemporalDetector();self.vad=EnergyVAD();self.enabled=True
        self.chunk=[];self.count=0;self.time=0;self.lock=asyncio.Lock();self.speech_lock=asyncio.Lock();self.context=deque(maxlen=4);self.gap=False
    def submit(self,x):
        x=waveform(x)
        if len(x)>16000: raise ValueError('Capture block exceeds one second')
        try: self.audio.put_nowait(x.copy());return True
        except asyncio.QueueFull:
            self.gap=True;self.on_status('audio_gap: capture queue full');return False
    async def capture_worker(self):
        while True:
            x=await self.audio.get()
            try:
                if self.gap:
                    self.windows.clear();self.detector=TemporalDetector();self.context.clear()
                    for fragment in self.chunk:fragment.fill(0)
                    self.chunk=[];self.count=0;self.gap=False
                for t,w in self.windows.feed(x):
                    async with self.lock: scores=await asyncio.to_thread(self.yamnet.predict,w)
                    event=self.detector.update(t+.975,scores['screaming'])
                    if event:
                        result=self.engine.update(max(self.engine.clock,t+.975),acoustic=event)
                        if result:self.on_event(result)
                self.time+=len(x)/16000
                if self.enabled and self.asr is not None:
                    self.chunk.append(x.copy());self.count+=len(x)
                    if self.count>=8*16000:
                        chunk=np.concatenate(self.chunk)
                        for fragment in self.chunk:fragment.fill(0)
                        self.chunk=[];self.count=0
                        if self.vad.speech(chunk):
                            try:self.speech.put_nowait(chunk)
                            except asyncio.QueueFull:
                                chunk.fill(0);self.on_status('conversation_gap: ASR backlog')
                else:self.chunk=[];self.count=0
            except Exception as exc:
                self.on_status('acoustic_unavailable:'+type(exc).__name__);self.windows.clear()
            finally:x.fill(0);self.audio.task_done()
    async def speech_worker(self):
        while True:
            chunk=await self.speech.get()
            try:
                if not self.enabled:
                    self.context.clear();continue
                async with self.speech_lock:
                    transcript=await asyncio.to_thread(self.asr.transcribe,chunk)
                    self.context=deque(((t,text) for t,text in self.context if self.time-t<=30),maxlen=4)
                    history=[text for _,text in self.context]
                    risk=await asyncio.to_thread(self.classifier.classify,transcript.text,history,transcript.uncertain)
                    self.context.append((self.time,transcript.text))
                event=self.engine.update(max(self.engine.clock,self.time),risk=risk)
                if event:self.on_event(event)
            except Exception as exc:self.on_status('conversation_unavailable:'+type(exc).__name__)
            finally:chunk.fill(0);self.speech.task_done()

    async def stop(self):
        """Caller must first cancel/join both workers, then clear queued audio."""
        self.enabled=False;self.windows.clear();self.context.clear()
        for x in self.chunk:x.fill(0)
        self.chunk=[];self.count=0
        for queue in [self.audio,self.speech]:
            while not queue.empty():
                queue.get_nowait().fill(0);queue.task_done()
        self.on_status('stopped; conversation and audio buffers cleared')
