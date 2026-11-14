"""One capture source -> independently bounded acoustic and speech workers.

No microphone is opened here. The host owns permission/opt-in and a single source
of mono16kHz PCM; native work and callbacks never run in submit().
"""
from collections import deque
from dataclasses import dataclass
import queue
import threading
import time
import numpy as np
from .audio.ingestion import RealtimeScreamMonitor, StreamConfig
from .audio.stream import waveform
from .asr.streaming import StreamingASR, ASRConfig
from .asr.offline import EnergyVAD
from .classifier.risk import ContextBaseline
from .classifier.text import Turn


@dataclass(frozen=True)
class Cue:
    session: int
    kind: str  # check_in_requested or manual_sos_requested; host confirms/owns flow.
    source: str
    sample_end: int
    uncertain: bool = True
    external_dispatch: bool = False
    provenance: str = 'MONITORING_ONLY'
    requires_user_confirmation: bool = True
    language: str | None = None
    asr_uncertain: bool | None = None


class MLCoordinator:
    """Fresh owned models per session, bounded queues, explicit loss/coverage.

    stop(timeout=False result) leaves retiring native work owned by its old worker;
    resume is rejected until cleanup completes. Callback consumers must discard old
    session IDs after lifecycle changes; a callback already executing is irrevocable.
    """
    def __init__(self, acoustic_factory, asr_factory, *, on_cue=lambda cue:None,
                 speech_enabled=False, language='en', segment_seconds=8.,
                 cooldown_seconds=10., queue_capacity=2, vad=None):
        if not isinstance(speech_enabled,bool):raise ValueError('Speech opt-in must be boolean')
        if language not in ('en','hi',None): raise ValueError('Explicit en/hi/unknown language required')
        if not np.isfinite(segment_seconds) or not .1<=segment_seconds<=8: raise ValueError('Invalid speech segment bound')
        if not np.isfinite(cooldown_seconds) or cooldown_seconds<0: raise ValueError('Invalid cooldown')
        if not isinstance(queue_capacity,int) or not 1<=queue_capacity<=8: raise ValueError('Invalid queue capacity')
        self.acoustic_factory,self.asr_factory=acoustic_factory,asr_factory
        self.on_cue=on_cue;self.speech_enabled=speech_enabled;self.language=language
        self.segment_samples=int(segment_seconds*16000);self.cooldown=cooldown_seconds
        self.capacity=queue_capacity;self.vad=vad or EnergyVAD()
        self._lock=threading.RLock();self._session=0;self._running=False;self._stopping=False;self._retiring=None
        self._next=0;self._epoch=0;self._context=deque(maxlen=4);self._last=-float('inf')
        self.acoustic=None;self.asr=None;self._speech_thread=None;self._speech_queue=None
        self._speech_stop=threading.Event();self._speech_pending=0
        self._stats=dict(speech_input_drops=0,speech_input_dropped_samples=0,cues=0,callback_errors=0,
                         speech_segments=0,nonempty_transcripts=0,positive_risk_results=0,risk_errors=0,max_risk_seconds=0.,max_asr_seconds=0.)
        self._coverage='stopped'

    @property
    def session(self):
        with self._lock:return self._session

    def _emit(self, source, end, session, manual=False, epoch=None, language=None, asr_uncertain=None):
        with self._lock:
            if not manual and (not self._running or session!=self._session or epoch is not None and epoch!=self._epoch):return
            clock=time.monotonic()
            if not manual and clock-self._last<self.cooldown:return
            if not manual:self._last=clock
            cue=Cue(self._session if manual else session,'manual_sos_requested' if manual else 'check_in_requested',source,end,
                    not manual,False,'USER_ACTION' if manual else 'MONITORING_ONLY',language=language,asr_uncertain=asr_uncertain)
            self._stats['cues']+=1
        try:self.on_cue(cue)
        except Exception:
            with self._lock:self._stats['callback_errors']+=1

    def manual_sos(self):
        # Independent of permissions, opt-in, all models and their worker locks.
        try:self.on_cue(Cue(self._session,'manual_sos_requested','manual',self._next,False,False,'USER_ACTION'))
        except Exception:pass

    def _invalidate(self):
        self._epoch+=1;self._context.clear()
        if self.asr is not None:self.asr.reset()

    def start(self, *, permission=True, opted_in=True):
        with self._lock:
            if self._running:raise RuntimeError('Already running')
            if self._stopping or self._retiring is not None and self._retiring.is_alive():raise RuntimeError('Prior native work still retiring')
            if not permission or not opted_in:
                self._coverage='permission_denied' if not permission else 'opted_out';return False
            self._session+=1;session=self._session;self._next=0;self._last=-float('inf');self._epoch+=1
            self._context.clear();self._running=True;self._coverage='starting';self._speech_stop=threading.Event()
            self._speech_queue=queue.Queue(self.capacity);self._speech_pending=0
            self.asr=StreamingASR({},language=self.language if self.speech_enabled else None,
                config=ASRConfig(max_chunk_seconds=8,queue_capacity=self.capacity),factory=self.asr_factory,
                on_result=lambda event:self._risk(event,session)).start()
            try:
                model=self.acoustic_factory()
                self.acoustic=RealtimeScreamMonitor(model,on_event=lambda event:self._emit('yamnet',round(event.time*16000),session),
                    on_status=lambda status:self._acoustic_status(status,session),config=StreamConfig()).start()
            except Exception:self.acoustic=None
            self._speech_thread=threading.Thread(target=self._segment,name='safora-speech-segmentation',daemon=True)
            self._speech_thread.start();self._coverage='monitoring' if self.acoustic else 'acoustic_unavailable'
            return True

    def _acoustic_status(self,status,session):
        with self._lock:
            if self._running and session==self._session:
                if status['kind']=='audio_gap':self._invalidate()
                elif status['kind']=='acoustic_unavailable':self._coverage='acoustic_unavailable'

    def configure_speech(self,enabled,language):
        if language not in ('en','hi',None) or not isinstance(enabled,bool):raise ValueError('Invalid speech selection')
        with self._lock:
            if (enabled,language)==(self.speech_enabled,self.language):return
            self.speech_enabled,self.language=enabled,language
            self._invalidate()
            if self.asr:self.asr.set_language(language if enabled else None)

    def submit(self,samples,*,start_sample=None,sample_rate=16000,discontinuity=False):
        if sample_rate!=16000:raise ValueError('Supply shared upstream mono16kHz stream')
        raw=np.asarray(samples)
        if raw.ndim!=1 or not 0<len(raw)<=1600:raise ValueError('Capture block must be mono, <=100ms')
        # Bounded validation/conversion only; never model, VAD, risk or callbacks.
        audio=waveform(raw)
        with self._lock:
            if not self._running:return False
            start=self._next if start_sample is None else start_sample
            if not isinstance(start,int) or isinstance(start,bool) or start<self._next:raise ValueError('Nonmonotonic source sample clock')
            gap=discontinuity or start!=self._next
            if gap:self._invalidate()
            self._next=start+len(audio)
            acoustic_ok=self.acoustic.submit(audio,start_sample=start,discontinuity=gap) if self.acoustic else False
            if self.speech_enabled and self.language is not None:
                try:
                    self._speech_queue.put_nowait((audio.copy(),start,self._epoch))
                    self._speech_pending+=1
                except queue.Full:
                    self._stats['speech_input_drops']+=1;self._stats['speech_input_dropped_samples']+=len(audio)
                    self._invalidate()
            return acoustic_ok

    def _segment(self):
        buffer=np.zeros(self.segment_samples,np.float32);size=0;origin=0;epoch=-1;expected=None
        try:
            while not self._speech_stop.is_set():
                try:audio,start,token=self._speech_queue.get(timeout=.05)
                except queue.Empty:continue
                try:
                    with self._lock:valid=self._running and self.speech_enabled and token==self._epoch
                    if not valid:continue
                    if token!=epoch or expected!=start:buffer.fill(0);size=0
                    epoch=token;expected=start+len(audio);offset=0
                    while offset<len(audio):
                        if size==0:origin=start+offset
                        count=min(len(audio)-offset,len(buffer)-size)
                        buffer[size:size+count]=audio[offset:offset+count];size+=count;offset+=count
                        if size==len(buffer):
                            if self.vad.speech(buffer):
                                with self._lock:
                                    if self._running and token==self._epoch:
                                        if not self.asr.submit(buffer,start_sample=origin,final=True,discontinuity=True):self._invalidate()
                                        self._stats['speech_segments']+=1
                            buffer.fill(0);size=0
                except Exception:
                    with self._lock:self._invalidate();self._stats['risk_errors']+=1
                finally:
                    audio.fill(0);self._speech_queue.task_done()
                    with self._lock:self._speech_pending-=1
        finally:buffer.fill(0)

    def _risk(self,event,session):
        with self._lock:
            if not self._running or session!=self._session or not self.speech_enabled or event.generation!=self.asr.snapshot()['generation']:return
            epoch=self._epoch
            self._context=deque(((end,turn) for end,turn in self._context if event.end-end<=30),maxlen=4)
            history=[turn for _,turn in self._context]
        began=time.perf_counter()
        with ContextBaseline() as classifier:risk=classifier.classify(event.transcript.text,history,event.transcript.uncertain)
        elapsed=time.perf_counter()-began
        with self._lock:
            if epoch!=self._epoch or not self._running or session!=self._session:return
            self._stats['nonempty_transcripts']+=bool(event.transcript.text)
            self._stats['positive_risk_results']+=risk.labels!={'neutral_or_uncertain'}
            self._stats['max_risk_seconds']=max(self._stats['max_risk_seconds'],elapsed)
            self._stats['max_asr_seconds']=max(self._stats['max_asr_seconds'],event.processing_seconds)
            self._context.append((event.end,Turn(event.transcript.text,event.transcript.uncertain)))
        if risk.labels!={'neutral_or_uncertain'}:self._emit('UNVALIDATED_HEURISTIC',event.end_sample,session,epoch=epoch,language=event.language,asr_uncertain=event.transcript.uncertain)

    def wait_idle(self,timeout=30.):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            with self._lock:pending=self._speech_pending
            if pending==0:break
            time.sleep(.005)
        return pending==0 and (self.acoustic is None or self.acoustic.wait_idle(max(0,end-time.monotonic()))) and self.asr.wait_idle(max(0,end-time.monotonic()))

    def snapshot(self):
        with self._lock:return dict(self._stats,session=self._session,running=self._running,coverage=self._coverage,
            context_turns=len(self._context),speech_enabled=self.speech_enabled,language=self.language,
            acoustic=self.acoustic.snapshot() if self.acoustic else None,asr=self.asr.snapshot() if self.asr else None)

    def stop(self,*,reason='stopped',timeout=5.):
        with self._lock:
            if not self._running:
                self._coverage=reason
                return self._retiring is None or not self._retiring.is_alive()
            self._running=False;self._stopping=True;self._epoch+=1;self._context.clear();self._coverage=reason;self._speech_stop.set()
            acoustic,asr,segment=self.acoustic,self.asr,self._speech_thread
        deadline=time.monotonic()+timeout
        segment.join(max(0,deadline-time.monotonic()))
        a=acoustic is None or acoustic.stop(max(0,deadline-time.monotonic()))
        b=asr.stop(max(0,deadline-time.monotonic()))
        def release():
            segment.join()
            if acoustic:
                acoustic.stop(None)
                close=getattr(acoustic.processor.model,'close',None)
                if close:close()
            asr.stop(None)
            while True:
                try:audio,_,_=self._speech_queue.get_nowait()
                except queue.Empty:break
                audio.fill(0);self._speech_queue.task_done()
            with self._lock:self._speech_pending=0;self._stopping=False
        self._retiring=threading.Thread(target=release,name='safora-retiring',daemon=True);self._retiring.start()
        self._retiring.join(max(0,deadline-time.monotonic()))
        return a and b and not self._retiring.is_alive()

    def permission_lost(self):return self.stop(reason='permission_lost')
    def opt_out(self):return self.stop(reason='opted_out')
