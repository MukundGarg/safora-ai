"""Bounded ASR-only worker. No acoustic locks, risk inference, dispatch or transcript logs."""
from dataclasses import dataclass, replace
import queue
import threading
import time
import numpy as np
from .offline import Transcript, VoskASR


@dataclass(frozen=True)
class ASRConfig:
    sample_rate: int = 16000
    queue_capacity: int = 2
    max_chunk_seconds: float = 1.
    max_segment_seconds: float = 12.
    emit_partials: bool = False

    def __post_init__(self):
        if self.sample_rate != 16000 or isinstance(self.sample_rate,bool):
            raise ValueError('ASR input must already be mono 16000 Hz')
        if not isinstance(self.queue_capacity,int) or isinstance(self.queue_capacity,bool) or not 1 <= self.queue_capacity <= 8:
            raise ValueError('Queue capacity must be 1..8')
        if (not np.isfinite([self.max_chunk_seconds,self.max_segment_seconds]).all()
                or not 0 < self.max_chunk_seconds <= 8 or not self.max_chunk_seconds <= self.max_segment_seconds <= 12):
            raise ValueError('Invalid ASR chunk/segment bounds')
        if not isinstance(self.emit_partials,bool): raise ValueError('emit_partials must be boolean')


@dataclass
class ASREvent:
    transcript: Transcript
    language: str
    start_sample: int
    end_sample: int
    segment_id: int
    generation: int
    processing_seconds: float

    @property
    def start(self): return self.start_sample/16000
    @property
    def end(self): return self.end_sample/16000


@dataclass
class _Chunk:
    samples: np.ndarray
    start: int
    end: int
    final: bool
    generation: int
    discontinuity: bool = False


class StreamingASR:
    """One selected model resident; bounded FIFO, drop newest, lazy local loading.

    finish a segment using final=True or ordered flush(). reset() discards queued/
    unfinished recognition and suppresses obsolete output. restart() also reloads
    the selected model. Language switches discard context, never concatenate models.
    None/unknown suspends recognition and releases the model after native work ends.
    stop(False) means an in-flight call is still completing; never share its backend.
    """
    def __init__(self, model_paths, *, language=None, config=None, on_result=lambda event:None,
                 on_status=lambda status:None, factory=None):
        self.config = config or ASRConfig()
        self.paths = dict(model_paths)
        self.on_result, self.on_status = on_result, on_status
        self.factory = factory or (lambda lang: VoskASR(self.paths[lang],language=lang,
                                                       max_seconds=self.config.max_segment_seconds))
        self.queue = queue.Queue(maxsize=self.config.queue_capacity)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._language = self._language_code(language)
        self._generation = self._applied = self._pending = self._next = 0
        self._reload = -1
        self._loaded = False
        self._error = None
        self._stats = dict(accepted_chunks=0,dropped_chunks=0,dropped_samples=0,discarded_chunks=0,
                           gaps=0,results=0,errors=0,callback_errors=0,unknown_language_chunks=0,
                           queue_high_water=0,stale_results=0)

    @staticmethod
    def _language_code(language):
        if language in (None,'unknown'): return None
        if language not in ('en','hi'):
            raise ValueError('Choose en/hi or unknown; automatic/Hinglish identification is not implemented')
        return language

    def start(self):
        with self._condition:
            if self._thread is not None or self._stop.is_set(): raise RuntimeError('Create a fresh ASR controller to start again')
            self._thread = threading.Thread(target=self._run,name='safora-asr',daemon=True)
            self._thread.start()
        return self

    def _discard_queued(self):
        while True:
            try: chunk=self.queue.get_nowait()
            except queue.Empty: break
            chunk.samples.fill(0);self.queue.task_done()
            self._pending-=1;self._stats['discarded_chunks']+=1
        self._condition.notify_all()

    def _invalidate(self,reload=False):
        self._generation+=1
        if reload:self._reload=self._generation
        self._discard_queued()
        return self._generation

    def set_language(self,language):
        selected=self._language_code(language)
        with self._condition:
            if self._stop.is_set():raise RuntimeError('ASR stopped')
            if selected!=self._language:
                self._language=selected;self._invalidate()
            return self._generation

    def reset(self):
        with self._condition:
            if self._stop.is_set():raise RuntimeError('ASR stopped')
            return self._invalidate()

    def restart(self):
        with self._condition:
            if self._stop.is_set():raise RuntimeError('ASR stopped')
            return self._invalidate(reload=True)

    def submit(self,samples,*,start_sample=None,sample_rate=16000,final=True,discontinuity=False):
        if sample_rate!=16000:raise ValueError('Resample before the independent ASR queue')
        x=np.asarray(samples)
        if x.ndim!=1 or not 0<len(x)<=int(self.config.max_chunk_seconds*16000) or not np.issubdtype(x.dtype,np.floating):
            raise ValueError('Expected bounded normalized mono float speech chunk')
        copy=x.copy()
        with self._condition:
            if self._stop.is_set():copy.fill(0);return False
            start=self._next if start_sample is None else start_sample
            if isinstance(start,bool) or not isinstance(start,int) or start<self._next:
                copy.fill(0);raise ValueError('Backwards/overlapping source sample clock')
            self._next=start+len(x)  # Advance even for drop-newest/unknown-language losses.
            if self._language is None:
                copy.fill(0);self._stats['unknown_language_chunks']+=1;return False
            chunk=_Chunk(copy,start,self._next,bool(final),self._generation,bool(discontinuity))
            return self._enqueue(chunk)

    def _enqueue(self,chunk):
        try:self.queue.put_nowait(chunk)
        except queue.Full:
            self._stats['dropped_chunks']+=1;self._stats['dropped_samples']+=len(chunk.samples)
            chunk.samples.fill(0);return False
        self._pending+=1;self._stats['accepted_chunks']+=1
        self._stats['queue_high_water']=max(self._stats['queue_high_water'],self.queue.qsize())
        return True

    def flush(self):
        """Ordered finalization barrier. False on full/unknown/stopped; caller may retry."""
        with self._condition:
            if self._stop.is_set() or self._language is None:return False
            return self._enqueue(_Chunk(np.empty(0,np.float32),self._next,self._next,True,self._generation))

    def _current(self,generation):
        with self._condition:return not self._stop.is_set() and generation==self._generation

    def _callback(self,callback,value):
        try:callback(value)
        except Exception:
            with self._condition:self._stats['callback_errors']+=1

    def _publish(self,transcript,language,start,end,segment,generation,seconds):
        if not self._current(generation):
            with self._condition:self._stats['stale_results']+=1
            return
        self._callback(self.on_result,ASREvent(transcript,language,start,end,segment,generation,seconds))
        with self._condition:self._stats['results']+=1

    def _status(self,kind,generation):
        if self._current(generation):self._callback(self.on_status,{'kind':kind,'generation':generation})

    def _run(self):
        adapter=None;loaded_language=None;applied=-1
        active=False;expected=None;start=end=segment=0;processing=0.
        try:
            while not self._stop.is_set():
                with self._condition: generation,language,reload=self._generation,self._language,self._reload
                if generation!=applied:
                    if adapter is not None:
                        if loaded_language!=language or reload>applied:
                            adapter.close();adapter=None;loaded_language=None
                        else:adapter.reset()
                    active=False;expected=None;processing=0.;applied=generation
                    with self._condition:
                        self._applied=generation;self._loaded=adapter is not None;self._error=None
                        self._condition.notify_all()
                    self._status('unknown_language' if language is None else 'asr_reset',generation)
                try:chunk=self.queue.get(timeout=.05)
                except queue.Empty:continue
                try:
                    if not self._current(chunk.generation) or chunk.generation!=applied:continue
                    if adapter is None:
                        adapter=self.factory(language);loaded_language=language
                        with self._condition:self._loaded=True
                    if not self._current(chunk.generation):continue
                    if chunk.discontinuity or (expected is not None and expected!=chunk.start):
                        adapter.reset();active=False;processing=0.
                        with self._condition:self._stats['gaps']+=1
                        self._status('speech_gap_partial_discarded',generation)
                    if active and chunk.end-start>int(self.config.max_segment_seconds*16000):
                        result=replace(adapter.finish(),uncertain=True,reason='segment_limit')
                        self._publish(result,language,start,end,segment,generation,processing)
                        active=False
                    if not active and len(chunk.samples):
                        adapter.begin();active=True;start=chunk.start;segment+=1;processing=0.
                    if active:
                        began=time.perf_counter()
                        part=adapter.accept(chunk.samples) if len(chunk.samples) else None
                        processing+=time.perf_counter()-began;end=chunk.end
                        if chunk.final or (part is not None and part.is_final):
                            began=time.perf_counter();result=adapter.finish();processing+=time.perf_counter()-began
                            self._publish(result,language,start,end,segment,generation,processing);active=False
                        elif self.config.emit_partials and part and part.text:
                            self._publish(part,language,start,end,segment,generation,processing)
                    expected=chunk.end
                    with self._condition:self._error=None
                except Exception as exc:
                    active=False;expected=None
                    if adapter is not None:
                        try:adapter.reset()
                        except Exception:
                            try:adapter.close()
                            except Exception:pass
                            adapter=None;loaded_language=None
                            with self._condition:self._loaded=False
                    with self._condition:self._stats['errors']+=1;self._error=type(exc).__name__
                    self._status('asr_unavailable_or_invalid_input',generation)
                finally:
                    chunk.samples.fill(0);self.queue.task_done()
                    with self._condition:self._pending-=1;self._condition.notify_all()
        finally:
            if adapter is not None:adapter.close()
            with self._condition:self._loaded=False;self._condition.notify_all()

    def wait_idle(self,timeout=None):
        with self._condition:
            return self._condition.wait_for(lambda:self._pending==0 and self._applied==self._generation,timeout)

    def snapshot(self):
        with self._condition:
            coverage='unknown_language' if self._language is None else 'unavailable' if self._error else 'enabled_with_possible_gaps' if self._loaded else 'awaiting_speech'
            return dict(self._stats,language=self._language,generation=self._generation,pending_chunks=self._pending,
                        model_loaded=self._loaded,coverage='stopped' if self._stop.is_set() else coverage,
                        worker_alive=self._thread is not None and self._thread.is_alive())

    def stop(self,timeout=5.):
        with self._condition:self._stop.set();self._discard_queued()
        if self._thread is threading.current_thread():return False
        if self._thread is not None:self._thread.join(timeout);return not self._thread.is_alive()
        return True
