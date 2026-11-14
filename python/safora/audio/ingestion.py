"""Acoustic-only ingestion. No ASR, conversation model, incident manager or dispatch."""
from dataclasses import dataclass
import json
from math import ceil, gcd
from pathlib import Path
import queue
import threading
import time
import numpy as np
from scipy.signal import firwin, upfirdn
from .stream import SAMPLE_RATE, AcousticEvent, TemporalDetector, WindowStream, waveform


@dataclass(frozen=True)
class StreamConfig:
    version: int = 1
    source_rate: int = 16000
    hop_samples: int = 8000
    threshold: float = .30
    confirmation_hits: int = 2
    confirmation_windows: int = 3
    cooldown_seconds: float = 10
    block_seconds: float = .1
    queue_capacity: int = 4
    max_channels: int = 2
    clip_input: bool = False

    def __post_init__(self):
        for value in (self.version, self.source_rate, self.hop_samples, self.queue_capacity, self.max_channels):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("Expected integer stream configuration")
        if self.version != 1 or not 1 <= self.queue_capacity <= 8 or not 1 <= self.max_channels <= 8:
            raise ValueError("Unsupported version or queue/channel bounds")
        if not np.isfinite(self.block_seconds) or not 0 < self.block_seconds <= 1 or self.hop_samples <= 0:
            raise ValueError("Invalid block/hop duration")
        if not isinstance(self.clip_input, bool):
            raise ValueError("clip_input must be boolean")
        StreamingResampler(self.source_rate)  # Validate bounded rate conversion, without model inference.
        TemporalDetector(self.threshold, self.confirmation_hits, self.confirmation_windows, self.cooldown_seconds)

    @property
    def max_block_samples(self):
        return max(1, int(self.source_rate * self.block_seconds))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class StreamingResampler:
    """Causal polyphase FIR, bounded history, phase invariant to callback chunking.

    No EOF ringing tail is synthesized. Group delay is reported explicitly; unlike
    offline resample_poly, live filtering cannot use future samples for zero phase.
    Unusual coprime rate ratios are rejected to bound filter memory and CPU.
    """
    def __init__(self, source_rate):
        if isinstance(source_rate, bool) or not isinstance(source_rate, int) or not 0 < source_rate <= 192000:
            raise ValueError("Invalid source sample rate")
        divisor = gcd(source_rate, SAMPLE_RATE)
        self.up, self.down = SAMPLE_RATE // divisor, source_rate // divisor
        if max(self.up, self.down) > 2048:
            raise ValueError("Unsupported streaming rate ratio; use a standard audio sample rate")
        self.rate = source_rate
        factor = max(self.up, self.down)
        self.h = (firwin(20 * factor + 1, 1 / factor).astype(np.float32) * self.up
                  if factor > 1 else np.ones(1, np.float32))
        self.delay_seconds = (len(self.h) - 1) / (2 * self.up * source_rate)
        self.clear()

    def clear(self):
        self.history = np.empty(0, np.float32)
        self.base = self.received = self.emitted = 0

    def feed(self, mono):
        x = np.asarray(mono, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("Resampler expects normalized mono")
        if not len(x):
            return x.copy()
        if self.up == self.down == 1:
            self.received += len(x)
            self.emitted += len(x)
            return x.copy()
        combined = np.concatenate((self.history, x))
        end = self.received + len(x)
        output_end = (end * self.up + self.down - 1) // self.down
        filtered = upfirdn(self.h, combined, self.up, self.down)
        offset = self.emitted - self.base * self.up // self.down
        out = filtered[offset:offset + output_end - self.emitted].copy()
        # Keep enough causal input, with an origin divisible by down for exact phase.
        memory = ceil((len(self.h) - 1) / self.up)
        next_base = max(0, end - memory) // self.down * self.down
        self.history = combined[next_base - self.base:].copy()
        self.base, self.received, self.emitted = next_base, end, output_end
        return np.clip(out, -1, 1)


@dataclass
class Observation:
    start: float
    end: float
    scores: dict
    event: AcousticEvent | None
    inference_seconds: float
    capture_to_decision_seconds: float | None
    padded: bool = False


class ScreamProcessor:
    """Synchronous single-owner core shared by the live worker and file evaluator."""
    def __init__(self, model, config=None):
        self.model, self.config = model, config or StreamConfig()
        self.windows = WindowStream(model.window_samples, self.config.hop_samples)
        self.detector = TemporalDetector(self.config.threshold, self.config.confirmation_hits,
                                         self.config.confirmation_windows, self.config.cooldown_seconds)
        self.resampler = StreamingResampler(self.config.source_rate)
        self.origin = 0.
        self.next_source = self.gaps = 0
        self.finished = False

    def feed(self, samples, *, start_sample=None, capture_time=None, discontinuity=False):
        if self.finished:
            raise RuntimeError("EOF already processed")
        start = self.next_source if start_sample is None else start_sample
        if not isinstance(start, int) or start < self.next_source:
            raise ValueError("Overlapping or backwards source sample clock")
        if capture_time is not None and (not np.isfinite(capture_time) or capture_time < 0):
            raise ValueError("capture_time must use the monotonic clock")
        raw = np.asarray(samples)
        if raw.ndim not in (1, 2) or (raw.ndim == 2 and not 1 <= raw.shape[1] <= self.config.max_channels):
            raise ValueError("Unexpected source channels/shape")
        if len(raw) > self.config.max_block_samples:
            raise ValueError("Block exceeds configured capture bound")
        x = waveform(raw, clip=self.config.clip_input)
        if start != self.next_source or discontinuity:
            self.windows.clear()
            self.resampler.clear()
            self.detector.reset_continuity()
            self.origin = start / self.config.source_rate
            self.gaps += 1
        self.next_source = start + len(x)
        converted = self.resampler.feed(x)
        observations = []
        for t, window in self.windows.feed(converted):
            end = self.origin + t + self.model.window_samples / SAMPLE_RATE
            capture_end = None if capture_time is None else capture_time + end - start / self.config.source_rate
            observations.append(self._observe(self.origin + t, end, window, capture_end))
        return observations

    def _observe(self, start, end, window, capture_end=None, padded=False):
        began = time.perf_counter()
        try:
            scores = self.model.predict(window)
        finally:
            window.fill(0)
        duration = time.perf_counter() - began
        event = self.detector.update(end, scores['screaming'])
        latency = None if capture_end is None else max(0., time.monotonic() - capture_end)
        if event is not None:
            event.window_start = start
            event.inference_seconds = duration
            event.capture_to_decision_seconds = latency
        return Observation(start, end, scores, event, duration, latency, padded)

    def finish(self, pad_tail=False):
        if self.finished:
            return []
        self.finished = True
        end = self.next_source / self.config.source_rate
        tail = self.windows.tail() if pad_tail else None
        self.windows.clear()
        self.resampler.clear()
        if tail is None:
            return []
        start, window = tail
        return [self._observe(self.origin + start, end, window, padded=True)]


@dataclass
class _Block:
    samples: np.ndarray
    start: int
    capture_time: float
    discontinuity: bool


class RealtimeScreamMonitor:
    """Thread-safe drop-newest ingestion; one dedicated inference worker, no ASR locks.

    submit copies only a bounded PCM block, and never invokes user callbacks/model/
    resampling. Callbacks run on the worker; slow receivers cause explicit queue
    drops, never capture waits. Native inference is not interruptible: stop returns
    False on timeout and suppresses late callbacks. Close the model only after True.
    """
    def __init__(self, model, on_event=lambda event: None, on_status=lambda status: None, config=None):
        self.processor = ScreamProcessor(model, config)
        self.config = self.processor.config
        self.on_event, self.on_status = on_event, on_status
        self.queue = queue.Queue(maxsize=self.config.queue_capacity)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._next_sample = self._pending = 0
        self._stats = dict(accepted_blocks=0, processed_blocks=0, dropped_blocks=0,
                           dropped_source_samples=0, gaps=0, windows=0, inference_errors=0,
                           callback_errors=0, discarded_on_stop=0, queue_high_water=0,
                           max_inference_seconds=0., max_capture_to_decision_seconds=0.)

    def submit(self, samples, *, start_sample=None, sample_rate=None, capture_time=None, discontinuity=False):
        if sample_rate is not None and sample_rate != self.config.source_rate:
            raise ValueError("Source rate changed; stop and create a new monitor")
        x = np.asarray(samples)
        if (x.ndim not in (1, 2) or not 0 < len(x) <= self.config.max_block_samples
                or (x.ndim == 2 and not 1 <= x.shape[1] <= self.config.max_channels)
                or not ((x.dtype.kind == 'i' and x.dtype.itemsize == 2) or np.issubdtype(x.dtype, np.floating))):
            raise ValueError("Expected bounded PCM16/float capture block with configured channels")
        stamp = time.monotonic() - len(x) / self.config.source_rate if capture_time is None else capture_time
        if not np.isfinite(stamp) or stamp < 0:
            raise ValueError("Invalid capture timestamp")
        copied = x.copy()
        with self._condition:
            if self._stop.is_set():
                copied.fill(0)
                return False
            start = self._next_sample if start_sample is None else start_sample
            if not isinstance(start, int) or start < self._next_sample:
                raise ValueError("Overlapping or backwards capture clock")
            self._next_sample = start + len(x)  # Includes dropped blocks: no compressed timeline.
            try:
                self.queue.put_nowait(_Block(copied, start, stamp, discontinuity))
            except queue.Full:
                copied.fill(0)
                self._stats['dropped_blocks'] += 1
                self._stats['dropped_source_samples'] += len(x)
                return False
            self._pending += 1
            self._stats['accepted_blocks'] += 1
            self._stats['queue_high_water'] = max(self._stats['queue_high_water'], self.queue.qsize())
            return True

    def start(self):
        with self._condition:
            if self._thread is not None or self._stop.is_set():
                raise RuntimeError("Monitor instances start only once")
            self._thread = threading.Thread(target=self._run, name='safora-acoustic', daemon=True)
            self._thread.start()
        return self

    def _callback(self, callback, value):
        if self._stop.is_set():
            return
        try:
            callback(value)
        except Exception:
            with self._condition:
                self._stats['callback_errors'] += 1

    def _done(self):
        with self._condition:
            self._pending -= 1
            self._condition.notify_all()

    def _run(self):
        force_gap = False
        try:
            while not self._stop.is_set():
                try:
                    block = self.queue.get(timeout=.05)
                except queue.Empty:
                    continue
                try:
                    if self._stop.is_set():
                        continue
                    before = self.processor.gaps
                    observations = self.processor.feed(block.samples, start_sample=block.start,
                        capture_time=block.capture_time, discontinuity=block.discontinuity or force_gap)
                    force_gap = False
                    with self._condition:
                        self._stats['processed_blocks'] += 1
                        self._stats['gaps'] = self.processor.gaps
                        self._stats['windows'] += len(observations)
                        for observation in observations:
                            self._stats['max_inference_seconds'] = max(self._stats['max_inference_seconds'], observation.inference_seconds)
                            self._stats['max_capture_to_decision_seconds'] = max(self._stats['max_capture_to_decision_seconds'], observation.capture_to_decision_seconds or 0.)
                    if self.processor.gaps != before:
                        self._callback(self.on_status, {'kind': 'audio_gap', 'source_sample': block.start})
                    for observation in observations:
                        if observation.event is not None:
                            self._callback(self.on_event, observation.event)
                except Exception as exc:
                    force_gap = True
                    with self._condition:
                        self._stats['inference_errors'] += 1
                    self._callback(self.on_status, {'kind': 'acoustic_unavailable', 'error': type(exc).__name__})
                finally:
                    block.samples.fill(0)
                    self.queue.task_done()
                    self._done()
        finally:
            self.processor.finish()

    def wait_idle(self, timeout=None):
        with self._condition:
            return self._condition.wait_for(lambda: self._pending == 0, timeout)

    def snapshot(self):
        with self._condition:
            return dict(self._stats, pending_blocks=self._pending,
                        worker_alive=self._thread is not None and self._thread.is_alive())

    def stop(self, timeout=5.):
        with self._condition:
            self._stop.set()
        while True:
            try:
                block = self.queue.get_nowait()
            except queue.Empty:
                break
            block.samples.fill(0)
            self.queue.task_done()
            with self._condition:
                self._stats['discarded_on_stop'] += 1
            self._done()
        if self._thread is threading.current_thread():
            return False
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        self.processor.finish()
        return True
