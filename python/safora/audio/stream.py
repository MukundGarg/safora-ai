"""YAMNet scores and sample-clock framing; no risk fusion or emergency dispatch."""
from collections import deque
from dataclasses import dataclass
from math import gcd
from numbers import Real
import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 16000  # Published YAMNet waveform contract; tensor shapes do not encode Hz.
CLASSES = {"screaming": 11, "shout": 6, "yell": 9, "crying": 19, "whimper": 21, "wail": 22}


def waveform(samples, rate=SAMPLE_RATE, *, clip=False):
    """PCM16 or normalized real floats; 2-D means [frames, channels], never channels-first.

    Offline resampling is zero-phase. For successive live blocks use StreamingResampler,
    whose causal filter preserves phase/history instead of resampling each block afresh.
    Out-of-range input is rejected unless saturation was explicitly requested.
    """
    if isinstance(rate, bool) or not isinstance(rate, (int, np.integer)) or not 0 < rate <= 192000:
        raise ValueError("Invalid integer sample rate")
    x = np.asarray(samples)
    if x.ndim not in (1, 2) or (x.ndim == 2 and not 1 <= x.shape[1] <= 8):
        raise ValueError("Expected [frames] or [frames, 1..8 channels]")
    if x.dtype.kind == 'i' and x.dtype.itemsize == 2:
        x = x.astype(np.float32) / 32768
    elif np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32, copy=False)
    else:
        raise ValueError("Only PCM16 integers or real floats are supported")
    if not np.isfinite(x).all():
        raise ValueError("Audio must be finite")
    if x.size and np.max(np.abs(x)) > 1:
        if not clip:
            raise ValueError("Audio outside [-1,1]; set clip=True for explicit saturation")
        x = np.clip(x, -1, 1)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if rate != SAMPLE_RATE and len(x):
        d = gcd(rate, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // d, rate // d)
        x = np.clip(x, -1, 1)  # FIR ringing can overshoot valid full-scale input.
    return np.ascontiguousarray(x, dtype=np.float32)


class WindowStream:
    """Bounded contiguous 16 kHz framing. Timestamps are window starts, in seconds.

    clear(start_sample) preserves an absolute normalized sample clock after a gap.
    Only EOF callers may request a single zero-padded tail; live feeds never pad.
    """
    def __init__(self, window=15600, hop=8000):
        if not isinstance(window, int) or not isinstance(hop, int) or not 0 < hop <= window:
            raise ValueError("Invalid window/hop")
        self.window, self.hop = window, hop
        self.buffer = np.zeros(window, dtype=np.float32)
        self.clear()

    def feed(self, samples):
        x = waveform(samples)
        pos = 0
        while pos < len(x):
            n = min(self.window - self.size, len(x) - pos)
            self.buffer[self.size:self.size + n] = x[pos:pos + n]
            self.size += n
            self.total += n
            pos += n
            if self.size == self.window:
                self.last_end = self.start + self.window
                result = self.buffer.copy()
                timestamp = self.start / SAMPLE_RATE
                keep = self.window - self.hop
                self.buffer[:keep] = self.buffer[self.hop:]
                self.size = keep
                self.start += self.hop
                yield timestamp, result

    def tail(self):
        """Return one padded EOF window, only if fresh samples remain; then clear."""
        result = None
        if self.size and self.total > self.last_end:
            x = np.zeros(self.window, dtype=np.float32)
            x[:self.size] = self.buffer[:self.size]
            result = (self.start / SAMPLE_RATE, x)
        self.clear()
        return result

    def clear(self, start_sample=0):
        if not isinstance(start_sample, int) or start_sample < 0:
            raise ValueError("Invalid sample clock")
        self.buffer.fill(0)
        self.size = 0
        self.start = self.total = self.last_end = start_sample


class Yamnet:
    """Single-owner CPU interpreter for classification YAMNet, not experimental LR/MIL.

    Derive N from the allocated fixed [N] or [1,N] input. Accept only one float32
    classification output [521] or [1,521]; reject embeddings/multi-output variants.
    Scores are independent model class scores, not probabilities of personal danger.
    """
    sample_rate = SAMPLE_RATE

    def __init__(self, path, threads=2, interpreter=None):
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            raise ValueError("threads must be positive")
        if interpreter is None:
            from ai_edge_litert.interpreter import Interpreter
            interpreter = Interpreter(model_path=str(path), num_threads=threads)
        self.model = interpreter
        self.model.allocate_tensors()
        inputs, outputs = self.model.get_input_details(), self.model.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one waveform input and one classification output")
        self.i, self.o = inputs[0], outputs[0]
        self.input_shape = tuple(int(n) for n in self.i['shape'])
        self.output_shape = tuple(int(n) for n in self.o['shape'])
        if (self.i['dtype'] != np.float32 or len(self.input_shape) not in (1, 2)
                or (len(self.input_shape) == 2 and self.input_shape[0] != 1)
                or self.input_shape[-1] <= 0):
            raise ValueError("Expected fixed float32[N] or float32[1,N] waveform")
        if any(n < 0 for n in self.i.get('shape_signature', self.input_shape)):
            raise ValueError("Dynamic waveform signatures must be resolved to a fixed model")
        if self.o['dtype'] != np.float32 or self.output_shape not in ((521,), (1, 521)):
            raise ValueError("Expected float32[521] or float32[1,521] classification scores")
        self.window_samples = self.input_shape[-1]
        self.window_seconds = self.window_samples / self.sample_rate

    def predict(self, x):
        if self.model is None:
            raise RuntimeError("YAMNet is closed")
        x = waveform(x)
        if x.shape != (self.window_samples,):
            raise ValueError(f"Expected {self.window_samples} mono samples")
        self.model.set_tensor(self.i['index'], x.reshape(self.input_shape))
        self.model.invoke()
        output = self.model.get_tensor(self.o['index'])
        if output.shape != self.output_shape or output.dtype != np.float32:
            raise ValueError("Runtime classification tensor differs from metadata")
        y = output.reshape(-1)
        if not np.isfinite(y).all() or np.any(y < 0) or np.any(y > 1):
            raise ValueError("Invalid classification scores")
        return {label: float(y[index]) for label, index in CLASSES.items()}

    def close(self):
        model, self.model = self.model, None
        if model is not None and callable(getattr(model, 'close', None)):
            model.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass
class AcousticEvent:
    time: float
    score: float
    reason: str = "temporally_confirmed_scream"
    monitoring_only: bool = True
    window_start: float | None = None
    inference_seconds: float | None = None
    capture_to_decision_seconds: float | None = None


class TemporalDetector:
    def __init__(self, threshold=.30, hits=2, windows=3, cooldown=10):
        if (not isinstance(threshold, Real) or not isinstance(cooldown, Real)
                or isinstance(threshold, bool) or isinstance(cooldown, bool)
                or not np.isfinite([threshold, cooldown]).all() or not 0 <= threshold <= 1
                or isinstance(hits, bool) or isinstance(windows, bool)
                or not isinstance(hits, int) or not isinstance(windows, int)
                or not 1 <= hits <= windows or cooldown < 0):
            raise ValueError("Invalid detector config")
        self.threshold, self.hits, self.cooldown = threshold, hits, cooldown
        self.history = deque(maxlen=windows)
        self.last = self.clock = -float('inf')

    def reset_continuity(self):
        """Forget votes after missing audio, but preserve the cooldown/sample clock."""
        self.history.clear()

    def update(self, time, score):
        if time < self.clock or not np.isfinite([time, score]).all() or not 0 <= score <= 1:
            raise ValueError("Invalid time/score")
        self.clock = time
        self.history.append(score >= self.threshold)
        if sum(self.history) >= self.hits and time - self.last >= self.cooldown:
            self.last = time
            self.history.clear()
            return AcousticEvent(time, score)
