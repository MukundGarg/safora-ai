from dataclasses import dataclass, field
import json
import math
from .assets import validate_model_dir
from pathlib import Path
import numpy as np

@dataclass
class Transcript:
    text: str = field(repr=False)
    uncertain: bool
    language: str | None
    # Whisper log scores are not calibrated danger probabilities.
    average_log_score: float
    is_final: bool = True
    reason: str | None = None
    confidence_kind: str = "backend_specific_uncalibrated"

class WhisperASR:
    """Local CTranslate2 model directory only; never downloads at runtime."""
    def __init__(self,path,threads=2):
        if not Path(path).is_dir(): raise FileNotFoundError(path)
        for name in ('model.bin','config.json','tokenizer.json','vocabulary.txt'):
            file=Path(path)/name
            if not file.is_file() or file.stat().st_size==0:
                raise ValueError('Incomplete local Whisper assets; refusing tokenizer download fallback')
        from faster_whisper import WhisperModel
        self.model=WhisperModel(str(path),device="cpu",compute_type="int8",cpu_threads=threads,num_workers=1,local_files_only=True)
    def transcribe(self,audio):
        if self.model is None: raise RuntimeError("ASR is closed")
        audio=normalized_mono(audio)
        if not len(audio): return Transcript('',True,None,-10.,reason='empty_audio')
        segments,info=self.model.transcribe(audio,beam_size=1,condition_on_previous_text=False,vad_filter=False)
        segments=list(segments)
        score=float(np.mean([s.avg_logprob for s in segments])) if segments else -10.
        text=" ".join(s.text.strip() for s in segments if s.no_speech_prob<.6)
        return Transcript(text,score<-.8 or not text,info.language,score)

    def close(self): self.model=None

class EnergyVAD:
    """Unvalidated energy fallback; quiet speech coverage requires calibration or Silero."""
    def __init__(self,threshold=.008): self.threshold=threshold
    def speech(self,audio): return bool(len(audio) and np.sqrt(np.mean(np.square(audio)))>=self.threshold)

class SileroVAD:
    """Optional local TorchScript Silero model. Caller supplies 512 samples at 16 kHz."""
    def __init__(self,path):
        import torch
        self.torch=torch;self.model=torch.jit.load(str(path),map_location="cpu").eval()
    def speech(self,audio):
        if len(audio)!=512: raise ValueError("Silero requires 512-sample frames")
        with self.torch.inference_mode(): return float(self.model(self.torch.from_numpy(audio),16000))>=.5
    def reset(self): self.model.reset_states()


def normalized_mono(audio, max_samples=16000*12):
    x = np.asarray(audio)
    if x.ndim != 1 or len(x) > max_samples or not np.issubdtype(x.dtype, np.floating):
        raise ValueError('Expected bounded normalized mono floats at 16000 Hz')
    if not np.isfinite(x).all() or (len(x) and np.max(np.abs(x)) > 1):
        raise ValueError('Audio must be finite and within [-1,1]')
    return np.ascontiguousarray(x, dtype=np.float32)


def pcm16le(audio, max_samples=16000*12):
    """Round-to-nearest/even and saturate signed PCM endpoints; no gain normalization."""
    x = normalized_mono(audio, max_samples)
    return np.clip(np.rint(x.astype(np.float64)*32768), -32768, 32767).astype('<i2')


class VoskASR:
    """Single-owner, explicit local Vosk Small model. No downloads or language guessing.

    accept() returns provisional partial/native-endpoint deltas. finish() returns
    ONE aggregate of committed endpoint results + final tail for the current bounded
    segment/model. Do not concatenate those two API levels yourself.
    """
    def __init__(self, path, language=None, *, confidence_threshold=.75, max_seconds=12,
                 file_checksums=None, model_factory=None, recognizer_factory=None):
        if language not in (None, 'en', 'hi'):
            raise ValueError('Select en/hi explicitly; None means unknown, not automatic identification')
        if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError('Invalid confidence threshold')
        if not math.isfinite(max_seconds) or not 0 < max_seconds <= 12:
            raise ValueError('Invalid segment bound')
        directory = validate_model_dir(path, file_checksums)
        if model_factory is None or recognizer_factory is None:
            from vosk import Model, KaldiRecognizer, SetLogLevel
            SetLogLevel(-1)  # Native debug logging is unnecessary for private live recognition.
            model_factory = model_factory or (lambda p: Model(model_path=str(p)))
            recognizer_factory = recognizer_factory or KaldiRecognizer
        self.model = model_factory(directory)
        self.language, self.threshold = language, confidence_threshold
        self.max_samples = int(max_seconds*16000)
        self.recognizer_factory = recognizer_factory
        self._recognizer = None
        self.closed = False
        self.reset()

    @staticmethod
    def _release(resource):
        # Python Vosk's native handles are released by object destruction, not a close()
        # API in 0.3.45. Honor close() on injected/newer backends when provided.
        close = getattr(resource, 'close', None)
        if callable(close): close()

    def _check_open(self):
        if self.closed: raise RuntimeError('ASR is closed')

    def reset(self):
        self._check_open()
        if self._recognizer is not None: self._release(self._recognizer)
        self._recognizer = None
        self._parts = []
        self._samples = 0
        self._confidence_sum = self._confidence_count = 0
        self._missing_confidence = self._corrupt = False

    def begin(self):
        self.reset()
        self._recognizer = self.recognizer_factory(self.model, 16000)
        self._recognizer.SetWords(True)

    def _parse(self, raw, partial=False):
        try:
            if not isinstance(raw, str) or len(raw) > 65536: raise ValueError()
            result = json.loads(raw)
            key = 'partial' if partial else 'text'
            if not isinstance(result, dict) or not isinstance(result.get(key), str): raise ValueError()
            text = result[key].strip()
            if len(text) > 4096: raise ValueError()
            words = result.get('result', [])
            if not isinstance(words, list) or len(words) > 2048: raise ValueError()
            confidence = []
            for word in words:
                if not isinstance(word, dict): raise ValueError()
                value = word.get('conf')
                if value is None: continue
                if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError()
                confidence.append(float(value))
            complete = bool(words) and len(confidence) == len(words)
            return text, confidence, complete
        except (ValueError, TypeError, RecursionError, json.JSONDecodeError):
            self._corrupt = True
            return '', [], False

    def _commit(self, parsed):
        text, confidence, complete = parsed
        if text:
            if sum(map(len, self._parts)) + len(text) > 4096:
                self._corrupt = True
                return
            self._parts.append(text)
            self._confidence_sum += sum(confidence)
            self._confidence_count += len(confidence)
            self._missing_confidence |= not complete

    def _result(self, text, *, final=True, partial=False):
        score = self._confidence_sum/self._confidence_count if self._confidence_count and not self._missing_confidence else None
        reason = ('malformed_native_result' if self._corrupt else 'empty_or_silence' if not text else
                  'partial' if partial else 'unknown_language' if self.language is None else
                  'confidence_unavailable' if score is None else 'low_confidence' if score < self.threshold else None)
        if self._corrupt: text = ''
        return Transcript(text, reason is not None, self.language,
                          math.log(max(score,1e-9)) if score is not None else -10., final, reason,
                          'vosk_word_confidence_not_whisper_logprob')

    def accept(self, audio):
        self._check_open()
        pcm = pcm16le(audio, self.max_samples)
        if self._samples + len(pcm) > self.max_samples:
            pcm.fill(0)
            raise ValueError('Segment limit exceeded; finish/reset before more samples')
        if not len(pcm): return self._result('', final=False, partial=True)
        if self._recognizer is None: self.begin()
        self._samples += len(pcm)
        try:
            endpoint = self._recognizer.AcceptWaveform(pcm.tobytes())
            parsed = self._parse(self._recognizer.Result() if endpoint else self._recognizer.PartialResult(), partial=not endpoint)
            if endpoint: self._commit(parsed)
            return self._result(parsed[0], final=bool(endpoint), partial=not endpoint)
        finally: pcm.fill(0)

    def finish(self):
        self._check_open()
        if self._recognizer is not None:
            try: self._commit(self._parse(self._recognizer.FinalResult()))
            finally:
                self._release(self._recognizer)
                self._recognizer = None
        result = self._result(' '.join(self._parts))
        self._parts = []
        self._samples = self._confidence_sum = self._confidence_count = 0
        self._missing_confidence = self._corrupt = False
        return result

    def transcribe(self, audio):
        self._check_open()
        x = normalized_mono(audio, self.max_samples)
        self.reset()
        try:
            for start in range(0,len(x),1600): self.accept(x[start:start+1600])
            return self.finish()
        except Exception:
            self.reset()
            raise

    def close(self):
        if self.closed: return
        self.reset()
        self._release(self.model)
        self.model = None
        self.closed = True

    def __enter__(self): return self
    def __exit__(self,*_): self.close()
