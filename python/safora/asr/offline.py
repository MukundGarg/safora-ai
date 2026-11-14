from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Transcript:
    text: str
    uncertain: bool
    language: str | None
    # Whisper log scores are not calibrated danger probabilities.
    average_log_score: float

class WhisperASR:
    """Local CTranslate2 model directory only; never downloads at runtime."""
    def __init__(self,path,threads=2):
        if not Path(path).is_dir(): raise FileNotFoundError(path)
        from faster_whisper import WhisperModel
        self.model=WhisperModel(str(path),device="cpu",compute_type="int8",cpu_threads=threads,num_workers=1,local_files_only=True)
    def transcribe(self,audio):
        if len(audio)>16000*12: raise ValueError("Chunk exceeds 12 seconds")
        segments,info=self.model.transcribe(audio,beam_size=1,condition_on_previous_text=False,vad_filter=False)
        segments=list(segments)
        score=float(np.mean([s.avg_logprob for s in segments])) if segments else -10.
        text=" ".join(s.text.strip() for s in segments if s.no_speech_prob<.6)
        return Transcript(text,score<-.8 or not text,info.language,score)

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

class VoskASR:
    """Smaller practical alternative; installed model determines language coverage."""
    def __init__(self,path):
        if not Path(path).is_dir():raise FileNotFoundError(path)
        from vosk import Model
        self.model=Model(str(path))
    def transcribe(self,audio):
        import json
        from vosk import KaldiRecognizer
        if len(audio)>16000*12:raise ValueError('Chunk exceeds 12 seconds')
        recognizer=KaldiRecognizer(self.model,16000);recognizer.SetWords(True)
        pcm=(np.clip(audio,-1,1)*32767).astype('<i2')
        try:
            recognizer.AcceptWaveform(pcm.tobytes());result=json.loads(recognizer.FinalResult())
            words=result.get('result',[])
            score=float(np.mean([w.get('conf',0.) for w in words])) if words else 0.
            return Transcript(result.get('text',''),score<.75,None,float(np.log(max(score,1e-9))))
        finally:pcm.fill(0)
