"""Opt-in one-microphone host demo. Local model paths only; monitoring/check-ins only."""
import argparse,json,threading
from pathlib import Path
from safora.audio.stream import Yamnet
from safora.asr.offline import VoskASR
from safora.coordinator import MLCoordinator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--yamnet',required=True);p.add_argument('--english-model');p.add_argument('--hindi-model')
    p.add_argument('--speech',action='store_true',help='Explicit opt-in to optional offline ASR + unvalidated rules')
    p.add_argument('--language',choices=['en','hi','unknown'],default='unknown')
    a=p.parse_args()
    import sounddevice as sd  # Optional microphone extra; no installation/download here.
    paths={'en':a.english_model,'hi':a.hindi_model};stopped=threading.Event()
    def speech(language):
        path=paths[language]
        if not path:raise ValueError('Local selected-language ASR model missing')
        receipt=Path(path)/'.safora-model.json'
        hashes=json.loads(receipt.read_text())['file_sha256'] if receipt.is_file() else None
        return VoskASR(path,language,file_checksums=hashes)
    def cue(value):
        print(json.dumps(dict(kind=value.kind,source=value.source,session=value.session,uncertain=value.uncertain,
                              external_dispatch=False,requires_user_confirmation=True)))
    c=MLCoordinator(lambda:Yamnet(a.yamnet),speech,on_cue=cue,speech_enabled=a.speech,
                    language=None if a.language=='unknown' else a.language)
    def capture(indata,frames,timing,status):
        try:c.submit(indata[:,0],discontinuity=bool(status.input_overflow))
        except Exception:stopped.set()  # No raw audio/native error text in logs.
    try:
        c.start()
        with sd.InputStream(samplerate=16000,channels=1,dtype='int16',blocksize=1600,callback=capture):
            stopped.wait()  # Ctrl-C stops the single capture source before releasing workers.
    except KeyboardInterrupt:pass
    finally:
        clean=c.stop(timeout=30)
        print(json.dumps(dict(stopped=clean,metrics=c.snapshot())))
    return 0 if clean else 1


if __name__=='__main__':raise SystemExit(main())
