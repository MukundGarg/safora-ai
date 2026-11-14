"""Real local-model smoke report. No raw audio, transcript text, WER or CER is saved."""
import argparse,json,time
from pathlib import Path
import numpy as np
from safora.asr.offline import VoskASR
from safora.asr.assets import validate_model_dir


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--english-model',required=True,type=Path);p.add_argument('--hindi-model',required=True,type=Path)
    p.add_argument('--english-pcm16',type=Path);p.add_argument('--hindi-pcm16',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();results=[]
    for language,path,fixture in [('en',a.english_model,a.english_pcm16),('hi',a.hindi_model,a.hindi_pcm16)]:
        receipt=json.loads((path/'.safora-model.json').read_text())
        validate_model_dir(path,receipt['file_sha256'])
        began=time.perf_counter()
        with VoskASR(path,language,file_checksums=receipt['file_sha256']) as model:
            load_seconds=time.perf_counter()-began
            audio=np.zeros(1600,np.float32) if fixture is None else np.frombuffer(fixture.read_bytes(),dtype='<i2').astype(np.float32)/32768
            try:
                began=time.perf_counter();out=model.transcribe(audio);elapsed=time.perf_counter()-began
                results.append(dict(language=language,archive_sha256=receipt['archive_sha256'],
                    workload='authored synthetic speech' if fixture else 'synthetic silence',
                    samples=len(audio),load_seconds=load_seconds,decode_seconds=elapsed,
                    transcript_characters=len(out.text),nonempty_transcript=bool(out.text),uncertain=out.uncertain,
                    reason=out.reason,final=out.is_final))
            finally:audio.fill(0)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(host_cpu_only=True,accuracy_evaluated=False,models=results),indent=2)+'\n')
    print('Local English/Hindi model smoke completed; transcript text and accuracy metrics are not exported.')


if __name__=='__main__':main()
