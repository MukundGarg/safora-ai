"""Local native model orchestration smoke. Synthetic/authorized fixtures, no accuracy claim.

Paces blocks by worker-idle for reproducibility; not a live-microphone benchmark.
No text, audio or speaker information is written to the JSON report.
"""
import argparse,json,resource,time
from pathlib import Path
import numpy as np
from safora.audio.stream import Yamnet
from safora.asr.offline import VoskASR
from safora.coordinator import MLCoordinator
from safora.asr.assets import validate_prepared_model


def run(yamnet,models,fixtures):
    results=[];cues=[]
    specs=json.loads((Path(__file__).resolve().parents[1]/'models/android-assets.json').read_text())['vosk']
    def factory(language):
        root=Path(models[language]);receipt=json.loads((root/'.safora-model.json').read_text())
        validate_prepared_model(root,specs[language])
        if receipt['archive_sha256']!=specs[language]['sha256']:raise ValueError('Model provenance mismatch')
        return VoskASR(root,language,file_checksums={name:item['sha256'] for name,item in specs[language]['files'].items()})
    coordinator=MLCoordinator(lambda:Yamnet(yamnet),factory,on_cue=lambda c:cues.append(dict(kind=c.kind,source=c.source,
        uncertain=c.uncertain,external_dispatch=c.external_dispatch)),speech_enabled=True)
    wall=time.perf_counter();cpu=time.process_time()
    assert coordinator.start()
    try:
        for language in ('en','hi'):
            coordinator.configure_speech(True,language)
            raw=np.frombuffer(Path(fixtures[language]).read_bytes(),dtype='<i2')
            if len(raw)>128000:raise ValueError('Fixture exceeds eight-second smoke segment')
            audio=np.zeros(128000,np.int16);audio[:len(raw)]=raw
            began=time.perf_counter()
            before=coordinator.snapshot()['context_turns']
            for start in range(0,len(audio),1600):
                coordinator.submit(audio[start:start+1600]);assert coordinator.wait_idle(60)
            state=coordinator.snapshot()
            assert state['asr']['errors']==0 and state['asr']['results']>=1
            assert state['context_turns']>=1
            results.append(dict(language=language,fixture_samples=len(raw),padded_segment_samples=len(audio),
                wall_seconds=time.perf_counter()-began,context_cleared_before_language=before==0))
            audio.fill(0)
        state=coordinator.snapshot();assert state['acoustic']['inference_errors']==0
    finally:stopped=coordinator.stop(timeout=60)
    assert stopped
    coordinator.manual_sos()
    return dict(workload='host CPU; paced synthetic speech; not live capture',accuracy_evaluated=False,
        python_version=__import__('platform').python_version(),wall_seconds=time.perf_counter()-wall,
        process_cpu_seconds=time.process_time()-cpu,peak_process_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        languages=results,cues=cues,counters=state,stopped=stopped,
        model_bytes=dict(yamnet=Path(yamnet).stat().st_size,**{lang:sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file()) for lang,root in models.items()}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('yamnet','english-model','hindi-model','english-pcm16','hindi-pcm16','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();report=run(a.yamnet,dict(en=a.english_model,hi=a.hindi_model),dict(en=a.english_pcm16,hi=a.hindi_pcm16))
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Local native coordinator smoke passed; no transcript text or accuracy exported.')


if __name__=='__main__':main()
