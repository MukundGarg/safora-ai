"""Research-only 250ms annotated-window experiment. Never used by live SAFORA.

No weak recording labels, no downloads, no promotion. Requires reviewed annotated
train/validation/test audio and a real local YAMNet for a matched baseline. Absence
of those inputs blocks training before any weight file is created.
"""
import argparse,json
from pathlib import Path
import wave
import numpy as np
from safora.audio.stream import waveform,Yamnet
from safora.audio.ingestion import StreamConfig
from safora.evaluation.audio_evidence import load,validate
from safora.evaluation.audio_experiment import scream_rows,enrich
from safora.evaluation.scream import evaluate
from safora.training.data import fingerprint

WINDOW=4000
HOP=2000


def target(start,end,events,negatives):
    # A positive window CONTAINS an annotated event; retain overlap separately.
    overlap=sum(max(0,min(end,b)-max(start,a)) for a,b in events)
    if overlap:return 1,overlap/(end-start)
    if any(a<=start and end<=b for a,b in negatives):return 0,0.
    return None,None  # Unknown span is not a made-up negative.


def features(x):
    x=np.asarray(x,dtype=np.float32)
    if x.shape!=(WINDOW,) or not np.isfinite(x).all():raise ValueError('250ms normalized frame required')
    spectrum=np.abs(np.fft.rfft(x*np.hanning(WINDOW)))**2
    bands=np.array([spectrum[a:b].sum() for a,b in zip(np.linspace(0,len(spectrum),9,dtype=int)[:-1],np.linspace(0,len(spectrum),9,dtype=int)[1:])])
    return np.r_[np.log1p(np.sqrt(np.mean(x*x))*1000),np.mean(x[1:]*x[:-1]<0),
                 np.log1p(np.max(np.abs(x))*1000),np.log1p(bands/(bands.sum()+1e-12)*100)].astype(np.float64)


class Probe:
    window_samples=WINDOW
    def __init__(self,mean,scale,weights):self.mean,self.scale,self.weights=mean,scale,weights
    def predict(self,x):
        f=np.r_[(features(x)-self.mean)/self.scale,1.]
        return {'screaming':float(1/(1+np.exp(-np.clip(f@self.weights,-30,30))))}


def collect(rows,base):
    data=[];labels=[]
    for row in rows:
        if row['split']!='train':continue
        with wave.open(str(base/row['path']),'rb') as f:
            x=waveform(np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').reshape(-1,f.getnchannels()),f.getframerate())
        try:
            for start in range(0,len(x)-WINDOW+1,HOP):
                label,_=target(start/16000,(start+WINDOW)/16000,row['events'],row['negative_segments'])
                if label is not None:data.append(features(x[start:start+WINDOW]));labels.append(label)
                if len(data)>100000:raise ValueError('Research frame bound exceeded')
        finally:x.fill(0)
    if not labels or set(labels)!={0,1}:raise ValueError('Strongly annotated train positives and negatives required')
    return np.stack(data),np.asarray(labels,dtype=np.float64)


def run(manifest,yamnet_path,output):
    rows=load(manifest);audit=validate(rows,manifest.parent)
    if any(not any(r['split']==s for r in rows) for s in ('train','validation','test')):raise ValueError('Independent train/validation/test required')
    if any(r['split']=='reused_external' for r in rows):raise ValueError('Reused external data excluded from this experiment')
    if not yamnet_path.is_file() or output.exists():raise ValueError('Local matched baseline/fresh output required')
    # No test/validation features enter fitting or training normalization.
    if not any(r['events'] for r in rows if r['split']=='validation') or not any(r['negative_segments'] for r in rows if r['split']=='validation'):raise ValueError('Validation positives and negative exposure required')
    x,y=collect(rows,manifest.parent);mean=x.mean(0);scale=np.maximum(x.std(0),1e-6)
    x=np.c_[(x-mean)/scale,np.ones(len(x))];weights=np.zeros(x.shape[1])
    for _ in range(200):
        scores=1/(1+np.exp(-np.clip(x@weights,-30,30)))
        weights-=.05*(x.T@(scores-y)/len(y)+.001*np.r_[weights[:-1],0.])
    model=Probe(mean,scale,weights);prepared=scream_rows(rows);candidates=[];validation_reports={}
    for threshold in (.3,.5,.7):
        config=StreamConfig(hop_samples=HOP,threshold=threshold)
        report=enrich(evaluate(prepared,model,config,split='validation',base=manifest.parent),rows)
        if not report['complete']:raise ValueError('Incomplete validation cannot select a research model')
        validation_reports[str(threshold)]=report
        candidates.append((report['event_metrics']['f1'] if report['event_metrics']['f1'] is not None else -1,threshold))
    _,threshold=max(candidates)
    # Reserve a fresh run before test inference. Partial failures cannot be silently
    # retried in this output, but external copies require research-governance review.
    output.mkdir(parents=True,exist_ok=False)
    (output/'test-consumed.json').write_text(json.dumps({'status':'reserved','manifest_sha256':fingerprint(rows)})+'\n')
    probe_report=enrich(evaluate(prepared,model,StreamConfig(hop_samples=HOP,threshold=threshold),split='test',base=manifest.parent),rows)
    with Yamnet(yamnet_path) as baseline:
        baseline_report=enrich(evaluate(prepared,baseline,StreamConfig(),split='test',base=manifest.parent),rows)
    (output/'research-weights.json').write_text(json.dumps({'kind':'EXPERIMENTAL_NOT_FOR_RUNTIME','mean':mean.tolist(),'scale':scale.tolist(),
        'weights':weights.tolist(),'threshold':threshold,'window_samples':WINDOW,'manifest_sha256':fingerprint(rows),
        'limitations':'handcrafted linear probe; not robust distress-scream model; no calibrated probability; not LR/MIL production weights'},indent=2)+'\n')
    (output/'comparison.json').write_text(json.dumps({'audit':audit,'probe':probe_report,'yamnet_fixed_baseline':baseline_report,
        'validation':validation_reports,'training_configuration':{'window_samples':WINDOW,'hop_samples':HOP,'gradient_steps':200,'step_size':.05,'l2':.001,'train_frames':len(y),'normalization_split':'train'},
        'production_promotion':False,'required_gate_status':'BLOCKED_STAKEHOLDER_HAZARD_REVIEW_AND_ANDROID_PARITY'},indent=2,allow_nan=False)+'\n')
    return {'status':'research_complete_not_promoted'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--yamnet',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-research-fit',action='store_true');a=p.parse_args()
    try:
        if not a.allow_research_fit:raise ValueError('Explicit research-only authorization required')
        report=run(a.manifest,a.yamnet,a.output);code=0
    except (OSError,ValueError,KeyError,TypeError) as e:report={'status':'BLOCKED','error_type':type(e).__name__,'accuracy_improvement':'NO_DEMONSTRATED_IMPROVEMENT'};code=2
    print(json.dumps(report));return code


if __name__=='__main__':raise SystemExit(main())
