"""Run: python -m safora.evaluation.calibrate validation_scores.json output.json"""
import json,sys
from safora.audio.stream import TemporalDetector
from safora.evaluation.metrics import event_metrics,prf

def sweep(records):
    if any(r.get('reused', False) or r.get('benchmark_status') in {'reused', 'reused_not_independent'} or r.get('split') == 'reused_external' for r in records):
        raise ValueError('Reused evaluation data must not be used for calibration')
    if any(r['split']!='validation' for r in records):raise ValueError('Calibration uses validation only')
    output=[]
    for threshold in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
        tp=fp=fn=0;pred=[];truth=[];offset=0;negative_hours=0;negative_fp=0;rtp=rfp=rfn=0;missed=[];delays=[]
        for r in records:
            detector=TemporalDetector(threshold=threshold)
            events=[t for t,s in r['scores'] if detector.update(t,s)]
            metrics=event_metrics(events,r['events'],r['duration']/3600 if not r['events'] else 0)
            tp+=metrics['tp'];fp+=metrics['fp'];fn+=metrics['fn'];delays+=metrics['delays_seconds']
            if not r['events']:
                negative_hours+=r['duration']/3600;negative_fp+=len(events)
            missed.extend({'recording':r['id'],'event_index':i} for i in metrics['missed_events'])
            actual=bool(r['events']);positive=bool(events)
            rtp+=actual and positive;rfp+=not actual and positive;rfn+=actual and not positive
        output.append({'threshold':threshold,**prf(tp,fp,fn),'recording':prf(rtp,rfp,rfn),
                       'false_alerts_per_negative_hour':negative_fp/negative_hours if negative_hours else None,
                       'delays_seconds':delays,'false_negatives':missed})
    return output
if __name__=='__main__':
    with open(sys.argv[1]) as f:records=json.load(f)
    with open(sys.argv[2],'w') as f:json.dump(sweep(records),f,indent=2)
