"""Strict annotated evidence -> frozen baseline/validation rule study/test-once.

No production-model promotion, no model download, no learned accuracy fabricated.
"""
import argparse,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .audio_evidence import load,validate
from .scream import evaluate
from .metrics import event_metrics
from .asr import evaluate as evaluate_asr
from safora.audio.ingestion import StreamConfig
from safora.audio.stream import Yamnet
from safora.training.data import fingerprint


def wilson(successes,total):
    if not total:return None
    z=1.959963984540054;p=successes/total;den=1+z*z/total
    middle=(p+z*z/(2*total))/den;half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,float(middle-half)),min(1.,float(middle+half))]


def scream_rows(rows):
    # Old evaluator clip negatives must mean ENTIRE recording reviewed negative.
    # Partially annotated files retain event labels but are not false-alarm-hour
    # denominators unless fully annotated. Do not infer labels for unknown spans.
    result=[]
    for row in rows:
        full_negative=row['negative_segments']==[[0,row['duration_seconds']]] and not row['events']
        result.append({**row,'label':True if row['events'] else False if full_negative else None})
    return result


def enrich(report,rows):
    counts=report['clip_metrics'];tn,fp=counts['confusion_matrix'][0];fn,tp=counts['confusion_matrix'][1]
    report['uncertainty_intervals']={'clip_precision_wilson_95':wilson(tp,tp+fp),'clip_recall_wilson_95':wilson(tp,tp+fn),
        'assumption':'independent recordings; clustered source correlation is not corrected',
        'event_f1_95':None,'delay_95':None,'cluster_adjusted_95':None}
    by_id={r['id']:r for r in rows}
    report['misses_by_condition']={condition:sum(not c['detected'] and c['label'] is True and by_id[c['id']]['condition']==condition for c in report['clips'])
        for condition in sorted({r['condition'] for r in rows})}
    # Exclude unannotated time from event scoring, and use explicitly reviewed
    # negative segments for exposure even inside positive recordings.
    totals=dict(tp=0,fp=0,fn=0);delays=[];negative_seconds=0.;analyzed_negative_seconds=0.;negative_alerts=0;ignored=0
    for clip in report['clips']:
        row=by_id[clip['id']];truth=row['events'];negatives=row['negative_segments']
        valid=[]
        for event in clip['events']:
            t=event['time']
            if any(a<=t<=b+report['event_tolerance_seconds'] for a,b in truth) or any(a<=t<=b for a,b in negatives):valid.append(t)
            else:ignored+=1
        remaining=set(range(len(truth)))
        for t in sorted(valid):
            matches=[i for i in remaining if truth[i][0]<=t<=truth[i][1]+report['event_tolerance_seconds']]
            if matches:remaining.remove(min(matches,key=lambda i:truth[i][0]))
            elif any(a<=t<=b for a,b in negatives):negative_alerts+=1
        one=event_metrics(valid,truth,0,report['event_tolerance_seconds'])
        for key in totals:totals[key]+=one[key]
        delays.extend(one['delays_seconds']);negative_seconds+=sum(b-a for a,b in negatives)
        analyzed_negative_seconds+=sum(max(0,min(b,clip['analyzed_audio_seconds'])-a) for a,b in negatives)
    tp,fp,fn=(totals[k] for k in ('tp','fp','fn'))
    report['event_metrics']={**totals,'precision':tp/(tp+fp) if tp+fp else None,
        'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if tp+fn else None,
        'detection_delay_seconds':delays,'unannotated_detections_unscored':ignored}
    report['clip_metrics']['precision']=counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None
    report['clip_metrics']['recall']=counts['tp']/(counts['tp']+counts['fn']) if counts['tp']+counts['fn'] else None
    if counts['tp']+counts['fn']==0:report['clip_metrics']['f1']=None
    from scipy.stats import chi2
    hours=analyzed_negative_seconds/3600
    poisson=None if not hours else [0. if negative_alerts==0 else float(.5*chi2.ppf(.025,2*negative_alerts)/hours),float(.5*chi2.ppf(.975,2*(negative_alerts+1))/hours)]
    report['uncertainty_intervals']['false_alarm_rate_poisson_95']=poisson
    report['uncertainty_intervals']['false_alarm_rate_reason']='conditional Poisson exposure interval; independent stationary event-process assumption needs review'
    report['false_alarms']={'events_in_reviewed_negative_segments':negative_alerts,
        'negative_audio_hours':negative_seconds/3600,'analyzed_negative_audio_hours':hours,
        'unscored_negative_seconds':negative_seconds-analyzed_negative_seconds,
        'per_negative_audio_hour':negative_alerts*3600/negative_seconds if negative_seconds and abs(negative_seconds-analyzed_negative_seconds)<1e-6 else None,
        'per_analyzed_negative_hour':negative_alerts/hours if hours else None}
    report['promotion_status']='BLOCKED_NO_STAKEHOLDER_HAZARD_GATES_OR_DEVICE_PARITY'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--task',choices=['audit','scream-validation','scream-test','asr'],default='audit')
    p.add_argument('--model');p.add_argument('--english-model');p.add_argument('--hindi-model')
    p.add_argument('--split',choices=['validation','test','reused_external'],default='validation');p.add_argument('--frozen',type=Path)
    a=p.parse_args()
    try:
        rows=load(a.manifest);audit=validate(rows,a.manifest.parent,'asr' if a.task=='asr' else 'scream')
        if a.output.exists():raise ValueError('Refusing overwrite or repeated held-out use')
        if a.task=='audit':report=audit
        elif a.task=='asr':
            from safora.asr.assets import validate_prepared_model
            specs=json.loads((Path(__file__).resolve().parents[3]/'models/android-assets.json').read_text())['vosk']
            artifacts={}
            for lang,path in [('en',a.english_model),('hi',a.hindi_model)]:
                if path:
                    validate_prepared_model(path,specs[lang])
                    artifacts[lang]={key:specs[lang][key] for key in ('artifact_id','model_version','sha256')}
            report=evaluate_asr(rows,{'en':a.english_model,'hi':a.hindi_model},split=a.split,base=a.manifest.parent)
            report['strict_provenance_audit']=audit;report['model_artifacts']=artifacts
        else:
            from safora.asr.assets import sha256
            if not a.model:raise ValueError('Local YAMNet required')
            model_sha=sha256(a.model)
            with Yamnet(a.model) as model:
                prepared=scream_rows(rows)
                if a.task=='scream-validation':
                    # A fixed, declared grid is provisional decision-rule research,
                    # not a performance/safety gate; only validation rows are read.
                    validation=[r for r in rows if r['split']=='validation']
                    if not any(r['events'] for r in validation) or not any(r['negative_segments'] for r in validation):raise ValueError('Validation event positives and reviewed negative exposure required')
                    candidates=[]
                    for threshold in (.2,.3,.5):
                        config=StreamConfig(threshold=threshold)
                        report=enrich(evaluate(prepared,model,config,split='validation',base=a.manifest.parent),rows)
                        if not report['complete']:raise ValueError('Incomplete validation cannot select thresholds')
                        candidates.append({'config':asdict(config),'metrics':report})
                    if not any(r['split']=='validation' for r in rows):raise ValueError('No validation split')
                    best=max(candidates,key=lambda r:(r['metrics']['event_metrics']['f1'] if r['metrics']['event_metrics']['f1'] is not None else -1,r['config']['threshold']))
                    report={'kind':'frozen_validation_rule_study','manifest_sha256':fingerprint(rows),'model_sha256':model_sha,
                            'candidates':candidates,'config':best['config'],'production_promotion':False}
                else:
                    if a.frozen is None:raise ValueError('Frozen validation result required')
                    frozen=json.loads(a.frozen.read_text())
                    if frozen.get('manifest_sha256')!=fingerprint(rows) or frozen.get('model_sha256')!=model_sha or frozen.get('kind')!='frozen_validation_rule_study':raise ValueError('Frozen identity mismatch')
                    if a.split!='test' or not any(r['split']=='test' for r in rows):raise ValueError('Untouched test split required')
                    # Reserve BEFORE inference. A failed run remains consumed.
                    a.output.parent.mkdir(parents=True,exist_ok=True)
                    with a.output.open('x') as f:json.dump({'status':'reserved','manifest_sha256':fingerprint(rows)},f)
                    report=enrich(evaluate(prepared,model,StreamConfig(**frozen['config']),split='test',base=a.manifest.parent),rows)
                    report['test_protocol']='local exclusive receipt; external copies/reruns cannot be prevented'
        a.output.parent.mkdir(parents=True,exist_ok=True)
        if not a.output.exists():
            with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
        else:a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        complete=report.get('complete',True)
        print(json.dumps({'status':'complete' if complete else 'incomplete','production_qualified':False}));return 0 if complete else 1
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'BLOCKED','error_type':type(exc).__name__,'accuracy_improvement':'NO_DEMONSTRATED_IMPROVEMENT'}));return 2


if __name__=='__main__':raise SystemExit(main())
