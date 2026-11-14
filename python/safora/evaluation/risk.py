"""Validation-only decision thresholds and text-free multi-label evaluation."""
import math
from safora.classifier.risk import LABELS, POSITIVE_LABELS, consistent_labels


def validate_scores(scores):
    for row in scores:
        if len(row)!=len(LABELS) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in row):
            raise ValueError('Expected finite probability-shaped scores for all labels')


def validate_thresholds(thresholds):
    if not isinstance(thresholds,dict) or set(thresholds)!=set(POSITIVE_LABELS): raise ValueError('Per-positive-label thresholds required')
    for value in thresholds.values():
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1):
            raise ValueError('Invalid threshold')


def select_labels(scores, thresholds):
    validate_scores([scores]); validate_thresholds(thresholds)
    labels = {label for label,score in zip(POSITIVE_LABELS,scores) if thresholds[label] is not None and score>=thresholds[label]}
    return labels or {'neutral_or_uncertain'}


def metrics(truth,predicted):
    if len(truth)!=len(predicted): raise ValueError('Truth/prediction length mismatch')
    truth=[consistent_labels(t) for t in truth];predicted=[consistent_labels(p) for p in predicted]
    from .metrics import multilabel_metrics
    classes = multilabel_metrics(truth,predicted,LABELS)
    supported=[classes[l]['f1'] for l in POSITIVE_LABELS if classes[l]['f1'] is not None]
    tp=sum(classes[l]['confusion_matrix'][1][1] for l in POSITIVE_LABELS)
    fp=sum(classes[l]['confusion_matrix'][0][1] for l in POSITIVE_LABELS)
    fn=sum(classes[l]['confusion_matrix'][1][0] for l in POSITIVE_LABELS)
    return dict(recordings=len(truth),per_class=classes,
                macro_f1=sum(supported)/len(supported) if supported else None,
                macro_supported_labels=len(supported),micro_f1=2*tp/(2*tp+fp+fn) if tp+fn else None,
                averaging='positive categories only; unsupported reference classes excluded from macro F1')


def calibrate(rows, scores):
    if not rows or any(r.get('split')!='validation' for r in rows): raise ValueError('Calibration requires validation rows only')
    if len(rows)!=len(scores): raise ValueError('Score count mismatch')
    validate_scores(scores)
    truth=[consistent_labels(r['labels']) for r in rows];thresholds={};details={}
    # Fixed grid, no test access. Tie -> higher threshold (more conservative).
    for i,label in enumerate(POSITIVE_LABELS):
        positives=sum(label in t for t in truth)
        if positives in (0,len(rows)):
            thresholds[label]=None;details[label]={'status':'unavailable_requires_positive_and_negative_validation_examples'};continue
        best=(-1.,-1.)
        for tick in range(5,100,5):
            threshold=tick/100
            tp=sum(label in t and s[i]>=threshold for t,s in zip(truth,scores))
            fp=sum(label not in t and s[i]>=threshold for t,s in zip(truth,scores))
            fn=positives-tp;f1=2*tp/(2*tp+fp+fn)
            best=max(best,(f1,threshold))
        thresholds[label]=best[1];details[label]={'status':'selected','validation_f1':best[0]}
    return {'thresholds':thresholds,'per_label':details,'method':'fixed-grid validation F1; ties choose higher threshold',
            'probability_calibrated':False,'split':'validation'}


def evaluate(rows,scores,thresholds):
    if len(rows)!=len(scores): raise ValueError('Score count mismatch')
    predicted=[select_labels(s,thresholds) for s in scores]
    result=metrics([r['labels'] for r in rows],predicted)
    result['by_language']={language:metrics([r['labels'] for r in rows if r['language']==language],
        [p for r,p in zip(rows,predicted) if r['language']==language]) for language in sorted({r['language'] for r in rows})}
    # Hash identifiers rather than saving raw text or potentially private source IDs.
    from safora.training.data import fingerprint
    hard=[(r,p) for r,p in zip(rows,predicted) if r.get('hard_negative')]
    result['hard_negatives']={'count':len(hard),'false_positive_count':sum(p!={'neutral_or_uncertain'} for _,p in hard),
        'errors':[{'row_sha256':fingerprint(r),'predicted_labels':sorted(p),'language':r['language']}
                  for r,p in hard if p!={'neutral_or_uncertain'}]}
    result['errors']=[{'row_sha256':fingerprint(r),'false_positive_labels':sorted(p-set(r['labels'])),
                      'false_negative_labels':sorted(set(r['labels'])-p)}
                     for r,p in zip(rows,predicted) if p!=set(r['labels'])]
    return result


def evaluate_test_once(rows, predict, thresholds, directory):
    """Local exclusive consumption receipt; never calibrates. No global benchmark registry.

    A failed evaluation consumes this bundle's receipt too: explicitly review the
    failed run rather than silently retrying/tuning against the same held-out set.
    """
    import json
    from pathlib import Path
    from safora.training.data import fingerprint
    if not rows or any(r.get('split')!='test' for r in rows): raise ValueError('Test rows only')
    validate_thresholds(thresholds)
    receipt=Path(directory)/'heldout-evaluation.json'
    with receipt.open('x') as stream:
        json.dump({'status':'reserved','test_sha256':fingerprint(rows),'thresholds_sha256':fingerprint(thresholds)},stream)
    report=evaluate(rows,predict(rows),thresholds)
    receipt.write_text(json.dumps({'status':'complete','test_sha256':fingerprint(rows),
        'thresholds_sha256':fingerprint(thresholds),'report':report},indent=2,allow_nan=False)+'\n')
    return report


def main():
    """Evaluate offline score manifests; bind validation thresholds to dataset/model.

    Scores must come from the declared real checkpoint to support any accuracy
    claim. This tool checks identity consistency, not authenticity of supplied scores.
    """
    import argparse,json
    from pathlib import Path
    from safora.training.data import load_jsonl,fingerprint
    p=argparse.ArgumentParser(description=main.__doc__)
    p.add_argument('--data',required=True);p.add_argument('--scores',required=True,type=Path)
    p.add_argument('--split',required=True,choices=['validation','test']);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--calibration',type=Path)
    a=p.parse_args()
    try:
        rows=load_jsonl(a.data);selected=[r for r in rows if r['split']==a.split]
        supplied=json.loads(a.scores.read_text());model_sha=supplied.get('model_sha256')
        if not isinstance(model_sha,str) or len(model_sha)!=64 or any(c not in '0123456789abcdef' for c in model_sha):
            raise ValueError('Model score provenance fingerprint required')
        if supplied.get('dataset_sha256')!=fingerprint(rows):raise ValueError('Score dataset fingerprint mismatch')
        items=supplied['rows']
        if not isinstance(items,list) or len(items)!=len(selected):raise ValueError('Score recording count mismatch')
        by_id={r['row_sha256']:r['scores'] for r in items}
        if len(by_id)!=len(items) or set(by_id)!={fingerprint(r) for r in selected}:raise ValueError('Score row identity mismatch')
        values=[by_id[fingerprint(r)] for r in selected];validate_scores(values)
        if a.output.exists():raise ValueError('Refusing to overwrite frozen calibration/evaluation')
        if a.split=='validation':
            result={**calibrate(selected,values),'dataset_sha256':fingerprint(rows),'model_sha256':model_sha,
                    'validation_sha256':fingerprint(selected)}
            a.output.parent.mkdir(parents=True,exist_ok=True)
            with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
        else:
            if a.calibration is None:raise ValueError('Frozen calibration required')
            calibration=json.loads(a.calibration.read_text())
            if (calibration.get('split')!='validation' or calibration.get('dataset_sha256')!=fingerprint(rows) or
                    calibration.get('model_sha256')!=model_sha or
                    calibration.get('validation_sha256')!=fingerprint([r for r in rows if r['split']=='validation'])):
                raise ValueError('Calibration provenance mismatch')
            a.output.mkdir(parents=True,exist_ok=False)
            evaluate_test_once(selected,lambda _:values,calibration['thresholds'],a.output)
        print(json.dumps({'status':'complete','accuracy_provenance':'supplier_scores_not_independently_verified'}));return 0
    except (OSError,ValueError,KeyError,TypeError):
        print(json.dumps({'status':'evaluation_blocked','reason':'invalid_or_unavailable_data_scores_or_frozen_calibration'}));return 2


if __name__=='__main__':raise SystemExit(main())
