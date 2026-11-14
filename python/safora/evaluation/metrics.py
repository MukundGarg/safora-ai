import numpy as np

def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
    return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,'tp':tp,'fp':fp,'fn':fn}
def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]
def error_rates(reference,hypothesis):
    a=reference.casefold();b=hypothesis.casefold()
    return {'wer':edit_distance(a.split(),b.split())/max(1,len(a.split())), 'cer':edit_distance(a,b)/max(1,len(a))}
def event_metrics(predicted,truth,negative_hours,tolerance=1.):
    """truth: [(start,end)]. Greedy chronological one-to-one interval matching."""
    remaining=set(range(len(truth)));delays=[];false=[]
    for t in sorted(predicted):
        options=[i for i in remaining if truth[i][0]<=t<=truth[i][1]+tolerance]
        if options:
            i=min(options,key=lambda i:truth[i][0]);remaining.remove(i);delays.append(t-truth[i][0])
        else:false.append(t)
    result=prf(len(delays),len(false),len(remaining))
    result.update(false_alerts_per_hour=len(false)/negative_hours if negative_hours>0 else None,delays_seconds=delays,missed_events=sorted(remaining))
    return result

def multilabel_metrics(truth,predicted,labels):
    if len(truth)!=len(predicted):raise ValueError('Truth/prediction length mismatch')
    if any(not set(values)<=set(labels) for values in [*truth,*predicted]):raise ValueError('Unknown evaluation label')
    result={}
    for label in labels:
        tp=sum(label in a and label in b for a,b in zip(truth,predicted))
        fp=sum(label not in a and label in b for a,b in zip(truth,predicted))
        fn=sum(label in a and label not in b for a,b in zip(truth,predicted))
        tn=len(truth)-tp-fp-fn
        result[label]={'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if tp+fn else None,
            'support':tp+fn,'tp':tp,'fp':fp,'fn':fn,
            'confusion_matrix':[[tn,fp],[fn,tp]]}
    return result

def validate_splits(rows):
    seen={}
    for row in rows:
        if row['split'] not in {'train','validation','test'}:raise ValueError('Unknown split')
        for field in ['source_id','session_id']:
            key=(field,row[field])
            if key in seen and seen[key]!=row['split']:raise ValueError('Source/session leakage: '+str(key))
            seen[key]=row['split']
    return True

def add_noise(audio,snr_db,seed=0):
    rng=np.random.default_rng(seed);noise=rng.normal(size=len(audio))
    scale=np.sqrt(np.mean(audio**2)/(10**(snr_db/10)*max(np.mean(noise**2),1e-12)))
    return np.clip(audio+noise*scale,-1,1).astype(np.float32)
