"""Fine tune only consented/licensed JSONL data; never auto-generates examples.
Rows: text, context (list), labels (list), source_id, session_id, split, license.
Run with --base pointing to a local multilingual encoder directory.
"""
import argparse,json
from pathlib import Path
from safora.classifier.risk import LABELS
from safora.evaluation.metrics import validate_splits,multilabel_metrics

def train(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    rows=[json.loads(x) for x in Path(args.data).read_text().splitlines() if x.strip()]
    validate_splits(rows)
    if not rows or any(not r.get('license') or not set(r['labels'])<=set(LABELS) for r in rows):raise ValueError('Missing license or invalid labels')
    splits={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
    if any(not v for v in splits.values()):raise ValueError('Need independent train/validation/test data')
    torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(args.base,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(args.base,num_labels=len(LABELS),problem_type='multi_label_classification',local_files_only=True,id2label=dict(enumerate(LABELS)),label2id={v:i for i,v in enumerate(LABELS)})
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
    def encode(batch):return tokenizer([' [SEP] '.join([*r.get('context',[]),r['text']]) for r in batch],padding=True,truncation=True,max_length=256,return_tensors='pt')
    for epoch in range(args.epochs):
        model.train()
        for start in range(0,len(splits['train']),args.batch):
            batch=splits['train'][start:start+args.batch]
            labels=torch.tensor([[float(l in r['labels']) for l in LABELS] for r in batch])
            optimizer.zero_grad();loss=model(**encode(batch),labels=labels).loss;loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    model.eval();report={}
    for split in ['validation','test']:
        predicted=[]
        with torch.inference_mode():
            for row in splits[split]:
                scores=model(**encode([row])).logits.sigmoid()[0]
                predicted.append({l for l,s in zip(LABELS,scores) if s>=.5})
        report[split]=multilabel_metrics([set(r['labels']) for r in splits[split]],predicted,LABELS)
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(dest);tokenizer.save_pretrained(dest)
    (dest/'safora_metadata.json').write_text(json.dumps({'labels':LABELS,'base':args.base,'validated_for_emergency_escalation':False,'threshold':.5}))
    (dest/'evaluation.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--base',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch',type=int,default=8)
    train(p.parse_args())
