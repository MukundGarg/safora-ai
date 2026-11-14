"""Local supervised research training only. No corpus download or synthetic training.

--check-data performs dependency-free strict validation and reports blockers.
Actual training requires a locally supplied fast tokenizer/base safetensors and
independently reviewed permission-attested train/validation/test JSONL.
"""
import argparse
import json
import os
from pathlib import Path
from .data import load_jsonl, dataset_metadata, fingerprint
from safora.classifier.risk import LABELS, VERSION
from safora.classifier.text import encoder_text
from safora.evaluation.risk import calibrate, evaluate, evaluate_test_once


def preflight(data, base):
    rows=load_jsonl(data)
    root=Path(base)
    if not root.is_dir() or not (root/'config.json').is_file() or not (root/'model.safetensors').is_file():
        raise ValueError('Training blocked: compatible local base safetensors/config unavailable')
    return rows


def train(args):
    rows=preflight(args.data,args.base)
    if args.check_data: return {'status':'data_schema_validated_not_trained','dataset':dataset_metadata(rows)}
    if not 1<=args.epochs<=20 or not 1<=args.batch<=32 or not 1<=args.threads<=4:
        raise ValueError('Invalid bounded training configuration')
    dest=Path(args.output)
    if dest.exists(): raise ValueError('Refusing to overwrite an existing model/evaluation')
    # Explicit offline local loading; never remote/custom Python model code.
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(args.threads);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    tokenizer=AutoTokenizer.from_pretrained(args.base,local_files_only=True,trust_remote_code=False,use_fast=True)
    if not tokenizer.is_fast: raise ValueError('Supported export requires a local fast tokenizer')
    model=AutoModelForSequenceClassification.from_pretrained(args.base,num_labels=len(LABELS),
        problem_type='multi_label_classification',local_files_only=True,trust_remote_code=False,
        use_safetensors=True,ignore_mismatched_sizes=True,id2label=dict(enumerate(LABELS)),
        label2id={v:i for i,v in enumerate(LABELS)}).cpu()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
    splits={s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
    def encode(batch):return tokenizer([encoder_text(r['text'],r['context']) for r in batch],padding=True,truncation=True,max_length=256,return_tensors='pt')
    def predict(batch):
        outputs=[]
        with torch.inference_mode():
            for start in range(0,len(batch),args.batch): outputs.extend(model(**encode(batch[start:start+args.batch])).logits.sigmoid().tolist())
        return outputs
    for _ in range(args.epochs):
        model.train()
        for start in range(0,len(splits['train']),args.batch):
            batch=splits['train'][start:start+args.batch]
            target=torch.tensor([[float(l in r['labels']) for l in LABELS] for r in batch])
            optimizer.zero_grad();loss=model(**encode(batch),labels=target).loss
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    model.eval()
    val_scores=predict(splits['validation']);calibration=calibrate(splits['validation'],val_scores)
    # No test data has reached model training/calibration. Reserve a fresh directory
    # only after actual optimization; no arbitrary base/random weights are published.
    dest.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(dest,safe_serialization=True);tokenizer.save_pretrained(dest)
    (dest/'calibration.json').write_text(json.dumps(calibration,indent=2,allow_nan=False)+'\n')
    validation=evaluate(splits['validation'],val_scores,calibration['thresholds'])
    test=evaluate_test_once(splits['test'],predict,calibration['thresholds'],dest)
    (dest/'evaluation.json').write_text(json.dumps({'validation':validation,'test':test},indent=2,allow_nan=False)+'\n')
    from safora.classifier.bundle import file_hashes, validate_bundle
    metadata={'schema_version':2,'taxonomy_version':VERSION,'labels':LABELS,
        'artifact_kind':'RESEARCH_FINETUNED_ENCODER','training_completed':True,
        'model_version':VERSION+'-'+fingerprint(rows)[:12],
        'validated_for_emergency_escalation':False,'dataset':dataset_metadata(rows),
        'thresholds':calibration['thresholds'],'threshold_source':'validation','max_tokens':256,
        'training':{'seed':42,'epochs':args.epochs,'batch':args.batch,'threads':args.threads,
                    'optimizer':'AdamW','learning_rate':2e-5,'base_config_sha256':fingerprint(json.loads((Path(args.base)/'config.json').read_text()))},
        'file_sha256':file_hashes(dest)}
    (dest/'safora_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    validate_bundle(dest)
    return {'status':'research_training_complete_not_safety_validated','output':str(dest)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--base',required=True);p.add_argument('--output')
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--check-data',action='store_true');p.add_argument('--status-output',type=Path)
    args=p.parse_args()
    if not args.check_data and not args.output:p.error('--output required for training')
    try: report=train(args);code=0
    except (ValueError,ImportError,OSError) as exc:
        # No dataset text, native exception detail, or full paths in persistent status.
        report={'status':'training_blocked','reason':str(exc) if isinstance(exc,ValueError) and str(exc).startswith('Training blocked:') else type(exc).__name__};code=2
    if args.status_output:
        args.status_output.parent.mkdir(parents=True,exist_ok=True)
        args.status_output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));return code


if __name__=='__main__': raise SystemExit(main())
