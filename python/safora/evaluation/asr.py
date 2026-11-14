"""Offline ASR evaluation by declared language/condition. Never writes transcript text."""
import argparse
import json
from pathlib import Path
import unicodedata
import wave
import numpy as np
from safora.asr.offline import VoskASR
from .metrics import edit_distance


def normalize(text):
    """NFC, casefold and whitespace collapse; retain punctuation/scripts, no transliteration."""
    return ' '.join(unicodedata.normalize('NFC',text).casefold().split())


def score(reference,hypothesis):
    ref,hyp=normalize(reference),normalize(hypothesis)
    words=ref.split();chars=list(ref)
    word_errors=edit_distance(words,hyp.split());char_errors=edit_distance(chars,list(hyp))
    return dict(word_errors=word_errors,reference_words=len(words),char_errors=char_errors,
                reference_chars=len(chars),wer=word_errors/len(words) if words else None,
                cer=char_errors/len(chars) if chars else None,
                empty_reference=not ref,empty_reference_nonempty_hypothesis=not ref and bool(hyp))


def validate_rows(rows,base):
    seen={};ids=set()
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('id'),str) or not row['id'] or row['id'] in ids:
            raise ValueError('Unique nonempty recording IDs required')
        ids.add(row['id'])
        if row.get('split') not in ('train','validation','test','reused_external'):
            raise ValueError('Explicit split required')
        for field in ('source_id','session_id'):
            if not isinstance(row.get(field),str) or not row[field]:raise ValueError('Source/session provenance required')
        if row.get('language') not in ('en','hi','hinglish','unknown'):raise ValueError('Explicit language stratum required')
        if row.get('asr_language') not in (None,'en','hi'):raise ValueError('Select one decoder language; no automatic/model concatenation mode')
        if not isinstance(row.get('path'),str) or not row['path']:raise ValueError('WAV path required')
        if row.get('reference') is not None and not isinstance(row['reference'],str):raise ValueError('reference must be text or null/omitted')
        if not isinstance(row.get('condition','unspecified'),str):raise ValueError('condition must be text')
        if row.get('group_id') is not None and (not isinstance(row['group_id'],str) or not row['group_id']):
            raise ValueError('group_id must be nonempty text when supplied')
        keys=[('path',str((base/row['path']).resolve()))]
        for field in ('source_id','session_id','group_id'):
            if row.get(field) is not None:keys.append((field,row[field]))
        for key in keys:
            if key in seen and seen[key]!=row['split']:raise ValueError('Source/session/path split leakage')
            seen[key]=row['split']


def _wav(path):
    with wave.open(str(path),'rb') as f:
        if f.getnchannels()!=1 or f.getframerate()!=16000 or f.getsampwidth()!=2 or f.getcomptype()!='NONE':
            raise ValueError('Evaluation requires PCM16 mono 16000 Hz WAV')
        if f.getnframes()>12*16000:raise ValueError('Recording exceeds 12-second ASR segment bound')
        pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2')
        return pcm.astype(np.float32)/32768


def evaluate(rows,model_paths,*,split,base=Path('.'),factory=None):
    base=Path(base);validate_rows(rows,base)
    if split not in ('train','validation','test','reused_external'):raise ValueError('Unknown split')
    factory=factory or (lambda lang:VoskASR(model_paths[lang],language=lang))
    results=[];errors=[];adapter=None;loaded=None
    try:
        # Group by selected decoder to load only one model at a time, deterministically.
        selected=[r for r in rows if r['split']==split]
        for row in sorted(selected,key=lambda r:(r.get('asr_language') or r['language'],r['id'])):
            language=row.get('asr_language') or (row['language'] if row['language'] in ('en','hi') else None)
            item=dict(id=row['id'],language=row['language'],asr_language=language,
                      condition=row.get('condition','unspecified'),reference_available=row.get('reference') is not None)
            if language is None:
                item.update(status='decoder_language_unavailable',wer=None,cer=None);results.append(item);continue
            audio=None
            try:
                audio=_wav(base/row['path'])
                if loaded!=language:
                    if adapter is not None:adapter.close();adapter=None
                    adapter=factory(language);loaded=language
                import time
                began=time.perf_counter();transcript=adapter.transcribe(audio);seconds=time.perf_counter()-began
                item.update(status='recognized',uncertain=transcript.uncertain,reason=transcript.reason,
                            duration_seconds=len(audio)/16000,processing_seconds=seconds)
                if transcript.reason=='malformed_native_result':
                    item.update(status='invalid_native_result',wer=None,cer=None)
                elif row.get('reference') is None:item.update(wer=None,cer=None)
                else:item.update(score(row['reference'],transcript.text))
            except Exception as exc:
                # Do not serialize exceptions containing paths, recognizer output, or text.
                item.update(status='unavailable',error_type=type(exc).__name__,wer=None,cer=None)
                errors.append({'id':row['id'],'error_type':type(exc).__name__})
            finally:
                if audio is not None:audio.fill(0)
            results.append(item)
    finally:
        if adapter is not None:adapter.close()
    def aggregate(items):
        scored=[r for r in items if 'word_errors' in r]
        counts={key:sum(r[key] for r in scored) for key in ('word_errors','reference_words','char_errors','reference_chars')}
        return dict(recordings=len(items),scored_recordings=len(scored),
                    missing_references=sum(not r['reference_available'] for r in items),
                    unavailable_recordings=sum(r['status']!='recognized' for r in items),
                    empty_references=sum(r['empty_reference'] for r in scored),
                    empty_reference_nonempty_hypotheses=sum(r['empty_reference_nonempty_hypothesis'] for r in scored),
                    **counts,wer=counts['word_errors']/counts['reference_words'] if counts['reference_words'] else None,
                    cer=counts['char_errors']/counts['reference_chars'] if counts['reference_chars'] else None)
    return dict(version=1,split=split,normalization='NFC + casefold + collapse whitespace; retain punctuation; code-point CER',
                benchmark_status='reused_not_independent' if split=='reused_external' else 'independence_not_verified',
                complete=all(r['status']=='recognized' for r in results),errors=errors,
                by_language={lang:aggregate([r for r in results if r['language']==lang]) for lang in ('en','hi','hinglish','unknown')},
                by_condition={condition:aggregate([r for r in results if r['condition']==condition]) for condition in sorted({r['condition'] for r in results})},
                recordings=results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--english-model');p.add_argument('--hindi-model');p.add_argument('--split',required=True,choices=['train','validation','test','reused_external'])
    a=p.parse_args();text=a.manifest.read_text()
    rows=json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]
    paths={key:value for key,value in [('en',a.english_model),('hi',a.hindi_model)] if value}
    report=evaluate(rows,paths,split=a.split,base=a.manifest.parent)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if report['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
