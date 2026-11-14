"""Strict, local, provenance-attested JSONL. Schema checks do not prove legal permission."""
import hashlib
import json
from pathlib import Path
from safora.classifier.risk import consistent_labels
from safora.classifier.text import normalize, prepare, encoder_text, MAX_TEXT, MAX_CONTEXT, MAX_TURNS

LANGUAGES = {'en','hi','romanized_hi','hinglish','unknown'}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_rows(rows, require_splits=True):
    if not isinstance(rows,list) or not rows or len(rows)>100000: raise ValueError('Dataset must have 1..100000 rows')
    seen = {}; ids = set(); texts = {}; counts = dict.fromkeys(('train','validation','test'),0)
    for row in rows:
        if not isinstance(row,dict): raise ValueError('Dataset row must be an object')
        for field in ('text','language','source_id','session_id','license'):
            if not isinstance(row.get(field),str) or not row[field].strip(): raise ValueError('Missing required text field: '+field)
        if len(row['text'])>MAX_TEXT: raise ValueError('Training text exceeds inference bound')
        if row['language'] not in LANGUAGES: raise ValueError('Unknown language annotation')
        if row.get('split') not in counts: raise ValueError('Explicit train/validation/test split required')
        counts[row['split']] += 1
        if not isinstance(row.get('labels'),list) or any(not isinstance(l,str) for l in row['labels']) or len(row['labels']) != len(set(row['labels'])):
            raise ValueError('Unique explicit labels required')
        consistent_labels(row['labels'])
        context = row.get('context')
        if (not isinstance(context,list) or len(context)>MAX_TURNS or
                any(not isinstance(t,str) or len(t)>MAX_TEXT for t in context) or sum(map(len,context))>MAX_CONTEXT):
            raise ValueError('Context must match bounded inference contract')
        clean, _, _ = prepare(row['text'],context)
        if not clean: raise ValueError('Empty training text')
        provenance = row.get('provenance')
        if not isinstance(provenance,dict) or provenance.get('verified') is not True:
            raise ValueError('Verified permission/provenance attestation required')
        for field in ('origin','reviewed_by','permission_basis'):
            if not isinstance(provenance.get(field),str) or not provenance[field].strip():
                raise ValueError('Missing provenance review field')
        if not isinstance(row.get('hard_negative',False),bool): raise ValueError('hard_negative must be boolean')
        if row.get('hard_negative') and row['labels'] != ['neutral_or_uncertain']:
            raise ValueError('Hard negatives must be explicitly neutral')
        annotations = row.get('annotations',[])
        if not isinstance(annotations,list): raise ValueError('annotations must be a list')
        for annotation in annotations:
            if (not isinstance(annotation,dict) or not isinstance(annotation.get('annotator_id'),str) or
                    not annotation['annotator_id'] or annotation.get('status') not in ('reviewed','adjudicated') or
                    not isinstance(annotation.get('labels'),list) or
                    consistent_labels(annotation['labels']) != set(row['labels'])):
                raise ValueError('Inconsistent or unresolved human annotation')
        if 'id' in row:
            if not isinstance(row['id'],str) or not row['id'] or row['id'] in ids: raise ValueError('Duplicate/invalid row ID')
            ids.add(row['id'])
        for field in ('source_id','session_id','group_id'):
            if field not in row: continue
            if not isinstance(row[field],str) or not row[field]: raise ValueError('Invalid provenance/group ID')
            key = (field,row[field])
            if key in seen and seen[key] != row['split']: raise ValueError('Source/session/group split leakage')
            seen[key] = row['split']
        # Hash normalized full context+text for same-split duplicates/conflicts;
        # also prohibit identical current text crossing splits conservatively.
        key = encoder_text(row['text'],context).casefold()
        if key in texts: raise ValueError('Duplicate or conflicting annotated conversation')
        texts[key] = True
        key = ('current_text',normalize(row['text']).casefold())
        if key in seen and seen[key] != row['split']: raise ValueError('Duplicate text crosses splits')
        seen[key] = row['split']
    if require_splits and any(n==0 for n in counts.values()): raise ValueError('Nonempty independent train/validation/test splits required')
    return counts


def _non_json_constant(value):
    raise ValueError('Non-JSON numeric constant')


def load_jsonl(path):
    path = Path(path)
    if not path.is_file(): raise ValueError('Training blocked: no verified dataset')
    if path.stat().st_size>64*1024*1024: raise ValueError('Dataset exceeds 64 MiB bound')
    rows = []
    with path.open(encoding='utf-8',errors='strict') as stream:
        for line in stream:
            if len(line)>65536 or not line.strip(): raise ValueError('Invalid/empty JSONL row')
            try: rows.append(json.loads(line,object_pairs_hook=_unique_object,parse_constant=_non_json_constant))
            except (ValueError,TypeError,RecursionError): raise ValueError('Malformed JSONL row') from None
    validate_rows(rows)
    return rows


def dataset_metadata(rows):
    validate_rows(rows)
    return {'sha256':fingerprint(rows),'counts':{s:sum(r['split']==s for r in rows) for s in ('train','validation','test')},
            'split_sha256':{s:fingerprint([r for r in rows if r['split']==s]) for s in ('train','validation','test')},
            'permission_status':'supplier_attested_not_independently_verified'}
