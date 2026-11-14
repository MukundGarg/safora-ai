"""Versioned research-model bundles. Integrity/attestations are not accuracy proof."""
import hashlib
import json
from pathlib import Path
from .risk import LABELS, VERSION


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def file_hashes(path):
    root=Path(path)
    return {str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='safora_metadata.json'}


def validate_bundle(path):
    root=Path(path)
    try: meta=json.loads((root/'safora_metadata.json').read_text())
    except (OSError,ValueError): raise ValueError('Verified fine-tuned risk bundle unavailable') from None
    if (meta.get('schema_version')!=2 or meta.get('labels')!=LABELS or meta.get('taxonomy_version')!=VERSION or
            meta.get('artifact_kind')!='RESEARCH_FINETUNED_ENCODER' or meta.get('training_completed') is not True or
            meta.get('validated_for_emergency_escalation') is not False or
            not isinstance(meta.get('model_version'),str) or not meta['model_version']):
        raise ValueError('Unsupported or untrained risk metadata')
    dataset=meta.get('dataset',{})
    if (not isinstance(dataset,dict) or len(dataset.get('sha256',''))!=64 or
            any(not isinstance(dataset.get('counts',{}).get(s),int) or dataset['counts'][s]<1 for s in ('train','validation','test'))):
        raise ValueError('Missing training dataset provenance')
    from safora.evaluation.risk import validate_thresholds
    validate_thresholds(meta.get('thresholds'))
    if meta.get('threshold_source')!='validation' or not isinstance(meta.get('max_tokens'),int) or not 16<=meta['max_tokens']<=256:
        raise ValueError('Missing frozen validation thresholds/token bound')
    hashes=meta.get('file_sha256')
    if not isinstance(hashes,dict) or not {'config.json','model.safetensors','tokenizer_config.json','tokenizer.json'}<=set(hashes):
        raise ValueError('Missing tokenizer or trained safetensors checksums')
    for name,expected in hashes.items():
        file=root/name
        if (not isinstance(expected,str) or not file.resolve().is_relative_to(root.resolve()) or
                file.is_symlink() or not file.is_file() or sha256(file)!=expected):
            raise ValueError('Risk bundle file integrity failure')
    return meta
