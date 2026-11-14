"""Permission-attested, strongly annotated audio manifests and leakage checks.

Offline research boundary, not a microphone runtime. No corpus is downloaded or
labels inferred. Fingerprints detect some duplicates, not all semantic derivatives.
"""
import hashlib
import json
from pathlib import Path
import wave
import numpy as np
from safora.audio.stream import waveform
from safora.asr.assets import sha256
from safora.training.data import _unique_object, _non_json_constant, fingerprint

SPLITS={'train','validation','test','reused_external'}
HARD_NEGATIVES={'baby_crying','laughter','cheering','singing','shouting','sirens','traffic',
    'conversation','household','noisy_environment','device_handling','other'}


def load(path):
    path=Path(path)
    if not path.is_file():raise ValueError('No permission-reviewed annotated audio manifest supplied')
    if path.stat().st_size>64*1024*1024:raise ValueError('Manifest exceeds bound')
    result=[]
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip() or len(line)>65536:raise ValueError('Malformed manifest row')
            result.append(json.loads(line,object_pairs_hook=_unique_object,parse_constant=_non_json_constant))
    return result


def inspect_wav(path):
    with wave.open(str(path),'rb') as stream:
        rate,channels=stream.getframerate(),stream.getnchannels()
        n=stream.getnframes()
        if (stream.getsampwidth()!=2 or stream.getcomptype()!='NONE' or not 1<=channels<=2 or
                not 8000<=rate<=48000 or not 0<n<=rate*600):raise ValueError('Unsupported/bounded PCM16 recording')
        raw=np.frombuffer(stream.readframes(n),dtype='<i2')
        if len(raw)!=n*channels:raise ValueError('Truncated recording')
        pcm=waveform(raw.reshape(-1,channels),rate)
    canonical=np.rint(np.clip(pcm,-1,1)*32767).astype('<i2')
    # Aligned 0.25s fingerprints catch exact decoded/cropped windows even when
    # WAV headers differ. Silence/near-constant blocks are excluded as nonunique.
    chunks=set()
    for start in range(0,len(canonical)-4000+1,2000):
        part=canonical[start:start+4000]
        if np.std(part.astype(np.float32))>16:
            chunks.add(hashlib.sha256(part.tobytes()).hexdigest())
    result={'duration_seconds':n/rate,'sample_rate':rate,'channels':channels,
            'file_sha256':sha256(path),'pcm_sha256':hashlib.sha256(canonical.tobytes()).hexdigest(),
            'chunk_fingerprints':chunks}
    pcm.fill(0);return result


def _intervals(values,duration):
    if not isinstance(values,list):raise ValueError('Explicit ordered annotation list required')
    previous=0
    for pair in values:
        if (not isinstance(pair,list) or len(pair)!=2 or
            any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in pair) or
            not 0<=pair[0]<pair[1]<=duration or pair[0]<previous):raise ValueError('Invalid annotation intervals')
        previous=pair[1]


def validate(rows,base=Path('.'),task='scream',check_audio=True):
    if task not in ('scream','asr') or not isinstance(rows,list) or not 1<=len(rows)<=100000:raise ValueError('Invalid evidence set')
    ids=set();seen={};fingerprints={};results=[]
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Recording object required')
        for key in ('id','path','source_id','session_id','root_recording_id','device_id','license','condition'):
            if not isinstance(row.get(key),str) or not row[key].strip():raise ValueError('Missing recording provenance: '+key)
        if row['id'] in ids:raise ValueError('Duplicate recording ID')
        ids.add(row['id'])
        if row.get('split') not in SPLITS:raise ValueError('Explicit split required')
        speakers=row.get('speaker_ids')
        if not isinstance(speakers,list) or not speakers or any(not isinstance(s,str) or not s.strip() or s=='unknown' for s in speakers):
            raise ValueError('Reviewed speaker grouping required; unknown identities cannot establish isolation')
        if len(set(speakers))!=len(speakers):raise ValueError('Duplicate speaker ID')
        permission=row.get('provenance',{})
        if not isinstance(permission,dict) or permission.get('verified') is not True or any(not isinstance(permission.get(k),str) or not permission[k].strip() for k in ('origin','reviewed_by','permission_basis')):
            raise ValueError('Permission review attestation required')
        if not isinstance(row.get('duration_seconds'),(int,float)) or isinstance(row['duration_seconds'],bool) or not np.isfinite(row['duration_seconds']) or not 0<row['duration_seconds']<=600:
            raise ValueError('Explicit bounded duration required')
        if not isinstance(row.get('channel_metadata'),dict) or not isinstance(row.get('noise_metadata'),dict):raise ValueError('Channel/noise metadata required')
        expected=row.get('sha256')
        if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):raise ValueError('File SHA256 required')
        keys=[('path',str((Path(base)/row['path']).resolve())),('file',expected)]
        keys.extend((key,row[key]) for key in ('source_id','session_id','root_recording_id'))
        keys.extend(('speaker',s) for s in speakers)
        if row.get('parent_recording_id') is not None:
            if not isinstance(row['parent_recording_id'],str) or not row['parent_recording_id']:raise ValueError('Invalid crop lineage')
            keys.append(('root_recording_id',row['parent_recording_id']))
        for key in keys:
            if key in seen and seen[key]!=row['split']:raise ValueError('Source/session/speaker/crop/file split leakage')
            seen[key]=row['split']
        if task=='scream':
            _intervals(row.get('events'),row['duration_seconds']);_intervals(row.get('negative_segments'),row['duration_seconds'])
            for a,b in row['events']:
                if any(a<d and c<b for c,d in row['negative_segments']):raise ValueError('Positive/negative annotations overlap')
            if row.get('hard_negative') is not None and row['hard_negative'] not in HARD_NEGATIVES:raise ValueError('Unknown hard-negative stratum')
            if not row['events'] and not row['negative_segments']:raise ValueError('No strongly labeled region; weak clip labels are not window labels')
        else:
            if row['duration_seconds']>12:raise ValueError('ASR evaluation requires annotated utterances at most12s, not invented cropped references')
            if row.get('language') not in ('en','hi','hinglish','unknown'):raise ValueError('Explicit ASR language stratum required')
            if row.get('reference') is not None and not isinstance(row['reference'],str):raise ValueError('Reference text or null required')
            if row.get('asr_language') not in (None,'en','hi'):raise ValueError('Explicit decoder selection only')
        info={}
        if check_audio:
            info=inspect_wav(Path(base)/row['path'])
            if info['file_sha256']!=expected or abs(info['duration_seconds']-row['duration_seconds'])>1/info['sample_rate']:
                raise ValueError('Recording integrity/duration mismatch')
            key=('pcm',info['pcm_sha256'])
            if key in seen and seen[key]!=row['split']:raise ValueError('Decoded audio duplicate crosses splits')
            seen[key]=row['split']
            for mark in info['chunk_fingerprints']:
                if mark in fingerprints and fingerprints[mark]!=row['split']:raise ValueError('Acoustic crop/fingerprint overlap crosses splits')
                fingerprints[mark]=row['split']
                if len(fingerprints)>250000:raise ValueError('Global fingerprint memory bound exceeded; use a reviewed external index for a larger corpus')
        results.append({k:v for k,v in info.items() if k!='chunk_fingerprints'})
    return {'manifest_sha256':fingerprint(rows),'recordings':len(rows),'task':task,'files':results,
            'independence':'supplier_grouping_checked_not_proven','permission':'supplier_attestation_not_legal_verification'}
