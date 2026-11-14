"""Authored PCM/schema fixtures only, not a real annotated safety corpus."""
import copy,json,tempfile,unittest,wave,struct
from pathlib import Path
import numpy as np
from safora.evaluation.audio_evidence import validate,inspect_wav,load
from safora.evaluation.audio_experiment import scream_rows,wilson,enrich
from safora.training.scream_probe import target,features,run
from safora.asr.assets import sha256

class AudioEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.audio=(np.sin(np.arange(16000)*.107)*20000).astype('<i2')
        self.write('fixture.wav',self.audio)
    def tearDown(self):self.temp.cleanup()
    def write(self,name,audio):
        with wave.open(str(self.root/name),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(audio.tobytes())
    def row(self):
        return dict(id='fixture',path='fixture.wav',sha256=sha256(self.root/'fixture.wav'),source_id='test-source',session_id='test-session',
            root_recording_id='test-root',speaker_ids=['test-speaker'],device_id='unit-generator',license='TEST-ONLY-NOT-TRAINING-DATA',
            condition='clean',split='train',duration_seconds=1.,events=[[.2,.3]],negative_segments=[[.4,1.]],channel_metadata={'mono':True},noise_metadata={'authored':True},
            provenance={'verified':True,'origin':'unit fixture','reviewed_by':'test','permission_basis':'parser contract fixture'})
    def test_valid_manifest_content_hash_and_duration(self):
        report=validate([self.row()],self.root);self.assertEqual(report['recordings'],1)
        self.assertEqual(report['files'][0]['sample_rate'],16000)
        self.assertNotIn('chunk_fingerprints',report['files'][0])
    def test_permission_missing_metadata_and_invalid_intervals(self):
        for update in [dict(license=''),dict(provenance={}),dict(speaker_ids=[]),dict(device_id=''),dict(events=[[.4,.2]]),dict(negative_segments=[[.25,.5]]),dict(events=[],negative_segments=[])]:
            row=self.row();row.update(update)
            with self.assertRaises(ValueError):validate([row],self.root)
    def test_source_speaker_session_crop_leakage(self):
        for field in ['source_id','session_id','root_recording_id','speaker_ids']:
            a=self.row();b=self.row();b.update(id='second',path='other.wav',sha256='a'*64,source_id='second',session_id='second',root_recording_id='second',speaker_ids=['second'],split='test');b[field]=a[field]
            with self.assertRaisesRegex(ValueError,'leakage'):validate([a,b],self.root,check_audio=False)
    def test_file_corruption_and_unknown_speaker_are_blocked(self):
        row=self.row();row['sha256']='0'*64
        with self.assertRaises(ValueError):validate([row],self.root)
        row=self.row();row['speaker_ids']=['unknown']
        with self.assertRaises(ValueError):validate([row],self.root)
    def test_decoded_duplicate_different_wav_headers_and_crops(self):
        self.write('copy.wav',self.audio.copy())
        path=self.root/'copy.wav';data=bytearray(path.read_bytes());data.extend(b'JUNK'+struct.pack('<I',4)+b'test');struct.pack_into('<I',data,4,len(data)-8);path.write_bytes(data)
        a=self.row();b=self.row();b.update(id='b',path='copy.wav',sha256=sha256(self.root/'copy.wav'),source_id='b',session_id='b',root_recording_id='b',speaker_ids=['b'],split='test')
        with self.assertRaisesRegex(ValueError,'Decoded'):validate([a,b],self.root)
        self.write('crop.wav',self.audio[4000:12000]);b.update(path='crop.wav',sha256=sha256(self.root/'crop.wav'),duration_seconds=.5,events=[],negative_segments=[[0,.5]])
        with self.assertRaisesRegex(ValueError,'fingerprint'):validate([a,b],self.root)
    def test_asr_missing_reference_remains_unknown(self):
        row=self.row();row.update(language='hinglish',reference=None)
        self.assertEqual(validate([row],self.root,'asr')['task'],'asr')
        row['reference']=123
        with self.assertRaises(ValueError):validate([row],self.root,'asr')
    def test_no_weak_recording_label_to_all_windows(self):
        self.assertEqual(target(0,.25,[[.1,.11]],[[.5,1]])[0],1)
        self.assertIsNone(target(.25,.5,[[.1,.11]],[[.5,1]])[0])
        self.assertEqual(target(.5,.75,[[.1,.11]],[[.5,1]])[0],0)
        self.assertEqual(features(np.zeros(4000)).shape,(11,))
        rows=scream_rows([{**self.row(),'events':[],'negative_segments':[[.4,1]]}]);self.assertIsNone(rows[0]['label'])
    def test_intervals_are_unavailable_without_denominator(self):
        self.assertIsNone(wilson(0,0));lo,hi=wilson(5,10);self.assertLess(lo,.5);self.assertGreater(hi,.5)
    def test_strict_json_and_absent_data_fit_creates_no_weights(self):
        path=self.root/'bad.jsonl';path.write_text('{"id":1,"id":2}\n')
        with self.assertRaises(ValueError):load(path)
        output=self.root/'no-model'
        with self.assertRaises(ValueError):run(self.root/'absent.jsonl',self.root/'absent-model',output)
        self.assertFalse(output.exists())
    def test_partial_annotations_exclude_unknown_time_and_unanalyzed_exposure(self):
        row=self.row()
        report={'clip_metrics':{'tp':1,'fp':0,'fn':0,'confusion_matrix':[[0,0],[0,1]]},
            'event_tolerance_seconds':0.,'clips':[{'id':row['id'],'label':True,'detected':True,'analyzed_audio_seconds':.8,
                'events':[{'time':.1},{'time':.25},{'time':.7}]}]}
        result=enrich(report,[row])
        self.assertEqual(result['event_metrics']['unannotated_detections_unscored'],1)
        self.assertEqual(result['event_metrics']['tp'],1);self.assertEqual(result['event_metrics']['fp'],1)
        self.assertAlmostEqual(result['false_alarms']['unscored_negative_seconds'],.2)
        self.assertIsNone(result['false_alarms']['per_negative_audio_hour'])
        self.assertGreater(result['false_alarms']['per_analyzed_negative_hour'],0)
        self.assertEqual(len(result['uncertainty_intervals']['false_alarm_rate_poisson_95']),2)
    def test_prepared_directory_does_not_trust_a_rewritten_receipt(self):
        from safora.asr.assets import REQUIRED_FILES,validate_prepared_model
        root=self.root/'model'
        for name in REQUIRED_FILES:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'unit-placeholder-not-weights')
        files={name:{'bytes':(root/name).stat().st_size,'sha256':sha256(root/name)} for name in REQUIRED_FILES}
        validate_prepared_model(root,{'files':files})
        path=root/REQUIRED_FILES[0];path.write_bytes(b'UNIT-placeholder-not-weights')
        (root/'.safora-model.json').write_text(json.dumps({'file_sha256':{REQUIRED_FILES[0]:sha256(path)}}))
        with self.assertRaises(ValueError):validate_prepared_model(root,{'files':files})
