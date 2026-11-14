"""Evaluator arithmetic on generated PCM markers + a stub model, NOT accuracy data."""
import tempfile
from pathlib import Path
import unittest
import wave
import numpy as np
from safora.audio.ingestion import StreamConfig
from safora.evaluation.scream import analyze_wav,evaluate,validate_manifest
from safora.evaluation.calibrate import sweep

class MarkerModel:
    window_samples=15600
    def predict(self,x):return {'screaming':.9 if np.max(np.abs(x))>.1 else 0.}

class ScreamEvaluationTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def wav(self,name,seconds,value=0,rate=16000):
        path=self.root/name
        with wave.open(str(path),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate)
            w.writeframes(np.full(int(seconds*rate),value,dtype='<i2').tobytes())
        return name
    def row(self,id,path,**kwargs):
        return dict(id=id,path=path,source_id=id,group_id=id,split='test',**kwargs)
    def test_clip_event_delay_and_missing_labels(self):
        pos=self.wav('p.wav',1.5,10000);neg=self.wav('n.wav',1.5);unknown=self.wav('u.wav',1.5,10000)
        rows=[self.row('p',pos,label=True,events=[[.4,.6]]),self.row('n',neg,label=False,events=[]),self.row('u',unknown)]
        r=evaluate(rows,MarkerModel(),split='test',base=self.root)
        self.assertEqual(r['clip_metrics']['confusion_matrix'],[[1,0],[0,1]])
        self.assertEqual(r['unknown_clip_labels'],1)
        self.assertAlmostEqual(r['event_metrics']['detection_delay_seconds'][0],1.075)
        self.assertEqual(r['event_metrics']['annotated_recordings'],2)
        self.assertEqual(r['false_alarms']['per_negative_audio_hour'],0)
        self.assertAlmostEqual(r['false_alarms']['unscored_negative_seconds'],.025)
    def test_false_alarm_rate_counts_events_not_false_positive_files(self):
        path=self.wav('negative_marker.wav',12,10000)
        r=evaluate([self.row('n',path,label=False)],MarkerModel(),split='test',base=self.root)
        self.assertEqual(r['clip_metrics']['fp'],1)
        self.assertEqual(r['false_alarms']['events'],2)
        self.assertEqual(r['false_alarms']['per_negative_audio_hour'],600.)
    def test_short_audio_is_unscored_and_not_silently_padded(self):
        path=self.wav('tiny.wav',.005,10000)
        r=evaluate([self.row('p',path,label=True)],MarkerModel(),split='test',base=self.root)
        self.assertEqual(r['clip_metrics']['fn'],1)
        self.assertEqual(r['clips'][0]['status'],'insufficient_audio')
        self.assertIsNone(r['clips'][0]['peak_screaming_score'])
        self.assertEqual(r['clips'][0]['windows'],0)
    def test_optional_pad_runs_one_window_not_repeated_confirmation(self):
        path=self.wav('tiny.wav',.005,10000)
        clip=analyze_wav(MarkerModel(),self.root/path,pad_tail=True)
        self.assertEqual(clip['padded_windows'],1);self.assertFalse(clip['detected'])
    def test_rate_conversion_uses_wav_header(self):
        path=self.wav('48k.wav',1.5,rate=48000)
        clip=analyze_wav(MarkerModel(),self.root/path)
        self.assertEqual(clip['source_rate'],48000);self.assertEqual(clip['windows'],2)
        self.assertGreater(clip['resampler_delay_seconds'],0)
    def test_cooldown_resets_between_independent_clips(self):
        rows=[self.row(str(i),self.wav(f'{i}.wav',1.5,10000),label=True) for i in range(2)]
        self.assertEqual(evaluate(rows,MarkerModel(),split='test',base=self.root)['clip_metrics']['tp'],2)
    def test_group_leakage_checked_before_filtering(self):
        rows=[self.row('a','a.wav'),dict(id='b',path='b.wav',source_id='b',group_id='a',split='validation')]
        with self.assertRaisesRegex(ValueError,'leakage'):evaluate(rows,MarkerModel(),split='test',base=self.root)
    def test_copied_path_cannot_appear_in_different_splits(self):
        rows=[self.row('a','same.wav'),dict(id='b',path='same.wav',source_id='b',split='validation')]
        with self.assertRaisesRegex(ValueError,'leakage'):validate_manifest(rows,self.root)
    def test_missing_recording_returns_explicit_incomplete_report(self):
        r=evaluate([self.row('missing','missing.wav',label=True)],MarkerModel(),split='test',base=self.root)
        self.assertFalse(r['complete']);self.assertEqual(r['evaluated_recordings'],0)
        self.assertEqual(len(r['errors']),1)
    def test_reused_external_is_labeled_and_cannot_be_calibrated(self):
        row=self.row('r',self.wav('r.wav',1.5),label=False);row['split']='reused_external'
        r=evaluate([row],MarkerModel(),split='reused_external',base=self.root)
        self.assertEqual(r['benchmark_status'],'reused_not_independent')
        with self.assertRaisesRegex(ValueError,'Reused'):sweep([dict(split='validation',reused=True)])
    def test_invalid_labels_annotations_and_missing_group_provenance(self):
        for updates in [dict(label='false'),dict(events=[[1,0]]),dict(events=[[0,1],[.5,2]]),dict(source_id=None),dict(label=True,events=[]),dict(events=[['bad',1]])]:
            row=self.row('a','a.wav');row.update(updates)
            with self.assertRaises(ValueError):validate_manifest([row])
    def test_short_negative_exposure_does_not_claim_zero_false_alarms_per_hour(self):
        r=evaluate([self.row('n',self.wav('tiny.wav',.005),label=False)],MarkerModel(),split='test',base=self.root)
        self.assertIsNone(r['false_alarms']['per_negative_audio_hour'])
        self.assertEqual(r['false_alarms']['analyzed_negative_audio_hours'],0)
    def test_unknown_labels_do_not_create_negative_exposure(self):
        r=evaluate([self.row('u',self.wav('u.wav',1.5))],MarkerModel(),split='test',base=self.root)
        self.assertEqual(r['unknown_clip_labels'],1)
        self.assertIsNone(r['clip_metrics']['precision'])
        self.assertIsNone(r['event_metrics']['recall'])
        self.assertEqual(r['false_alarms']['negative_audio_hours'],0)
        self.assertIsNone(r['false_alarms']['per_negative_audio_hour'])

if __name__=='__main__':unittest.main()
