"""Metric/manifest tests with declared text fixtures, not recognition benchmarks."""
import json,tempfile,unittest,wave
from pathlib import Path
import numpy as np
from safora.asr.offline import Transcript
from safora.evaluation.asr import evaluate,score,validate_rows


class ASREvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.created=[]
        with wave.open(str(self.root/'fixture.wav'),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(np.zeros(320,dtype='<i2').tobytes())
    def tearDown(self):self.temp.cleanup()
    def row(self,id='a',language='en',**kwargs):
        return dict(id=id,path='fixture.wav',source_id=id,session_id=id,split='test',language=language,**kwargs)
    def factory(self,language):
        owner=self
        class Decoder:
            def transcribe(self,audio):return Transcript('hello' if language=='en' else 'नमस्ते',False,language,-.1)
            def close(self):owner.created.append(('closed',language))
        self.created.append(('loaded',language));return Decoder()
    def test_word_character_arithmetic_unicode_and_empty_reference(self):
        m=score('a b','a c');self.assertEqual(m['wer'],.5);self.assertAlmostEqual(m['cer'],1/3)
        self.assertEqual(score('नमस्ते','नमस्ते')['cer'],0)
        self.assertEqual(score('e\u0301','é')['cer'],0)
        empty=score('','hallucinated');self.assertIsNone(empty['wer']);self.assertTrue(empty['empty_reference_nonempty_hypothesis'])
    def test_separate_language_metrics_and_missing_references(self):
        rows=[self.row('en','en',reference='hello',condition='clean'),self.row('hi','hi',reference='नमस्ते',condition='quiet'),
              self.row('missing','en',condition='noise')]
        report=evaluate(rows,{},split='test',base=self.root,factory=self.factory)
        self.assertEqual(report['by_language']['en']['wer'],0);self.assertEqual(report['by_language']['hi']['cer'],0)
        self.assertEqual(report['by_condition']['noise']['missing_references'],1)
        self.assertIsNone(report['by_condition']['noise']['wer'])
        self.assertNotIn('नमस्ते',json.dumps(report,ensure_ascii=False));self.assertNotIn('hello',json.dumps(report))
        self.assertEqual(self.created,[('loaded','en'),('closed','en'),('loaded','hi'),('closed','hi')])
    def test_hinglish_is_a_stratum_not_automatic_language_identification(self):
        rows=[self.row('mixed','hinglish',reference=None),self.row('selected','hinglish',asr_language='hi',condition='code_switch')]
        report=evaluate(rows,{},split='test',base=self.root,factory=self.factory)
        self.assertFalse(report['complete']);self.assertIsNone(report['by_language']['hinglish']['wer'])
        self.assertEqual(self.created,[('loaded','hi'),('closed','hi')])
        self.assertEqual(report['by_language']['hinglish']['unavailable_recordings'],1)
    def test_unknown_language_is_unavailable_without_a_decoder(self):
        report=evaluate([self.row(language='unknown',reference='known words')],{},split='test',base=self.root,factory=self.factory)
        self.assertEqual(self.created,[]);self.assertIsNone(report['by_language']['unknown']['wer'])
    def test_leakage_checked_before_split_selection(self):
        a=self.row();b=self.row('b');b['split']='validation';b['path']='different.wav';b['session_id']='a'
        with self.assertRaisesRegex(ValueError,'leakage'):evaluate([a,b],{},split='test',base=self.root,factory=self.factory)
    def test_bad_file_and_model_errors_do_not_echo_exception_text(self):
        row=self.row();row['path']='absent.wav'
        report=evaluate([row],{},split='test',base=self.root,factory=self.factory)
        self.assertFalse(report['complete']);self.assertIsNone(report['by_language']['en']['wer'])
        def fail(language):raise RuntimeError('private transcript inside backend exception')
        report=evaluate([self.row()],{},split='test',base=self.root,factory=fail)
        self.assertNotIn('private transcript',json.dumps(report));self.assertFalse(report['complete'])
    def test_non_text_reference_is_not_coerced_and_unknown_mode_rejected(self):
        for updates in [dict(reference=5),dict(asr_language='auto'),dict(session_id=None)]:
            row=self.row();row.update(updates)
            with self.assertRaises(ValueError):validate_rows([row],self.root)
