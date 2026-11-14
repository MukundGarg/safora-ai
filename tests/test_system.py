import unittest
import numpy as np
from safora.audio.stream import waveform,WindowStream,TemporalDetector,Yamnet
from safora.classifier.risk import ContextBaseline,Risk
from safora.decision import DecisionEngine
from safora.resources import Controller,Pressure
from safora.evaluation.metrics import event_metrics,error_rates,validate_splits,add_noise
from safora.evaluation.calibrate import sweep
from safora.asr.offline import EnergyVAD,WhisperASR

class AudioTests(unittest.TestCase):
    def test_resample_pcm_and_stereo(self):
        x=waveform(np.full((48000,2),16384,dtype=np.int16),48000)
        self.assertEqual(len(x),16000);self.assertAlmostEqual(float(x[8000]),.5,places=3)
    def test_invalid_waveform(self):
        for x in [[float('nan')],[1.2],np.zeros((2,2,2))]:
            with self.assertRaises(ValueError):waveform(x)
    def test_chunk_invariant_windows_and_clock(self):
        x=np.linspace(-1,1,60000,dtype=np.float32)
        expected=list(WindowStream().feed(x));s=WindowStream();actual=[]
        for i in range(0,len(x),317):actual.extend(s.feed(x[i:i+317]))
        self.assertEqual(len(expected),len(actual))
        for (t,a),(u,b) in zip(expected,actual):self.assertEqual(t,u);np.testing.assert_array_equal(a,b)
        self.assertEqual(actual[1][0],.5)
        self.assertLessEqual(s.size,15600)
    def test_confirmation_and_cooldown(self):
        d=TemporalDetector()
        self.assertIsNone(d.update(0,.4));self.assertIsNone(d.update(.5,.1));self.assertIsNotNone(d.update(1,.668))
        for i in range(2,11):self.assertIsNone(d.update(i,.8516))
        self.assertIsNotNone(d.update(11,.8516))
    def test_supplied_negative_maxima(self):
        for maximum in [0,.0078]:
            d=TemporalDetector()
            self.assertTrue(all(d.update(i*.5,maximum) is None for i in range(122)))
    def test_peak_alone_does_not_confirm(self):
        d=TemporalDetector()
        self.assertIsNone(d.update(0,.668));self.assertIsNone(d.update(.5,0));self.assertIsNone(d.update(1,0))
    def test_model_contract_with_fake_runtime(self):
        class Fake:
            def allocate_tensors(self):pass
            def get_input_details(self):return [dict(shape=[15600],dtype=np.float32,index=0)]
            def get_output_details(self):return [dict(shape=[1,521],dtype=np.float32,index=1)]
            def set_tensor(self,i,x):self.x=x
            def invoke(self):pass
            def get_tensor(self,i):
                y=np.zeros((1,521),np.float32);y[0,11]=.4;return y
        m=Yamnet('unused',interpreter=Fake());self.assertAlmostEqual(m.predict(np.zeros(15600))['screaming'],.4,places=6)

class SafetyTests(unittest.TestCase):
    def test_ordinary_ambiguity_is_not_emergency(self):
        c=ContextBaseline()
        for text in ["Let's go somewhere for coffee.","Can I touch you there?","Don't tell anyone what happened.","Come somewhere alone with me."]:
            self.assertEqual(c.classify(text).labels,{'neutral_or_uncertain'})
    def test_context_and_hindi(self):
        c=ContextBaseline()
        self.assertIn('refusal_ignored',c.classify('I do it anyway', ['I said no']).labels)
        self.assertIn('explicit_threat',c.classify('jaan se maar dunga').labels)
        self.assertNotIn('refusal_ignored',c.classify('Would you like coffee?', ['I said no']).labels)
    def test_uncertainty_and_manual(self):
        e=DecisionEngine();c=ContextBaseline()
        out=e.update(1,risk=c.classify('If you leave, I will hurt you.'))
        self.assertEqual(out.level,1)
        out=e.update(1.1,manual=True);self.assertTrue(out.manual);self.assertEqual(out.level,2)
    def test_higher_priority_bypasses_lower_incident_cooldown(self):
        from safora.audio.stream import AcousticEvent
        e=DecisionEngine()
        self.assertEqual(e.update(1,risk=Risk({'distress_expression'})).level,1)
        self.assertEqual(e.update(1.1,acoustic=AcousticEvent(1.1,.6)).level,2)
    def test_validated_risk_and_duplicate(self):
        e=DecisionEngine();r=Risk({'explicit_threat'},False,'test')
        self.assertEqual(e.update(0,risk=r).level,2)
        self.assertIsNone(e.update(20,risk=r));e.resolve()
        self.assertIsNotNone(e.update(21,risk=r))
    def test_resource_hysteresis(self):
        c=Controller('enhanced')
        for _ in range(3):out=c.update(Pressure(900))
        self.assertEqual(out['mode'],'enhanced')
        self.assertFalse(c.update(Pressure(900,thermal=3))['conversation_enabled'])
    def test_quiet_speech_not_gated_on_yamnet(self):
        self.assertTrue(EnergyVAD().speech(np.full(16000,.02)))
    def test_missing_asr_asset_fails_offline(self):
        with self.assertRaises(FileNotFoundError):WhisperASR('/missing/model')

class EvaluationTests(unittest.TestCase):
    def test_event_matching_duplicates_and_delay(self):
        m=event_metrics([1.5,1.7,10],[(1,2),(8,8.5)],1)
        self.assertEqual((m['tp'],m['fp'],m['fn']),(1,2,1));self.assertEqual(m['delays_seconds'],[.5])
    def test_wer_cer(self):
        self.assertEqual(error_rates('hello there','hello there')['wer'],0)
        self.assertEqual(error_rates('a b','a c')['wer'],.5)
    def test_source_leakage(self):
        rows=[dict(source_id='a',session_id='b',split='train'),dict(source_id='a',session_id='c',split='test')]
        with self.assertRaises(ValueError):validate_splits(rows)
    def test_calibration_rejects_test_data(self):
        with self.assertRaises(ValueError):sweep([{'split':'test'}])
    def test_noise_reproducibility(self):
        a=np.full(1000,.1,dtype=np.float32)
        np.testing.assert_array_equal(add_noise(a,5),add_noise(a,5))
        self.assertLessEqual(float(abs(add_noise(a,-10)).max()),1.)
if __name__=='__main__':unittest.main()
