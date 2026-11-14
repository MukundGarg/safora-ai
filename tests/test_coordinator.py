"""Deterministic orchestration contracts, not safety/ASR accuracy."""
import threading,unittest
import numpy as np
from safora.coordinator import MLCoordinator
from safora.asr.offline import Transcript

class Acoustic:
    window_samples=15600
    def __init__(self,score=.0):self.score=score;self.calls=0;self.closed=False
    def predict(self,audio):self.calls+=1;return {'screaming':self.score}
    def close(self):self.closed=True

class Speech:
    def __init__(self,language,text='I will hurt you',entered=None,release=None):
        self.language=language;self.text=text;self.entered=entered;self.release=release;self.closed=False
    def begin(self):pass
    def accept(self,audio):
        if self.entered:self.entered.set();self.release.wait(3)
        return Transcript('',True,self.language,-10,False)
    def finish(self):return Transcript(self.text,False,self.language,-.1)
    def reset(self):pass
    def close(self):self.closed=True

class CoordinatorTests(unittest.TestCase):
    def feed(self,c,blocks=10):
        for _ in range(blocks):c.submit(np.full(1600,.02,np.float32));self.assertTrue(c.wait_idle(3))
    def test_silence_no_cue_and_manual_independent_after_denial_optout(self):
        events=[];c=MLCoordinator(lambda:Acoustic(),lambda l:Speech(l),on_cue=events.append)
        self.assertFalse(c.start(permission=False));c.manual_sos();self.assertEqual(events[-1].kind,'manual_sos_requested')
        self.assertTrue(c.start());
        try:
            for _ in range(10):c.submit(np.zeros(1600));c.wait_idle(3)
            self.assertEqual(len(events),1)
        finally:self.assertTrue(c.opt_out())
        c.manual_sos();self.assertFalse(events[-1].external_dispatch)
        self.assertFalse(c.submit(np.zeros(1600)))
    def test_acoustic_is_only_uncertain_confirmable_checkin_cooldown(self):
        events=[];c=MLCoordinator(lambda:Acoustic(.9),lambda l:Speech(l),on_cue=events.append)
        c.start()
        try:self.feed(c,25);self.assertEqual(len(events),1);self.assertTrue(events[0].uncertain);self.assertEqual(events[0].kind,'check_in_requested')
        finally:c.stop()
    def test_speech_risk_language_switch_gap_and_resume_clear_context(self):
        created=[];events=[]
        def factory(language):d=Speech(language);created.append(d);return d
        c=MLCoordinator(lambda:Acoustic(),factory,on_cue=events.append,speech_enabled=True,segment_seconds=.1,cooldown_seconds=0)
        c.start()
        try:
            self.feed(c,1);self.assertEqual(events[0].source,'UNVALIDATED_HEURISTIC');self.assertEqual(c.snapshot()['context_turns'],1)
            c.configure_speech(True,'hi');self.assertEqual(c.snapshot()['context_turns'],0)
            self.feed(c,1);self.assertEqual(created[-1].language,'hi')
            c.submit(np.full(1600,.02),start_sample=6400);self.assertTrue(c.wait_idle(3))
            self.assertLessEqual(c.snapshot()['context_turns'],1)
            c.configure_speech(False,None);self.assertEqual(c.snapshot()['context_turns'],0)
        finally:self.assertTrue(c.stop())
        session=c.session;c.start();self.assertGreater(c.session,session);self.assertEqual(c.snapshot()['context_turns'],0);c.stop()
    def test_slow_asr_does_not_hold_capture_or_acoustic_worker_and_stale_suppressed(self):
        entered,release=threading.Event(),threading.Event();model=Acoustic();events=[]
        c=MLCoordinator(lambda:model,lambda l:Speech(l,entered=entered,release=release),on_cue=events.append,speech_enabled=True,segment_seconds=.1)
        c.start()
        try:
            c.submit(np.full(1600,.02));self.assertTrue(entered.wait(1))
            for _ in range(12):c.submit(np.full(1600,.02));c.acoustic.wait_idle(2)
            self.assertGreater(model.calls,0);self.assertFalse(release.is_set())
            c.configure_speech(False,None);release.set();self.assertTrue(c.wait_idle(3));self.assertEqual(events,[])
        finally:release.set();c.stop()
    def test_queue_loss_reset_missing_models_and_recovery(self):
        def fail():raise RuntimeError('private failure')
        events=[];c=MLCoordinator(fail,lambda l:Speech(l),on_cue=events.append)
        self.assertTrue(c.start());self.assertEqual(c.snapshot()['coverage'],'acoustic_unavailable')
        c.manual_sos();self.assertEqual(len(events),1);self.assertTrue(c.permission_lost())
        c.acoustic_factory=lambda:Acoustic();self.assertTrue(c.start());self.feed(c,1);self.assertTrue(c.stop())
    def test_source_contract_and_unknown_language(self):
        c=MLCoordinator(lambda:Acoustic(),lambda l:Speech(l),speech_enabled=True,language=None)
        c.start()
        try:
            for kwargs in [dict(sample_rate=48000),dict(start_sample=-1)]:
                with self.assertRaises(ValueError):c.submit(np.zeros(1600),**kwargs)
            with self.assertRaises(ValueError):c.submit(np.zeros(1601))
            self.feed(c,2);self.assertFalse(c.snapshot()['asr']['model_loaded'])
        finally:c.stop()

    def test_bounded_speech_ingestion_overflow_does_not_block_capture(self):
        entered,release=threading.Event(),threading.Event()
        class VAD:
            def speech(self,audio):entered.set();release.wait(3);return True
        c=MLCoordinator(lambda:Acoustic(),lambda l:Speech(l),speech_enabled=True,segment_seconds=.1,queue_capacity=1,vad=VAD())
        c.start()
        try:
            c.submit(np.full(1600,.02));self.assertTrue(entered.wait(1))
            for _ in range(10):c.submit(np.full(1600,.02));c.acoustic.wait_idle(2)
            self.assertGreater(c.snapshot()['speech_input_drops'],0)
            self.assertEqual(c.snapshot()['context_turns'],0)
        finally:release.set();self.assertTrue(c.stop())
    def test_stop_timeout_suppresses_late_outputs_and_retains_native_owner(self):
        entered,release=threading.Event(),threading.Event();created=[];events=[]
        def factory(language):d=Speech(language,entered=entered,release=release);created.append(d);return d
        c=MLCoordinator(lambda:Acoustic(),factory,on_cue=events.append,speech_enabled=True,segment_seconds=.1)
        c.start()
        try:
            c.submit(np.full(1600,.02));self.assertTrue(entered.wait(1))
            self.assertFalse(c.stop(timeout=.01));self.assertFalse(created[0].closed)
            with self.assertRaises(RuntimeError):c.start()
            c.manual_sos();self.assertEqual(events[0].kind,'manual_sos_requested')
        finally:
            release.set();c._retiring.join(3);self.assertTrue(c.stop())
        self.assertTrue(created[0].closed);self.assertEqual(len(events),1)

    def test_shared_rule_contract_fixtures_match_python(self):
        import csv
        from pathlib import Path
        from safora.classifier.risk import ContextBaseline
        from safora.classifier.text import Turn
        with (Path(__file__).resolve().parents[1]/'configs/risk-rule-contract.tsv').open() as file:
            for row in csv.DictReader(file,delimiter='\t',quoting=csv.QUOTE_NONE):
                history=[] if row['history']=='-' else [Turn(row['history'],row['history_uncertain']=='true')]
                self.assertEqual(ContextBaseline().classify(row['text'],history).labels,set(row['expected_labels'].split(',')))
