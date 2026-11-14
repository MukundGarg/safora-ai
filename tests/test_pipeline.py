import unittest,asyncio,threading
import numpy as np
from safora.pipeline import Pipeline
from safora.decision import DecisionEngine
from safora.classifier.risk import ContextBaseline
from safora.asr.offline import Transcript
class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_speech_runs_without_acoustic_alarm(self):
        class Model:
            def predict(self,x):return {'screaming':0.}
        class ASR:
            def transcribe(self,x):return Transcript('I will hurt you',False,'en',-.1)
        events=[];p=Pipeline(Model(),ASR(),ContextBaseline(),DecisionEngine(),events.append)
        capture=asyncio.create_task(p.capture_worker());speech=asyncio.create_task(p.speech_worker())
        try:
            for _ in range(8):
                self.assertTrue(p.submit(np.full(16000,.02,dtype=np.float32)))
                await p.audio.join()
            await p.speech.join()
            self.assertEqual(events[0].level,1)
            self.assertEqual(len(events),1)
        finally:
            capture.cancel();speech.cancel();await asyncio.gather(capture,speech,return_exceptions=True)
    async def test_bounded_capture(self):
        p=Pipeline(None,None,None,None,lambda e:None)
        for _ in range(8):self.assertTrue(p.submit(np.zeros(100)))
        self.assertFalse(p.submit(np.zeros(100)))

    async def test_legacy_asr_does_not_hold_acoustic_inference_lock(self):
        entered,release=threading.Event(),threading.Event()
        class Model:
            calls=0
            def predict(self,x):self.calls+=1;return {'screaming':0.}
        class ASR:
            def transcribe(self,x):entered.set();release.wait(3);return Transcript('',True,'en',-10)
        model=Model();p=Pipeline(model,ASR(),ContextBaseline(),DecisionEngine(),lambda e:None)
        capture=asyncio.create_task(p.capture_worker());speech=asyncio.create_task(p.speech_worker())
        try:
            for _ in range(8):p.submit(np.full(16000,.02,dtype=np.float32));await p.audio.join()
            self.assertTrue(await asyncio.to_thread(entered.wait,1))
            before=model.calls;p.submit(np.full(16000,.02,dtype=np.float32))
            await asyncio.wait_for(p.audio.join(),1)
            self.assertGreater(model.calls,before);self.assertFalse(release.is_set())
        finally:
            release.set();capture.cancel();speech.cancel()
            await asyncio.gather(capture,speech,return_exceptions=True)

    async def test_history_retains_asr_uncertainty_for_risk_classifier(self):
        from safora.classifier.text import Turn
        class ASR:
            def transcribe(self,x):return Transcript('I said no',True,'en',-10)
        class Classifier:
            def classify(self,text,history,uncertain):
                self.uncertain=uncertain;self.history=history
                from safora.classifier.risk import Risk
                return Risk({'neutral_or_uncertain'})
        classifier=Classifier();p=Pipeline(None,ASR(),classifier,DecisionEngine(),lambda e:None)
        worker=asyncio.create_task(p.speech_worker())
        try:
            for _ in range(2):await p.speech.put(np.zeros(16));await p.speech.join()
            self.assertTrue(classifier.uncertain);self.assertIsInstance(classifier.history[0],Turn)
            self.assertTrue(classifier.history[0].asr_uncertain)
        finally:worker.cancel();await asyncio.gather(worker,return_exceptions=True)
