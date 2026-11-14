import unittest,asyncio
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
