import os,unittest
import numpy as np
from safora.audio.stream import Yamnet,WindowStream,TemporalDetector

@unittest.skipUnless(os.environ.get('SAFORA_YAMNET_MODEL'),'Set SAFORA_YAMNET_MODEL for real model tests')
class RealYamnetTests(unittest.TestCase):
    def test_real_model_silence_stream(self):
        m=Yamnet(os.environ['SAFORA_YAMNET_MODEL'])
        detector=TemporalDetector()
        count=0
        for time,x in WindowStream().feed(np.zeros(16000*5,dtype=np.float32)):
            scores=m.predict(x);count+=1
            self.assertEqual(len(scores),6)
            self.assertTrue(all(0<=score<=1 for score in scores.values()))
            self.assertIsNone(detector.update(time+.975,scores['screaming']))
        self.assertEqual(count,9)

@unittest.skipUnless(os.environ.get('SAFORA_WHISPER_MODEL'),'Set SAFORA_WHISPER_MODEL for real ASR test')
class RealWhisperTests(unittest.TestCase):
    def test_silence_is_uncertain(self):
        from safora.asr.offline import WhisperASR
        m=WhisperASR(os.environ['SAFORA_WHISPER_MODEL'])
        out=m.transcribe(np.zeros(16000*8,np.float32))
        self.assertEqual(out.text,'');self.assertTrue(out.uncertain)
