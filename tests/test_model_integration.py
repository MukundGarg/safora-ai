import os,unittest
import numpy as np
from safora.audio.stream import Yamnet,WindowStream,TemporalDetector

@unittest.skipUnless(os.environ.get('SAFORA_YAMNET_MODEL'),'Set SAFORA_YAMNET_MODEL for real model tests')
class RealYamnetTests(unittest.TestCase):
    def test_real_monitor_48khz_stereo_silence(self):
        from safora.audio.ingestion import RealtimeScreamMonitor,StreamConfig
        events=[]
        with Yamnet(os.environ['SAFORA_YAMNET_MODEL']) as model:
            monitor=RealtimeScreamMonitor(model,events.append,config=StreamConfig(source_rate=48000)).start()
            try:
                for _ in range(10):
                    self.assertTrue(monitor.submit(np.zeros((4800,2),np.int16)))
                    self.assertTrue(monitor.wait_idle(5))
                self.assertEqual(monitor.snapshot()['windows'],1)
                self.assertEqual(monitor.snapshot()['inference_errors'],0)
                self.assertEqual(events,[])
            finally:self.assertTrue(monitor.stop())
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
