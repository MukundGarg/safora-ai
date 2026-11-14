import threading,unittest
import numpy as np
from safora.asr.offline import Transcript
from safora.asr.streaming import ASRConfig,StreamingASR
from safora.audio.ingestion import RealtimeScreamMonitor,StreamConfig


class Decoder:
    def __init__(self,language,entered=None,release=None):
        self.language=language;self.entered=entered;self.release=release;self.closed=False;self.resets=0;self.count=0
    def begin(self):self.count=0
    def accept(self,samples):
        if self.entered:self.entered.set();self.release.wait(3)
        if not np.isfinite(samples).all():raise ValueError('private native text must not enter logs')
        self.count+=len(samples)
        return Transcript('provisional',True,self.language,-1,False,'partial')
    def finish(self):return Transcript(self.language,False,self.language,-.1)
    def reset(self):self.resets+=1;self.count=0
    def close(self):self.closed=True


class ASRStreamingTests(unittest.TestCase):
    def factory(self,created,**kwargs):
        def make(language):
            decoder=Decoder(language,**kwargs);created.append(decoder);return decoder
        return make
    def test_final_timestamped_result_and_close(self):
        created=[];results=[];controller=StreamingASR({},language='en',on_result=results.append,factory=self.factory(created)).start()
        try:
            self.assertTrue(controller.submit(np.zeros(160),start_sample=3200));self.assertTrue(controller.wait_idle(2))
            self.assertEqual(results[0].start,.2);self.assertEqual(results[0].end,.21)
            self.assertEqual(results[0].transcript.text,'en')
        finally:self.assertTrue(controller.stop())
        self.assertTrue(created[0].closed)
    def test_unknown_language_does_not_guess_or_load_a_model(self):
        created=[];controller=StreamingASR({},factory=self.factory(created)).start()
        try:
            self.assertFalse(controller.submit(np.zeros(16)));self.assertEqual(created,[])
            self.assertEqual(controller.snapshot()['coverage'],'unknown_language')
            with self.assertRaises(ValueError):controller.set_language('hinglish')
            with self.assertRaises(ValueError):controller.set_language('auto')
        finally:controller.stop()
    def test_language_switch_discards_inflight_and_queued_old_language(self):
        entered,release=threading.Event(),threading.Event();created=[];results=[]
        def make(language):
            d=Decoder(language,entered if language=='en' else None,release);created.append(d);return d
        controller=StreamingASR({},language='en',on_result=results.append,factory=make).start()
        try:
            controller.submit(np.zeros(16));self.assertTrue(entered.wait(1));controller.submit(np.zeros(16))
            generation=controller.set_language('hi');release.set();self.assertTrue(controller.wait_idle(2))
            self.assertEqual(results,[]);self.assertTrue(created[0].closed)
            controller.submit(np.zeros(16));self.assertTrue(controller.wait_idle(2))
            self.assertEqual(results[0].language,'hi');self.assertEqual(results[0].generation,generation)
        finally:release.set();controller.stop()
    def test_drop_newest_and_gap_discard_partial_before_flush(self):
        created=[];results=[];controller=StreamingASR({},language='en',config=ASRConfig(queue_capacity=1),on_result=results.append,factory=self.factory(created))
        self.assertTrue(controller.submit(np.zeros(16),final=False));self.assertFalse(controller.submit(np.zeros(16),final=False))
        controller.start()
        try:
            self.assertTrue(controller.wait_idle(2));controller.submit(np.zeros(16),final=False)
            self.assertTrue(controller.wait_idle(2));self.assertEqual(created[0].count,16)
            self.assertTrue(controller.flush());self.assertTrue(controller.wait_idle(2))
            self.assertEqual(results[0].start_sample,32);self.assertEqual(controller.snapshot()['dropped_samples'],16)
            self.assertEqual(controller.snapshot()['gaps'],1)
        finally:controller.stop()
    def test_slow_asr_cannot_block_task1_capture(self):
        entered,release=threading.Event(),threading.Event();created=[]
        controller=StreamingASR({},language='en',factory=self.factory(created,entered=entered,release=release)).start()
        class Acoustic:
            window_samples=16
            def predict(self,x):return {'screaming':0.}
        monitor=RealtimeScreamMonitor(Acoustic(),config=StreamConfig(hop_samples=8)).start()
        try:
            controller.submit(np.zeros(16));self.assertTrue(entered.wait(1))
            monitor.submit(np.zeros(16));self.assertTrue(monitor.wait_idle(1))
            self.assertEqual(monitor.snapshot()['windows'],1);self.assertFalse(release.is_set())
            self.assertFalse(controller.stop(.01))
        finally:release.set();controller.stop(2);monitor.stop()
    def test_reset_restart_and_low_memory_suspension(self):
        created=[];results=[];controller=StreamingASR({},language='en',on_result=results.append,factory=self.factory(created)).start()
        try:
            controller.submit(np.zeros(16),final=False);self.assertTrue(controller.wait_idle(2))
            controller.reset();self.assertTrue(controller.wait_idle(2));self.assertEqual(results,[])
            self.assertFalse(created[0].closed)
            controller.restart();self.assertTrue(controller.wait_idle(2));self.assertTrue(created[0].closed)
            controller.submit(np.zeros(16));self.assertTrue(controller.wait_idle(2));self.assertEqual(len(created),2)
            controller.set_language(None);self.assertTrue(controller.wait_idle(2));self.assertTrue(created[-1].closed)
        finally:controller.stop()
    def test_chunk_segment_bounds_and_unknown_callback_failure(self):
        results=[];created=[];c=StreamingASR({},language='en',config=ASRConfig(max_chunk_seconds=.1,max_segment_seconds=.15),on_result=results.append,factory=self.factory(created)).start()
        try:
            for _ in range(2):c.submit(np.zeros(1600),final=False);self.assertTrue(c.wait_idle(2))
            self.assertEqual(results[0].transcript.reason,'segment_limit');self.assertTrue(results[0].transcript.uncertain)
            with self.assertRaises(ValueError):c.submit(np.zeros(1601))
            with self.assertRaises(ValueError):c.submit(np.zeros(16),sample_rate=48000)
            with self.assertRaises(ValueError):c.submit(np.zeros((1,1)))
        finally:c.stop()
    def test_status_does_not_log_transcripts_on_errors_and_recovery(self):
        status=[];created=[];c=StreamingASR({},language='en',on_status=status.append,factory=self.factory(created)).start()
        try:
            c.submit(np.full(16,np.nan));self.assertTrue(c.wait_idle(2))
            self.assertNotIn('private native text',repr(status));self.assertEqual(c.snapshot()['errors'],1)
            c.submit(np.zeros(16));self.assertTrue(c.wait_idle(2))
            self.assertEqual(c.snapshot()['coverage'],'enabled_with_possible_gaps')
        finally:c.stop()

    def test_broken_native_reset_releases_backend_and_worker_recovers(self):
        created=[]
        class Broken(Decoder):
            def reset(self):raise RuntimeError('private reset failure')
        def make(language):
            d=Broken(language) if not created else Decoder(language)
            created.append(d);return d
        c=StreamingASR({},language='en',factory=make).start()
        try:
            c.submit(np.full(16,np.nan));self.assertTrue(c.wait_idle(2));self.assertTrue(created[0].closed)
            c.submit(np.zeros(16));self.assertTrue(c.wait_idle(2))
            self.assertEqual(c.snapshot()['results'],1);self.assertTrue(c.snapshot()['worker_alive'])
        finally:c.stop()
