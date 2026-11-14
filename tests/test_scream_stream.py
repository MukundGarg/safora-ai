"""DSP/queue/control fixtures only; no claims of real scream accuracy."""
import threading
import time
import unittest
import numpy as np
from scipy.signal import upfirdn
from safora.audio.stream import Yamnet, TemporalDetector, WindowStream, waveform
from safora.audio.ingestion import StreamConfig, StreamingResampler, ScreamProcessor, RealtimeScreamMonitor


class TensorRuntime:
    def __init__(self, shape=(15600,), output=(1, 521), dtype=np.float32):
        self.shape, self.output, self.dtype = shape, output, dtype
        self.values = np.zeros(output, dtype=dtype)
        self.values.reshape(-1)[11] = .4
        self.values.reshape(-1)[6] = .95
    def allocate_tensors(self): pass
    def get_input_details(self): return [dict(shape=self.shape, dtype=np.float32, index=3)]
    def get_output_details(self): return [dict(shape=self.output, dtype=self.dtype, index=7)]
    def set_tensor(self, index, value):
        assert index == 3
        self.input = value.copy()
    def invoke(self): pass
    def get_tensor(self, index):
        assert index == 7
        return self.values


class FixedModel:
    window_samples = 16
    def __init__(self, score=.9): self.score, self.calls = score, 0
    def predict(self, x):
        self.calls += 1
        return {'screaming': self.score}


def small_config(**kwargs):
    return StreamConfig(hop_samples=8, **kwargs)


class ModelInputTests(unittest.TestCase):
    def test_metadata_window_batch_and_class_index(self):
        runtime = TensorRuntime((1, 32000), (521,))
        model = Yamnet('unused', interpreter=runtime)
        self.assertEqual(model.window_samples, 32000)
        self.assertEqual(model.window_seconds, 2.)
        scores = model.predict(np.zeros(32000, np.float32))
        self.assertEqual(runtime.input.shape, (1, 32000))
        self.assertAlmostEqual(scores['screaming'], .4, places=6)
        self.assertGreater(scores['shout'], scores['screaming'])
    def test_rejects_wrong_output_metadata(self):
        with self.assertRaises(ValueError): Yamnet('unused', interpreter=TensorRuntime(output=(1, 22)))
        with self.assertRaises(ValueError): Yamnet('unused', interpreter=TensorRuntime(dtype=np.float64))
    def test_rejects_runtime_shape_range_and_nan(self):
        for values in [np.zeros(521, np.float32), np.full((1,521), np.nan, np.float32),
                       np.full((1,521), -1, np.float32), np.full((1,521), 1.1, np.float32)]:
            runtime = TensorRuntime(); model = Yamnet('unused', interpreter=runtime); runtime.values = values
            with self.assertRaises(ValueError): model.predict(np.zeros(15600))
    def test_rejects_multitensor_and_dynamic_models(self):
        runtime = TensorRuntime()
        runtime.get_input_details = lambda: [dict(shape=[15600], shape_signature=[-1], dtype=np.float32, index=0)]
        with self.assertRaises(ValueError): Yamnet('unused', interpreter=runtime)
        runtime = TensorRuntime(); runtime.get_output_details = lambda: [{}, {}]
        with self.assertRaises(ValueError): Yamnet('unused', interpreter=runtime)
    def test_wrong_audio_length_and_closed_model(self):
        model = Yamnet('unused', interpreter=TensorRuntime())
        with self.assertRaises(ValueError): model.predict(np.zeros(15599))
        model.close()
        with self.assertRaises(RuntimeError): model.predict(np.zeros(15600))
    def test_pcm16_saturation_and_stereo_validation_before_mix(self):
        np.testing.assert_allclose(waveform(np.array([-32768,32767], np.int16)), [-1,32767/32768])
        with self.assertRaises(ValueError): waveform(np.array([[2., -2.]]))
        np.testing.assert_array_equal(waveform([2., -2.], clip=True), [1, -1])
        with self.assertRaises(ValueError): waveform([1.001])
    def test_invalid_types_rates_and_channels(self):
        for samples in [np.array([True]), np.array([1j]), np.zeros((2,9)), np.array(['bad'])]:
            with self.assertRaises(ValueError): waveform(samples)
        for rate in [0, -1, 16000.5, True]:
            with self.assertRaises(ValueError): waveform([0.], rate)


class FramingTests(unittest.TestCase):
    def test_streaming_resampling_matches_whole_causal_reference(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0, .1, 10003).astype(np.float32)
        for rate in [8000, 16000, 22050, 44100, 48000, 96000]:
            resampler = StreamingResampler(rate)
            expected = np.clip(upfirdn(resampler.h, x, resampler.up, resampler.down)
                               [:int(np.ceil(len(x)*16000/rate))], -1, 1)
            chunks = [resampler.feed(x[i:i+53]) for i in range(0,len(x),53)]
            np.testing.assert_allclose(np.concatenate(chunks), expected, atol=2e-6, rtol=2e-6)
            self.assertLessEqual(len(resampler.history), (len(resampler.h)-1)//resampler.up + resampler.down + 1)
    def test_downsample_filter_rejects_above_nyquist_tone(self):
        t = np.arange(48000)/48000
        high = StreamingResampler(48000).feed(np.sin(2*np.pi*12000*t).astype(np.float32))
        low = StreamingResampler(48000).feed(np.sin(2*np.pi*2000*t).astype(np.float32))
        self.assertLess(float(np.sqrt(np.mean(high[100:]**2))), .01)
        self.assertGreater(float(np.sqrt(np.mean(low[100:]**2))), .65)
    def test_unusual_rate_rejected_without_unbounded_filter(self):
        with self.assertRaises(ValueError): StreamingResampler(44101)
    def test_short_and_exact_length_tail_policy(self):
        stream = WindowStream(16,8)
        self.assertEqual(list(stream.feed(np.zeros(3))), [])
        t, padded = stream.tail(); self.assertEqual(t,0);self.assertEqual(padded.shape,(16,))
        self.assertIsNone(stream.tail())
        self.assertEqual(len(list(stream.feed(np.zeros(16)))), 1)
        self.assertIsNone(stream.tail())  # Retained overlap alone is not fresh tail audio.
    def test_microphone_timestamp_falls_back_when_driver_clock_is_missing(self):
        import importlib.util
        from pathlib import Path
        from types import SimpleNamespace
        spec=importlib.util.spec_from_file_location('microphone_cli',Path(__file__).resolve().parents[1]/'scripts/monitor_screams.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        valid=SimpleNamespace(inputBufferAdcTime=4.,currentTime=4.1)
        self.assertAlmostEqual(module.capture_timestamp(valid,10.),9.9)
        for adc,current in [(0.,4.),(5.,4.),(float('nan'),4.)]:
            self.assertIsNone(module.capture_timestamp(SimpleNamespace(inputBufferAdcTime=adc,currentTime=current),10.))
    def test_absolute_clock_after_gap(self):
        stream=WindowStream(16,8); stream.clear(32000)
        self.assertEqual(list(stream.feed(np.zeros(16)))[0][0],2.)
    def test_processor_short_live_audio_does_not_pad(self):
        model=FixedModel();p=ScreamProcessor(model,small_config())
        self.assertEqual(p.feed(np.zeros(3)),[]);self.assertEqual(p.finish(),[]);self.assertEqual(model.calls,0)
    def test_explicit_eof_padding_is_once_and_uses_real_end(self):
        p=ScreamProcessor(FixedModel(),small_config(confirmation_hits=1))
        p.feed(np.zeros(3));observations=p.finish(True)
        self.assertTrue(observations[0].padded);self.assertEqual(observations[0].end,3/16000)
        self.assertEqual(p.finish(True),[])
        with self.assertRaises(RuntimeError):p.feed(np.zeros(3))
    def test_gap_forgets_votes_but_preserves_cooldown(self):
        d=TemporalDetector(hits=1)
        self.assertIsNotNone(d.update(1,.3));d.reset_continuity()
        self.assertIsNone(d.update(2,.9));self.assertIsNotNone(d.update(11,.9))
        with self.assertRaises(ValueError):d.update(10,.9)
    def test_deterministic_decisions_and_threshold_equality(self):
        values=[.3,.1,.3,.9,.9,.9,.9]
        def run():
            d=TemporalDetector(cooldown=2)
            return [i*.5 for i,x in enumerate(values) if d.update(i*.5,x)]
        self.assertEqual(run(),[1.,3.]);self.assertEqual(run(),run())
    def test_config_rejects_unbounded_queue_and_unknown_version(self):
        for kwargs in [dict(queue_capacity=0),dict(queue_capacity=9),dict(version=2),dict(block_seconds=2),dict(threshold='0.3'),dict(confirmation_hits=True)]:
            with self.assertRaises(ValueError):StreamConfig(**kwargs)


class IngestionTests(unittest.TestCase):
    def test_drop_newest_clock_and_gap_confirmation(self):
        events=[];m=RealtimeScreamMonitor(FixedModel(),events.append,config=small_config(queue_capacity=1))
        self.assertTrue(m.submit(np.zeros(16)))
        self.assertFalse(m.submit(np.zeros(16)))  # sample16..31 lost, not moved before queued sample0.
        m.start()
        try:
            self.assertTrue(m.wait_idle(2));self.assertEqual(events,[])
            m.submit(np.zeros(16));self.assertTrue(m.wait_idle(2));self.assertEqual(events,[])
            m.submit(np.zeros(8));self.assertTrue(m.wait_idle(2))
            self.assertEqual(events[0].time,56/16000)
            self.assertTrue(events[0].monitoring_only)
            self.assertEqual(m.snapshot()['dropped_source_samples'],16)
            self.assertEqual(m.snapshot()['gaps'],1)
        finally:self.assertTrue(m.stop())
    def test_slow_native_work_does_not_wait_in_capture_and_stop_suppresses_late_alert(self):
        entered,release=threading.Event(),threading.Event()
        class SlowModel(FixedModel):
            def predict(self,x):
                entered.set();release.wait(3);return super().predict(x)
        events=[];m=RealtimeScreamMonitor(SlowModel(),events.append,config=small_config(queue_capacity=1,confirmation_hits=1)).start()
        try:
            m.submit(np.zeros(16));self.assertTrue(entered.wait(1))
            self.assertTrue(m.submit(np.zeros(16)));self.assertFalse(m.submit(np.zeros(16)))
            self.assertFalse(release.is_set())  # Both capture calls completed while inference is still blocked.
            self.assertFalse(m.stop(.01));self.assertFalse(m.submit(np.zeros(16)))
        finally:release.set();self.assertTrue(m.stop(2))
        self.assertEqual(events,[]);self.assertTrue(m.wait_idle(1))
    def test_submit_owns_copy_and_validates_clock_rate_and_size(self):
        m=RealtimeScreamMonitor(FixedModel(),config=small_config())
        x=np.ones(16,np.float32);m.submit(x);x.fill(0)
        np.testing.assert_array_equal(m.queue.queue[0].samples,np.ones(16))
        for kwargs in [dict(start_sample=0),dict(sample_rate=48000)]:
            with self.assertRaises(ValueError):m.submit(np.zeros(16),**kwargs)
        with self.assertRaises(ValueError):m.submit(np.zeros(1601))
        m.stop();self.assertTrue(m.wait_idle(1))
    def test_bad_block_reports_error_then_recovers_without_false_confirmation(self):
        status=[];events=[];m=RealtimeScreamMonitor(FixedModel(),events.append,status.append,small_config()).start()
        try:
            m.submit(np.full(16,np.nan));self.assertTrue(m.wait_idle(2))
            m.submit(np.zeros(16));self.assertTrue(m.wait_idle(2));self.assertEqual(events,[])
            m.submit(np.zeros(8));self.assertTrue(m.wait_idle(2));self.assertEqual(len(events),1)
            self.assertEqual(m.snapshot()['inference_errors'],1)
            self.assertEqual(status[0]['error'],'ValueError')
        finally:m.stop()
    def test_receiver_error_does_not_kill_worker(self):
        def fail(event):raise RuntimeError('receiver failed')
        m=RealtimeScreamMonitor(FixedModel(),fail,config=small_config(confirmation_hits=1)).start()
        try:
            m.submit(np.zeros(16));self.assertTrue(m.wait_idle(2))
            self.assertEqual(m.snapshot()['callback_errors'],1)
            self.assertTrue(m.snapshot()['worker_alive'])
        finally:m.stop()
    def test_callback_can_request_stop_without_joining_its_own_thread(self):
        outcomes=[]
        m=RealtimeScreamMonitor(FixedModel(),lambda event:outcomes.append(m.stop()),
                               config=small_config(confirmation_hits=1)).start()
        try:
            m.submit(np.zeros(16));self.assertTrue(m.wait_idle(2))
        finally:self.assertTrue(m.stop(2))
        self.assertEqual(outcomes,[False]);self.assertEqual(m.snapshot()['callback_errors'],0)
    def test_capture_latency_and_monitoring_fields_are_exposed(self):
        p=ScreamProcessor(FixedModel(),small_config(confirmation_hits=1))
        observation=p.feed(np.zeros(16),capture_time=time.monotonic()-.01)[0]
        event=observation.event
        self.assertEqual(event.window_start,0.)
        self.assertGreaterEqual(event.capture_to_decision_seconds,.009)
        self.assertGreaterEqual(event.inference_seconds,0.)
        self.assertTrue(event.monitoring_only)
        self.assertFalse(hasattr(event,'level'))
    def test_processor_enforces_channels_before_conversion(self):
        p=ScreamProcessor(FixedModel(),small_config(max_channels=1))
        with self.assertRaises(ValueError):p.feed(np.zeros((16,2)))
    def test_explicit_driver_discontinuity_cannot_combine_votes(self):
        p=ScreamProcessor(FixedModel(),small_config())
        self.assertIsNone(p.feed(np.zeros(16))[0].event)
        self.assertIsNone(p.feed(np.zeros(16),discontinuity=True)[0].event)
        self.assertEqual(p.gaps,1)


if __name__=='__main__':unittest.main()
