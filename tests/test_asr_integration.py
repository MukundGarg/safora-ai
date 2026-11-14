"""Optional real Vosk CPU integration; no representative speech accuracy claim."""
import json,os,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from safora.asr.offline import VoskASR
from safora.asr.streaming import ASRConfig,StreamingASR


def receipt(path):
    file=Path(path)/'.safora-model.json'
    return json.loads(file.read_text())['file_sha256'] if file.exists() else None


class RealVoskTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('SAFORA_VOSK_EN_MODEL'),'Set SAFORA_VOSK_EN_MODEL for native English tests')
    def test_real_english_speech_endpoint_reset_and_close(self):
        import vosk,requests
        directory=os.environ['SAFORA_VOSK_EN_MODEL']
        fixture=Path(__file__).resolve().parents[1]/'android/app/src/androidTest/assets/synthetic-refusal.pcm16'
        audio=np.frombuffer(fixture.read_bytes(),dtype='<i2').astype(np.float32)/32768
        with patch.object(requests.sessions.Session,'request',side_effect=AssertionError('Network forbidden during inference')):
            with VoskASR(directory,'en',file_checksums=receipt(directory)) as model:
                model.accept(audio[:1600]);model.reset()
                result=model.transcribe(audio)
                self.assertTrue(result.text.strip());self.assertTrue(result.is_final);self.assertEqual(result.language,'en')
                self.assertTrue(model.transcribe(np.zeros(1600)).uncertain)
        self.assertTrue(model.closed)
    @unittest.skipUnless(os.environ.get('SAFORA_VOSK_HI_MODEL'),'Set SAFORA_VOSK_HI_MODEL for native Hindi tests')
    def test_real_hindi_model_decoding_and_streaming(self):
        directory=os.environ['SAFORA_VOSK_HI_MODEL']
        audio=np.zeros(1600,np.float32)
        fixture=os.environ.get('SAFORA_VOSK_HI_FIXTURE')
        if fixture:audio=np.frombuffer(Path(fixture).read_bytes(),dtype='<i2').astype(np.float32)/32768
        results=[]
        def factory(language):return VoskASR(directory,language,file_checksums=receipt(directory))
        controller=StreamingASR({},language='hi',config=ASRConfig(max_chunk_seconds=8),on_result=results.append,factory=factory).start()
        try:
            self.assertTrue(controller.submit(audio));self.assertTrue(controller.wait_idle(30))
            self.assertEqual(controller.snapshot()['errors'],0);self.assertEqual(results[-1].language,'hi')
            self.assertTrue(results[-1].transcript.is_final)
            if fixture:self.assertTrue(results[-1].transcript.text.strip())
            controller.set_language(None);self.assertTrue(controller.wait_idle(10));self.assertFalse(controller.snapshot()['model_loaded'])
        finally:self.assertTrue(controller.stop())
