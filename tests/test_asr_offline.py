"""Deterministic native-response fixtures; not real recognition accuracy data."""
import hashlib,json,tempfile,unittest,zipfile,stat
from pathlib import Path
import numpy as np
from safora.asr.offline import Transcript,VoskASR,WhisperASR,pcm16le
from safora.asr.assets import REQUIRED_FILES,prepare_model,sha256,validate_archive,validate_model_dir


def fake_directory(root):
    for name in REQUIRED_FILES:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'unit-test-placeholder-not-model-weights')
    return root


def response(text,confidence=.9):
    return json.dumps({'text':text,'result':[{'word':'fixture','conf':confidence}] if text else []})


class Native:
    def __init__(self,model,rate):
        self.model=model;self.rate=rate;self.accepted=[];self.closed=False
        self.endpoint=False;self.result=response('endpoint');self.partial='{"partial":"partial"}';self.final=response('tail')
    def SetWords(self,value):self.words=value
    def AcceptWaveform(self,pcm):self.accepted.append(pcm);return self.endpoint
    def Result(self):return self.result
    def PartialResult(self):return self.partial
    def FinalResult(self):return self.final
    def close(self):self.closed=True


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=fake_directory(Path(self.temp.name)/'model');self.created=[]
        class Model:
            closed=False
            def close(inner):inner.closed=True
        self.model=Model()
    def tearDown(self):self.temp.cleanup()
    def adapter(self,language='en',**kwargs):
        def factory(model,rate):
            native=Native(model,rate);self.created.append(native);return native
        return VoskASR(self.root,language,model_factory=lambda path:self.model,recognizer_factory=factory,**kwargs)
    def test_pcm_endpoints_little_endian_and_no_silent_clipping(self):
        pcm=pcm16le(np.array([-1.,-.5,0.,.5,1.]))
        np.testing.assert_array_equal(pcm,[-32768,-16384,0,16384,32767])
        self.assertEqual(pcm16le(np.array([-1.,0.,1.])).tobytes(),b'\x00\x80\x00\x00\xff\x7f')
        for x in [np.array([1.1]),np.array([np.nan]),np.array([np.inf]),np.zeros((2,1)),np.array([1],np.int16),np.zeros(192001)]:
            with self.assertRaises(ValueError):pcm16le(x)
    def test_partial_is_provisional_and_not_appended_as_another_utterance(self):
        with self.adapter() as adapter:
            first=adapter.accept(np.zeros(16));self.assertFalse(first.is_final);self.assertTrue(first.uncertain)
            self.created[-1].partial='{"partial":"new partial"}'
            adapter.accept(np.zeros(16));out=adapter.finish()
            self.assertEqual(out.text,'tail');self.assertFalse(out.uncertain);self.assertEqual(out.language,'en')
    def test_endpoint_result_and_final_tail_are_not_lost(self):
        with self.adapter() as adapter:
            adapter.begin();self.created[-1].endpoint=True
            part=adapter.accept(np.zeros(16));self.assertTrue(part.is_final)
            final=adapter.finish();self.assertEqual(final.text,'endpoint tail')
            self.assertEqual(adapter.finish().text,'')
    def test_batch_uses_bounded_accept_calls(self):
        with self.adapter() as adapter:
            adapter.transcribe(np.zeros(3201))
            self.assertEqual([len(x) for x in self.created[-1].accepted],[3200,3200,2])
            self.assertTrue(self.created[-1].closed)
    def test_empty_input_has_no_recognizer_and_is_uncertain(self):
        with self.adapter('hi') as adapter:
            out=adapter.transcribe(np.zeros(0));self.assertEqual(out.text,'');self.assertTrue(out.uncertain)
            self.assertEqual(out.language,'hi');self.assertEqual(self.created,[])
    def test_silence_final_has_no_invented_text(self):
        with self.adapter() as adapter:
            adapter.begin();self.created[-1].final=response('')
            self.assertEqual(adapter.finish().reason,'empty_or_silence')
    def test_unknown_language_is_not_inferred_from_directory(self):
        with self.adapter(None) as adapter:
            out=adapter.transcribe(np.zeros(16));self.assertIsNone(out.language)
            self.assertTrue(out.uncertain);self.assertEqual(out.reason,'unknown_language')
        with self.assertRaises(ValueError):self.adapter('auto')
    def test_malformed_native_responses_return_uncertain_empty_results(self):
        for raw in ['invalid','[]','{}','{"text":1}',response('secret',float('nan')),
                    '{"text":"secret","result":"wrong"}', '{"text":"'+('x'*4097)+'"}']:
            with self.adapter() as adapter:
                adapter.begin();self.created[-1].final=raw
                out=adapter.finish();self.assertEqual(out.text,'');self.assertTrue(out.uncertain)
                self.assertEqual(out.reason,'malformed_native_result')
    def test_missing_word_confidence_does_not_become_a_fake_high_score(self):
        with self.adapter() as adapter:
            adapter.begin();self.created[-1].final='{"text":"fixture"}'
            out=adapter.finish();self.assertTrue(out.uncertain);self.assertEqual(out.reason,'confidence_unavailable')
    def test_segment_limit_reset_and_close(self):
        adapter=self.adapter(max_seconds=.01)
        adapter.accept(np.zeros(160))
        with self.assertRaises(ValueError):adapter.accept(np.zeros(1))
        previous=self.created[-1];adapter.reset();self.assertTrue(previous.closed)
        self.assertEqual(adapter.finish().text,'')
        adapter.close();adapter.close();self.assertTrue(self.model.closed)
        for action in [lambda:adapter.accept(np.zeros(1)),adapter.reset,adapter.finish]:
            with self.assertRaises(RuntimeError):action()
    def test_transcript_repr_never_contains_text(self):
        self.assertNotIn('private words',repr(Transcript('private words',True,'en',-1)))
    def test_optional_whisper_rejects_missing_tokenizer_before_native_import(self):
        for name in ('model.bin','config.json','vocabulary.txt'):(self.root/name).write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError,'refusing tokenizer download'):WhisperASR(self.root)
    def test_optional_checksums_detect_directory_corruption(self):
        name=REQUIRED_FILES[0];expected=sha256(self.root/name)
        validate_model_dir(self.root,{name:expected});(self.root/name).write_bytes(b'changed')
        with self.assertRaises(ValueError):validate_model_dir(self.root,{name:expected})


class AssetTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def archive(self,extra=None):
        path=self.root/'fixture.zip'
        with zipfile.ZipFile(path,'w') as z:
            for name in REQUIRED_FILES:z.writestr('fixture/'+name,b'test-only')
            if extra:extra(z)
        return path,dict(root='fixture',sha256=sha256(path),bytes=path.stat().st_size,language='fixture')
    def test_verified_extraction_and_receipt(self):
        archive,spec=self.archive();dest=prepare_model(archive,self.root/'out',spec)
        receipt=json.loads((dest/'.safora-model.json').read_text());validate_model_dir(dest,receipt['file_sha256'])
        self.assertEqual(receipt['archive_sha256'],spec['sha256'])
        with self.assertRaises(FileExistsError):prepare_model(archive,dest,spec)
    def test_incomplete_directory_and_wrong_archive_checksum(self):
        with self.assertRaises(ValueError):validate_model_dir(self.root)
        archive,spec=self.archive();spec['sha256']='0'*64
        with self.assertRaises(ValueError):validate_archive(archive,spec)
    def test_zip_traversal_and_symlink_are_rejected_without_writing(self):
        for extra in [lambda z:z.writestr('fixture/../escape',b'bad'),lambda z:z.writestr('/escape',b'bad')]:
            archive,spec=self.archive(extra)
            with self.assertRaises(ValueError):prepare_model(archive,self.root/'out',spec)
            self.assertFalse((self.root/'out').exists())
        def link(z):
            info=zipfile.ZipInfo('fixture/link');info.create_system=3;info.external_attr=(stat.S_IFLNK|0o777)<<16
            z.writestr(info,'/elsewhere')
        archive,spec=self.archive(link)
        with self.assertRaises(ValueError):validate_archive(archive,spec)
