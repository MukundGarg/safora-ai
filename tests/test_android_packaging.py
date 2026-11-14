import hashlib,json,os,unittest,zipfile
from pathlib import Path

@unittest.skipUnless(os.environ.get('SAFORA_APK'),'Set SAFORA_APK for real APK packaging checks')
class AndroidPackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.apk=zipfile.ZipFile(os.environ['SAFORA_APK'])
        cls.metadata=json.loads(cls.apk.read('assets/model-assets.json'))
        cls.expected=json.loads((Path(__file__).resolve().parents[1]/'models/android-assets.json').read_text())
    @classmethod
    def tearDownClass(cls):cls.apk.close()
    def test_all_pinned_models_are_bundled(self):
        self.assertEqual(self.metadata,self.expected)
        for spec in [self.expected['yamnet'],*self.expected['vosk'].values()]:
            digest=hashlib.sha256()
            with self.apk.open('assets/'+spec['asset']) as f:
                while block:=f.read(1024*1024):digest.update(block)
            self.assertEqual(digest.hexdigest(),spec['sha256'])
    def test_risk_classifier_is_explicitly_untrained(self):
        self.assertIs(self.metadata['risk_classifier']['trained_weights'],False)
        self.assertEqual(self.metadata['risk_classifier']['type'],'UNVALIDATED_HEURISTIC')
    def test_no_recordings_or_synthetic_fixture_in_main_assets(self):
        assets=[n for n in self.apk.namelist() if n.startswith('assets/')]
        self.assertFalse(any(n.endswith(('.wav','.pcm16','.mp3')) for n in assets))
        self.assertIn('assets/MODEL-LICENSE-APACHE-2.0.txt',assets)
        self.assertIn('assets/MODEL-NOTICES.txt',assets)
    def test_native_runtime_abi_consistency(self):
        names=set(self.apk.namelist())
        abis={n.split('/')[1] for n in names if n.startswith('lib/') and n.endswith('.so')}
        self.assertIn('arm64-v8a',abis);self.assertIn('x86_64',abis)
        for abi in abis:
            for library in ['libtensorflowlite_jni.so','libvosk.so','libjnidispatch.so']:
                self.assertIn(f'lib/{abi}/{library}',names)
