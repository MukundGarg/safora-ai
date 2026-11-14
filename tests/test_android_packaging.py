import hashlib,json,os,unittest,zipfile,struct
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

    def test_extracted_file_metadata_matches_pinned_archives(self):
        for spec in self.expected['vosk'].values():
            with self.apk.open('assets/'+spec['asset']) as handle,zipfile.ZipFile(handle) as archive:
                self.assertEqual(sum(v['bytes'] for v in spec['files'].values()),spec['uncompressed_bytes'])
                for name,expected in spec['files'].items():
                    with archive.open(spec['root']+'/'+name) as stream:
                        digest=hashlib.sha256();count=0
                        while block:=stream.read(65536):digest.update(block);count+=len(block)
                    self.assertEqual(digest.hexdigest(),expected['sha256']);self.assertEqual(count,expected['bytes'])
        self.assertIn('assets/RUNTIME-NOTICES.txt',self.apk.namelist())

    def test_64bit_native_elf_load_segments_are_16kb_aligned(self):
        for name in self.apk.namelist():
            if name.startswith(('lib/arm64-v8a/','lib/x86_64/')) and name.endswith('.so'):
                data=self.apk.read(name);self.assertEqual(data[:6],b'\x7fELF\x02\x01')
                offset=struct.unpack_from('<Q',data,32)[0];width,count=struct.unpack_from('<HH',data,54)
                for i in range(count):
                    start=offset+i*width
                    if struct.unpack_from('<I',data,start)[0]==1:
                        self.assertGreaterEqual(struct.unpack_from('<Q',data,start+48)[0],16384,name)
                        file_offset,address=struct.unpack_from('<QQ',data,start+8)
                        self.assertEqual(file_offset%16384,address%16384,name)
