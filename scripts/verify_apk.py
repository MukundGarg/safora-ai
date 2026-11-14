"""Check signatures, ZIP alignment, metadata, and pinned assets. Not a device test."""
import argparse,hashlib,json,os,re,subprocess,unittest,zipfile,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('apk',type=Path);p.add_argument('--build-tools',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];apk=a.apk.resolve()
os.environ['SAFORA_APK']=str(apk)
suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_android_packaging.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():raise SystemExit(1)
def run(*cmd):return subprocess.check_output([str(c) for c in cmd],stderr=subprocess.STDOUT,text=True)
signing=run(a.build_tools/'apksigner','verify','--verbose','--print-certs',apk)
alignment=run(a.build_tools/'zipalign','-c','-P','16','4',apk)
badging=run(a.build_tools/'aapt2','dump','badging',apk)
elf={}
with zipfile.ZipFile(apk) as z:
    for name in z.namelist():
        if not name.startswith('lib/') or not name.endswith('.so'):continue
        data=z.read(name)
        if data[:4]!=b'\x7fELF' or data[5]!=1:raise ValueError('Unexpected ELF format: '+name)
        is64=data[4]==2
        offset=struct.unpack_from('<Q' if is64 else '<I',data,32 if is64 else 28)[0]
        width,count=struct.unpack_from('<HH',data,54 if is64 else 42)
        alignments=[]
        for i in range(count):
            start=offset+i*width
            if struct.unpack_from('<I',data,start)[0]==1:
                alignments.append(struct.unpack_from('<Q' if is64 else '<I',data,start+(48 if is64 else 28))[0])
        elf[name]=min(alignments) if alignments else 0
native16=all(value>=16384 for name,value in elf.items() if name.split('/')[1] in ['arm64-v8a','x86_64'])
if "uses-permission: name='android.permission.INTERNET'" in badging:raise ValueError('Unexpected INTERNET permission')
report={'apk':apk.name,'bytes':apk.stat().st_size,'sha256':hashlib.sha256(apk.read_bytes()).hexdigest(),
        'packaging_tests_passed':result.testsRun,'apksigner':signing,'zipalign_passed':True,
        'badging':badging,'android_runtime_tested_by_this_check':False,'elf_min_load_alignment_bytes':elf,'native_16kb_static_alignment_passed':native16,
        'notes':'Signature/asset/ZIP checks do not verify installation, ELF 16KB compatibility, microphone, ASR accuracy, or device behavior.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
print('APK package verified; device/emulator runtime status must be recorded separately.')
