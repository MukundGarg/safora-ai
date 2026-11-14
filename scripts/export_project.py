"""Export reviewable source and the package-verified debug APKs to a writable directory."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
apk=root/'android/app/build/outputs/apk/debug/app-debug.apk'
verified=json.loads((root/'docs/verification/apk-verification.json').read_text())
if hashlib.sha256(apk.read_bytes()).hexdigest()!=verified['sha256']:raise ValueError('APK differs from package-verification report; verify the current APK before export')
shutil.copyfile(apk,out/'safora-ai-debug.apk')
shutil.copyfile(root/'android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk',out/'safora-ai-device-tests.apk')
archive=out/'safora-ai-source.zip'
excluded={'.git','.gradle','build','work','output','__pycache__','.venv'}
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(root)
        if any(part in excluded for part in rel.parts):continue
        if rel.parts[:5]==('android','app','src','main','assets'):continue
        if path.suffix in {'.pyc','.tflite','.onnx','.bin','.pt','.safetensors','.zip','.apk','.keystore','.jks','.wav','.mp3'}:continue
        z.write(path,Path('safora-ai')/rel)
shutil.copyfile(root/'docs/test-report.md',out/'test-report.md')
shutil.copyfile(root/'docs/integration.md',out/'integration-guide.md')
shutil.copyfile(root/'docs/verification/apk-verification.json',out/'apk-verification.json')
sums=[]
for path in sorted(out.iterdir()):
    if path.is_file() and path.name!='SHA256SUMS':sums.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.name)
(out/'SHA256SUMS').write_text('\n'.join(sums)+'\n')
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None:raise ValueError('Export archive failed integrity check')
    if 'safora-ai/android/gradlew' not in z.namelist():raise ValueError('Missing Gradle wrapper')
print('Exported to',out)
for path in sorted(out.iterdir()):print(path.name,path.stat().st_size)
