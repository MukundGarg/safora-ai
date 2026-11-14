"""Compile Android Kotlin source and run pure JVM tests without an Android SDK.
This is NOT an APK build, lint check, emulator test or on-device verification.
Downloads pinned compiler/framework artifacts into work/kotlin-check/.
"""
from pathlib import Path
import argparse,subprocess,urllib.request,zipfile
p=argparse.ArgumentParser();p.add_argument('--cache',default='work/kotlin-check');a=p.parse_args()
root=Path(__file__).resolve().parents[1];cache=Path(a.cache).resolve();cache.mkdir(parents=True,exist_ok=True)
artifacts={
'compiler.jar':'org/jetbrains/kotlin/kotlin-compiler-embeddable/2.0.21/kotlin-compiler-embeddable-2.0.21.jar',
'stdlib.jar':'org/jetbrains/kotlin/kotlin-stdlib/2.0.21/kotlin-stdlib-2.0.21.jar',
'script.jar':'org/jetbrains/kotlin/kotlin-script-runtime/2.0.21/kotlin-script-runtime-2.0.21.jar',
'reflect.jar':'org/jetbrains/kotlin/kotlin-reflect/1.6.10/kotlin-reflect-1.6.10.jar',
'annotations.jar':'org/jetbrains/annotations/13.0/annotations-13.0.jar',
'trove.jar':'org/jetbrains/intellij/deps/trove4j/1.0.20200330/trove4j-1.0.20200330.jar',
'coroutines.jar':'org/jetbrains/kotlinx/kotlinx-coroutines-core-jvm/1.9.0/kotlinx-coroutines-core-jvm-1.9.0.jar',
'android.jar':'org/robolectric/android-all/15-robolectric-12650502/android-all-15-robolectric-12650502.jar',
'vosk.aar':'com/alphacephei/vosk-android/0.3.47/vosk-android-0.3.47.aar',
'jna.aar':'net/java/dev/jna/jna/5.13.0/jna-5.13.0.aar',
'lite.aar':'org/tensorflow/tensorflow-lite/2.16.1/tensorflow-lite-2.16.1.aar',
'lite-api.aar':'org/tensorflow/tensorflow-lite-api/2.16.1/tensorflow-lite-api-2.16.1.aar',
'junit.jar':'junit/junit/4.13.2/junit-4.13.2.jar',
'hamcrest.jar':'org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar'}
for name,uri in artifacts.items():
    target=cache/name
    if not target.exists():urllib.request.urlretrieve('https://repo.maven.apache.org/maven2/'+uri,target)
    if target.suffix=='.aar':
        with zipfile.ZipFile(target) as z:target.with_suffix('.jar').write_bytes(z.read('classes.jar'))
import os
output=cache/'tests.jar'
classpath=os.pathsep.join(str(f) for f in cache.glob('*.jar') if f!=output)
subprocess.run(['java','-cp',classpath,'org.jetbrains.kotlin.cli.jvm.K2JVMCompiler','-no-stdlib','-no-reflect','-jvm-target','17','-classpath',classpath,'-d',str(output),*[str(p) for p in (root/'android').rglob('*.kt')]],check=True)
subprocess.run(['java','-cp',str(output)+os.pathsep+classpath,'org.junit.runner.JUnitCore','org.safora.ai.EngineTest','org.safora.ai.AudioRingTest','org.safora.ai.PipelineTest'],check=True)
