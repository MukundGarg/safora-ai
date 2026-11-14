# SAFORA Member 2 — final integration test report

Date:14 November2026. Branch `feat/task4-ml-integration`, base
`feat/task3-conversation-risk` at `d53d3697d0f134284dee6c182a1b6411148fb2dc`.
Reports describe actual checks; historical Task 1–3 evidence is not reassigned to
the changed Task 4 APK. No representative safety/ASR/risk accuracy is measured.

## Final results

| Check | Actual result | Scope |
|---|---|---|
| Full Python unittest suite | **147 passed,0 failed,0 skipped** | Includes all prior138 tests,9 coordinator/shared-rule tests, real YAMNet/Whisper/Vosk EN/HI host regressions and packaging checks |
| Kotlin JVM suite | **22 passed,0 failed,0 skipped** | All prior14 plus8 core concurrency/lifecycle/monitoring/rule-contract tests; real core compilation |
| Debug app APK | **Built successfully** | Actual SDK35/Gradle8.9/AGP8.7.3/Kotlin2.0.21/JDK21, JVM17 |
| Debug instrumentation APK | **Compiled successfully** | Existing DeviceSmokeTest plus new2-case MlCoreSmokeTest, not executed in Task 4 |
| lintDebug | **0 errors,12 warnings** | Existing static warnings; not device behavior |
| APK signature/ZIP alignment | **Passed** | apksigner verification,16KB ZIP-entry alignment,4 asset/ABI/fixture packaging tests |
| Bundled asset checks/notices | **Passed** | Pinned YAMNet/en/hi SHA256s, no trained-risk claim, no main-APK test recordings, notices included |
|64-bit native ELF16KB alignment | **Failed static readiness** | Existing4096-byte PT_LOAD alignment in native runtimes; distribution blocker |
| Native host coordinator | **Passed execution/cleanup** | Real YAMNet + real EN/HI Vosk, authored synthetic speech, zero errors/drops; no selected positive heuristic labels |
| Emulator startup | **Blocked** | Insufficient disk space; no Task 4 install/run/permission/microphone/background test executed |
| Physical phone / live host microphone | **Not executed** | No physical-device/real-microphone/resource/battery evidence |
| Genuine trained risk model | **Absent** | No licensed human-labeled risk dataset/base/dependencies; no training/export/learned accuracy |

Full logs: [Python](verification/task4/python-tests.txt),
[Android final build](verification/task4/android-final-build.txt),
[APK verification](verification/task4/apk-verification.json),
[native coordinator](verification/task4/native-coordinator.json),
[emulator attempt](verification/task4/emulator-attempt.txt).

Final APK `android/app/build/outputs/apk/debug/app-debug.apk`:
**143,165,938bytes**, SHA256
`42d8c476dbe7366f9834a04b2523b31d83595f351e6c5feb25caf1f4bfa6b736`.
This signed debug artifact was **not installed/run in Task 4**. Main package remains
`org.safora.demo`, API26min/API35target, no INTERNET permission. APK distribution
notices are11358-byte Apache license and1295-byte model notices. Source Git excludes
model binaries, APKs and signing keys. Rebuild/provision commands are in the handoff.

## Reproducible regression/build commands

From repository root, actual Python3.12.14 run used existing local optional runtime
directories (Vosk0.3.45, ai-edge-litert2.3.0, faster-whisper1.2.1/CTranslate2 4.9.0):

```bash
SAFORA_APK=/workspace/safora-ai/android/app/build/outputs/apk/debug/app-debug.apk \
SAFORA_YAMNET_MODEL=/workspace/work/model/1.tflite \
SAFORA_WHISPER_MODEL=/workspace/work/whisper-tiny \
SAFORA_VOSK_EN_MODEL=/workspace/work/asr-task2/en \
SAFORA_VOSK_HI_MODEL=/workspace/work/asr-task2/hi \
SAFORA_VOSK_HI_FIXTURE=/workspace/work/asr-task2/hindi-synthetic.pcm16 \
HF_HUB_OFFLINE=1 PYTHONPATH=python:/workspace/work/runtime:/workspace/work/asr-runtime:/workspace/work/vosk-runtime \
python -m unittest discover -s tests -v

ANDROID_USER_HOME=/workspace/work/android-user ANDROID_HOME=/workspace/work/android-sdk \
GRADLE_USER_HOME=/workspace/work/gradle-home /workspace/work/gradle-8.9/bin/gradle \
-p android :safora-ai-core:testDebugUnitTest :app:assembleDebug \
:app:assembleDebugAndroidTest :app:lintDebug --console=plain

PYTHONPATH=python python scripts/verify_apk.py \
android/app/build/outputs/apk/debug/app-debug.apk \
--build-tools /workspace/work/android-sdk/build-tools/35.0.0 \
--output docs/verification/task4/apk-verification.json
```

The final Android run completed116tasks (29executed/87up-to-date) in1m42s. Changed
core/test/app artifacts compiled; JVM suite reran. Kotlin XML totals1AudioRing+
3PCM+4Engine+6legacyPipeline+8MlController=22, zero failures/skips. Initial daemon
startup warnings occurred in earlier builds and recovered; final build passed.
SDK analytics initialization also reports unwritable default settings. Lint's
12warnings remain recorded in the generated report; no new claim of release/Play
readiness follows from debug build success. Reported Python tool CLI status output
at the bottom of the suite is from deterministic evaluation fixtures, not learned
classifier metrics.

## Native host orchestration and resource measurements

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=python:/workspace/work/runtime:/workspace/work/vosk-runtime \
python scripts/ml_coordinator_smoke.py --yamnet /workspace/work/model/1.tflite \
--english-model /workspace/work/asr-task2/en --hindi-model /workspace/work/asr-task2/hi \
--english-pcm16 android/app/src/androidTest/assets/synthetic-refusal.pcm16 \
--hindi-pcm16 /workspace/work/asr-task2/hindi-synthetic.pcm16 \
--output docs/verification/task4/native-coordinator.json
```

Fixtures are authored synthetic TTS from Tasks 2/earlier demo, not human-labeled
risk/accuracy data. English contains128000samples/8s; Hindi32637samples/~2.04s,
padded to8s for the fixed coordinator segment. Hindi text/generation provenance
is recorded in `asr-task2-report.md`. Source audio is not saved by the coordinator.

Actual single host run:16s total padded input,160submitted/processed100ms blocks,
31YAMNet windows,2ASR segments, both nonempty transcripts,0native errors,0queue
losses, speech models switched en→hi with context cleared. Heuristic returned no
positive categories for either recognized result, so **0ML check-in cues**. The
manual-SOS hook was invoked after stop and emitted a separate user-action cue.
No result was altered or threshold tuned to force a demonstration alert.
Deterministic mocked tests verify positive risk/acoustic cues reach only uncertain,
user-confirmed check-ins; this does not establish recognition/safety accuracy.

| Measurement | Single-run host observation |
|---|---:|
| Total wall / process CPU |8.291s /6.477s|
| Peak process RSS (Linux ru_maxrss) |328152KiB (~320.5MiB)|
| Max YAMNet inference wall |3.718ms|
| Max ASR native processing wall |3091.506ms|
| Max risk-rule wall |1.422ms|
| Extracted English / Hindi directory bytes, including receipt |70,900,373 /82,085,649|
| YAMNet file bytes |4,126,810|

Blocks were **paced by worker-idle barriers** for reproducibility, not delivered by
live microphone timing. These are whole-process host observations including native
runtime allocations, not incremental Android PSS or low-RAM/battery benchmarks.
ASR/max metrics exclude some queue/load overhead. The Task 1 synthetic capture-
to-decision field is not an actual microphone latency measurement in this file
smoke. No real Android CPU/PSS/drop measurements exist for Task 4; core metrics
instrumentation is ready for the device owner to collect them.

## Failures found, fixed and remaining

1. Python Task 2 worker could consume/discard the first new-language chunk when a
   control generation changed while its blocking queue read waited. Coordinator
   language-switch test initially failed (`en` remained loaded). It now adopts
   current control state before consuming the new chunk; prior tests preserved.
2. Initial Kotlin pressure test failed because speech FIFO overflow advanced a
   shared continuity generation and repeatedly reset the acoustic ring. Separate
   capture and conversation generations now preserve YAMNet windows independently.
3. Immediate-switch Kotlin regression reproduced the same poll/control race as
   Python. It now applies fresh state after poll; the regression passes. The
   [earlier failed build log](verification/task4/android-regression-first-failure.txt)
   retains the21-case/1-failure run.
4. The shared TSV test parser initially stripped literal quote characters before
   Python classification, turning a quoted-threat fixture into a direct statement.
   Parsing now preserves literal quotes (`QUOTE_NONE`), and both language ports
   pass the12shared contract fixtures. The [failed Python log](verification/task4/python-first-failure.txt)
   retains the147-case/1-failure run. These fixtures are not a trained corpus.
5. Emulator37.3.9 software CPU startup on API26 AVD failed its disk-space check
   (workspace ~1.2GiB free). Smaller requested partition/config was also rejected;
   AVD config was restored and no user data deleted. No KVM device exists. The
   stock adb daemon also encountered a read-only default `.android` location.
   These environment constraints were not bypassed or described as runtime success.
6. Native ELF16KB alignment still fails static readiness. Real phone lifecycle,
   microphone/permission/background, actual notification integration, battery,
   low-RAM PSS/latency and quiet/noisy speech quality remain unverified.

Actual attempted emulator command (full environment/paths in log/handoff):

```bash
ANDROID_AVD_HOME=/workspace/work/android-user/avd \
ANDROID_USER_HOME=/workspace/work/android-user \
ANDROID_EMULATOR_HOME=/workspace/work/android-user \
ANDROID_HOME=/workspace/work/android-sdk \
/workspace/work/android-sdk/emulator/emulator -avd safora26 \
-no-window -no-audio -no-boot-anim -gpu swiftshader -accel off -no-snapshot \
-no-metrics -partition-size 768 \
-data /workspace/work/android-user/avd/safora26.avd/userdata-2gb.img -port 5554
```

On an authorized available device/emulator, install both generated APKs and run
`./gradlew :app:connectedDebugAndroidTest`, or
`adb shell am instrument -w org.safora.demo.test/androidx.test.runner.AndroidJUnitRunner`.
New MlCoreSmokeTest executes actual native models through the new independent core;
its permission/opt-in policy case uses injected denied flags, not OS permission
revocation. Existing DeviceSmokeTest covers native batch decoding/local notification,
manual SOS and service virtual microphone startup/shutdown. Neither suite was run
in Task 4. Add real OS denial/revoke, background interruptions, queue contention and
physical mic/device tests before claiming those behaviors verified.

No labeled independent scream/speech/risk benchmark was supplied, and the30clip
external scream set is reused. No learned risk weights/accuracy, reliable Hindi/
Hinglish coverage, female/gender identification, automatic LID or production safety
performance is claimed. See [handoff](member2-final-handoff.md) for exact ownership,
asset hashes and prioritized blockers rather than treating unit counts as safety
validation.
