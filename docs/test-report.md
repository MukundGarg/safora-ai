# SAFORA AI test report — 14 November 2026

## Result

A signed debug APK was built, **installed and run on an Android 8.0.0
API26 x86_64 emulator**, with four passing instrumented tests. This is
Android runtime evidence, not just compilation. **No physical Android phone was
tested.** The emulator used software CPU emulation, with host audio disabled.
Microphone verification therefore means virtual AudioRecord capture, not a real
person speaking into phone hardware.

APK: `org.safora.demo`, version0.2/code2, minSDK26, target/compileSDK35.
Size: **143,109,312 bytes**. SHA256:
`72fcf6417118c52b6fdb131580ee781ca643ebe90e26975cd718eeb1506a0e28`.
The installed artifact hash matches the package-verification report and exported
APK. Its debug signature verifies under APK Signature Scheme v2.

## Actual executed checks

| Check | Result | What it verifies |
|---|---|---|
| Python unittest suite | **28 passed** | Includes real YAMNet/Whisper host smoke tests and four real APK packaging tests |
| Kotlin Gradle JVM suite | **11 passed** | Ring continuity/gaps, temporal confirmation/cooldown, manual priority, uncertainty, contextual baseline, resource downgrade and metric aggregation |
| Actual SDK Gradle build | **Passed** | Debug app APK and instrumented-test APK generated |
| Android lintDebug | **0 errors, 12 warnings** | Static checks; remaining warnings are recorded in the lint text report |
| APK packaging checks | **4 passed** | Pinned model bytes, untrained-classifier metadata, no raw/test audio in main assets, complete runtime set for each ABI |
| APK signature / ZIP alignment | **Passed** | apksigner v2 verification and zipalign 16KB ZIP-entry alignment |
| Gradle wrapper | **Passed** | Gradle8.9 download with pinned distribution SHA256; wrapper --version executed |
| Android instrumented suite | **4 passed** | Real native models, synthetic speech risk path, local notifications, manual SOS, virtual microphone service start/inference/stop |
| Standalone Activity launch | **Passed** | am start reported Status:ok; real UI screenshot/hierarchy captured |
| Physical-phone validation | **Not executed** | No physical device connected |
| Trained conversation classifier / labeled accuracy | **Unavailable / not measured** | No licensed labeled dataset or SAFORA fine-tuned weights |
| Native ELF 16KB readiness | **Fails static alignment check** | TFLite/Vosk 64-bit PT_LOAD segments align to4096 bytes; see limitations |

Build tools: Gradle8.9, AGP8.7.3, Kotlin2.0.21, OpenJDK21.0.13 (JVM target17),
Android SDK35/build-tools35.0.0. The earlier source-only surrogate compilation
has been superseded by the real SDK build. Full build logs, JUnit summaries,
package report, device metadata and instrumented output are in `verification/`.
The raw Python log includes INFO output from the real LiteRT CPU interpreter.

The instrumented suite took **576.508 seconds** in the software CPU emulator;
installation plus execution took 681.0 host seconds.
These are emulator test durations, not phone inference benchmarks.

### Exactly what the four Android tests exercised

1. Bundled YAMNet checksum and English/Hindi Vosk archive extraction, native model
   creation and silence transcription. Hindi speech accuracy was not evaluated.
2. Real YAMNet inference and real English Vosk transcription of an authored
   synthetic refusal clip, followed by the development classifier, decision
   engine and an actual local Android notification. This fixture enters through
   a test audio buffer, not the microphone. It tests integration, not accuracy.
3. The actual Activity's manual SOS control triggers a notification without
   microphone monitoring; incident resolution clears the local state.
4. The actual opted-in Activity starts the microphone foreground service in Lite,
   virtual AudioRecord capture produces a completed YAMNet run and performance
   counters, then stopService completes cleanup and reports Monitoring stopped.

Real spoken microphone→ASR→risk alerts on a physical phone have **not** been tested.
The microphone and synthetic-ASR paths were verified separately. Android13+
notification permission behavior and Android14+/15 microphone-service restrictions
were compiled/linted but have not been runtime-tested on those OS versions.

## Implemented, simulated, incomplete and unverified inventory

| Component | Current implementation | Evidence / limit |
|---|---|---|
| YAMNet | Real pretrained TFLite model, streaming ring/confirmation | Host silence and actual Android native inference passed; scream accuracy unmeasured |
| Offline ASR | Python Whisper Tiny CPU int8 and Vosk adapters; Android Vosk | Real host smoke tests and Android native English fixture/Hindi silence passed; no representative WER/CER |
| Conversation risk | Heuristic baseline plus encoder training/loading/export infrastructure | **No trained SAFORA encoder weights**; baseline always uncertain, text alerts limited to Level1 |
| VAD | Energy threshold in the working pipelines; optional Python Silero adapter | Energy logic exercised; quiet-speech coverage and Silero weights unverified; Android Silero absent |
| Decision engine | Transparent levels, cooldowns, active incident, manual bypass | JVM and Android callback/notification paths passed; no external dispatch |
| Resource controller | Available memory, thermal, battery, latency/backlog; optional ASR release | Unit policy checks passed; long-running physical-device adaptation not validated |
| Android lifecycle/UI | Opt-in, permission request, foreground notification, start/stop, language/mode controls | API26 virtual microphone/start-stop and Activity tested; other OEM/API/call/revocation cases unverified |
| Metrics | Wall/worker-thread CPU latency, PSS/RSS, app drops, skipped-window estimate and JSON export UI | Aggregation unit tests, live emulator counters and UI checks; phone targets unmeasured |
| Main-app integration | IncidentSink, session-scoped IDs, decision timestamps, guide | Main incident manager/ESP32 transport/trusted contacts are external integration work |
| Evidence/privacy | Transient buffers/context, no transcript retention/network, disabled backup/transfer | No incident evidence store is implemented or claimed; host must add encrypted opt-in evidence separately |

Mocked interpreter outputs and authored text/score sequences occur in isolated unit
tests only. They are not presented as model accuracy results. Instrumented model
checks use actual bundled native runtimes/weights, not mocked ASR or YAMNet.
Synthetic fixtures are eSpeak NG authored audio, not a labeled safety dataset,
representative women's voices, or training evidence. No gender classifier exists.

## Models and classifier inspection

`models/android-assets.json` and the APK manifest pin real YAMNet and English/Hindi
Vosk archives. Asset bytes are verified before build, in APK inspection, and during
Android loading/extraction. The APK is self-contained and has no INTERNET permission.
Archives expand to approximately68MiB English and78MiB Hindi in private storage;
only one native ASR model is resident at a time. Source exports omit generated
weight files and include explicit hash-checked provisioning scripts/licenses.

`python/safora/classifier/risk.py` implements ContextBaseline and a loader for
locally fine-tuned encoder weights. `python/safora/training/train.py` implements
training/evaluation with source/session split checks; `scripts/export_encoder.py`
implements ONNX export/quantization. These steps were **not executed** without
licensed labeled data/local encoder. Android uses `DevelopmentClassifier`, not an
encoder. No ONNX/Safetensors/trained safety head is packaged. The APK explicitly
states `UNVALIDATED_HEURISTIC` / `trained_weights:false`. Model confidence scores
are not probabilities of danger; no text detection accuracy is claimed.

## Performance evidence and remaining constraints

The app records YAMNet/ASR wall mean/max and worker-thread CPU mean/run count,
current/peak sampled PSS, kernel peak RSS, dropped sample/block counts, skipped
analysis-window estimates and queue high-water. It records no audio/transcript,
risk labels, identity or location in metrics. Memory is sampled every five seconds.
Worker CPU excludes native worker threads; PSS sampling can miss brief peaks;
process RSS includes other app work and prior sessions. Hardware/driver audio loss
is not measured. Skipped windows are inferred from sample-clock gaps.

Live Lite monitoring on the software CPU emulator visibly produced substantial
capture gaps; the UI reported them rather than claiming continuous coverage.
This is evidence that drop instrumentation/coverage reporting works, not a phone
performance estimate. ASR workloads are serialized with acoustic inference; on slow
devices they can delay acoustic work. Lite does not dynamically retune the acoustic
hop, and real-phone cadence/queue calibration remains necessary. Targets below
100/200/300MB incremental AI RAM are unmeasured goals, not guarantees.

Earlier host-only measurements remain in `benchmarks/`: YAMNet silence p50 1.505ms /
p95 2.055ms (100 runs; peak process RSS103,684KiB); Whisper Tiny int8 eight-second
silence transcription0.921s (peak process RSS251,344KiB). Vosk synthetic refusal
and coffee clips exercised host ASR→baseline→decision, yielding Level1 and Level0,
respectively. These workloads/host-process RSS values are not comparable phone
benchmarks and cannot select a production ASR model or establish safety accuracy.

### Verified anonymous metrics export on the emulator

The actual Activity was operated through the emulator UI: select Lite, opt in,
Start, then Export anonymous performance metrics and Save in Android DocumentsUI.
The saved JSON was pulled back and parsed successfully. This additionally verifies
the Storage Access Framework export path; the first immediate read raced the
asynchronous Activity result and saw an empty file, and the subsequent completed
read verified the JSON. The source app did not need to be changed for that test race.
Evidence: `verification/emulator-api26/metrics-lite-ui-export.json` and the active
UI screenshot. Snapshot: 31 YAMNet runs, mean wall latency
1032.2ms/max 2465.7ms, worker CPU mean
564.4ms, peak sampled PSS 45,598KiB,
process peak RSS 121,216KiB, 25 dropped blocks
(200,000 samples), estimated 37
skipped analysis windows. ASR runs were zero because Lite was selected. These are
software-emulator observations and are **not affordable-phone benchmarks or proof
of memory targets**. The substantial drops are recorded explicitly, not hidden. The subsequent actual
UI Pause/Stop operation also reported Monitoring stopped (ui-stop-check.txt).

## Remaining blockers before production or broad compatibility claims

- No licensed representative safety dataset, trained risk encoder, held-out labeled
  evaluation, calibrated thresholds, acoustic event F1/false-alerts per hour, real
  speech WER/CER or Hindi-English code-switching results.
- No physical-phone microphone/end-to-end spoken distress test, affordable-device
  performance matrix, long-duration battery/thermal test, permission revocation,
  calls/audio coexistence, lock-screen behavior, rapid restart or process-death test.
- API35 APK packaging is verified, but API35 runtime was not: its emulator's fresh
  userdata allocation exceeded workspace storage. API26 runtime passed after using
  a dedicated2GB userdata image. An earlier550MB userdata image could not install
  the APK (INSTALL_FAILED_INSUFFICIENT_STORAGE); no phone data was deleted.
- The 64-bit TFLite/Vosk libraries have4096-byte ELF load alignment. Although APK
  ZIP entries pass16KB alignment, the native libraries fail16KB ELF readiness.
  Rebuild/upgrade those libraries and validate16KB-page devices before claiming
  that compatibility. Only API26 x86_64/4KB-page runtime was tested here.
- Energy VAD can miss quiet speech. Speaker attribution, negation, quotations,
  background TV, nuanced coercion and calibrated contextual risk remain unvalidated.
- Main application's durable incident workflow, ESP32 physical transport,
  user-configured contacts/cancellation policy and optional encrypted evidence are
  not implemented in the demo. All notifications are local; no accusations or
  contact messages are sent.

## Reproduce and export

```bash
python scripts/prepare_android_assets.py
cd android
./gradlew :safora-ai-core:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest :app:lintDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r -t app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w -r org.safora.demo.test/androidx.test.runner.AndroidJUnitRunner
```

In this managed environment the standard ADB host daemon could not create its
home-directory files. The authorized local emulator was accessed using the normal
ADB protocol through `adb-shell==0.4.4`, with evidence recorded by
`scripts/run_device_smoke.py`. No microphone or Android permission restrictions
were bypassed; the test rule grants permissions on the dedicated test emulator.

See `docs/integration.md` for callbacks and lifecycle contracts. Package verification
is reproducible with `scripts/verify_apk.py`; export uses
`scripts/export_project.py /writable/deliverables`. Export includes signed debug
APK, instrumented-test APK, source ZIP, integration/test reports and checksums,
without private recordings, signing keys or fabricated trained safety weights.
The writable export directory for this task is `/workspace/output/`; it replaces
the read-only download directory encountered during the initial implementation.
