# SAFORA Android integration guide

## What the demo actually contains

Import `android/safora-ai-core` as a Gradle module. Its dependencies are Kotlin
coroutines Android 1.9.0, TensorFlow Lite 2.16.1, Vosk Android 0.3.47 and JNA
5.13.0. Compile SDK35, min SDK26, JVM target17; Gradle8.9 wrapper and AGP8.7.3
are included. The demo targets API35. Its ABI filters are arm64-v8a, armeabi-v7a,
x86 and x86_64. Other ABIs are deliberately excluded because all runtimes must
exist for each packaged ABI. The current TFLite/Vosk 64-bit ELF segments have4096-byte alignment and fail
static16KB readiness. Upgrade/rebuild those native libraries and validate before
claiming16KB-page compatibility; passing APK ZIP alignment is insufficient.

There are real pretrained YAMNet and Vosk ASR weights. **There are no trained
SAFORA conversation-risk weights.** `DevelopmentClassifier` is an unvalidated
contextual heuristic; it returns uncertain results, which can produce Level1
local check-ins only. The Python encoder training/loading/export infrastructure
is available for a properly licensed labeled dataset. Do not describe the demo
as a validated harassment detector or imply code-switching accuracy.

## Build and provision

```bash
python scripts/prepare_android_assets.py
export ANDROID_HOME=/path/to/android-sdk
cd android
./gradlew :safora-ai-core:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest :app:lintDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Use Android Studio's SDK Manager or `sdkmanager "platform-tools"
"platforms;android-35" "build-tools;35.0.0"`. A JDK17 or compatible newer JDK is
required (this environment built using OpenJDK21). On managed computers, set
`ANDROID_USER_HOME` and `GRADLE_USER_HOME` to writable task directories. Do not
copy this workspace's local SDK paths or proxy settings into your project.

Provisioning is an explicit development step, not an app network dependency.
`models/android-assets.json` pins YAMNet and English/Hindi Vosk ZIP hashes. The
Gradle pre-build task verifies all hashes and rejects missing/stale metadata.
The APK bundles these assets and notices; no download or external storage is
needed on the phone. Vosk archives are verified again and extracted atomically
to app-private `files/models/vosk-<language>-<hash>`, with traversal/size checks.
Only the selected language's native model is resident. Stop/restart to change
language. First-use extraction/load can cause capture drops; coverage reports
those gaps, and partial context is discarded. Test cold-start behavior on phones.

Run `scripts/verify_apk.py ... --build-tools /sdk/build-tools/35.0.0 --output report.json`
to validate assets, ABI completeness, APK signature and ZIP alignment. A successful
package check is not evidence of installation or runtime behavior.

## Connect to Member 1's incident manager

The demo shows an actual adapter in `MonitoringService.kt`, `LocalAlerts.kt` and
`SaforaDemoApp`. The application initializes local notifications once. The decision
engine's single sink handles acoustic, conversational and manual events; it is
not a second, disconnected callback attached only to the service.

Create an `IncidentSink`, then a `DecisionEngine(sink)` and an `AudioPipeline`:

```kotlin
val ui = android.os.Handler(android.os.Looper.getMainLooper())
val sink = object : org.safora.ai.IncidentSink {
    override fun onRisk(event: org.safora.ai.RiskEvent) {
        ui.post {
            // Replace with your existing host-app workflow. Do not send transcripts.
            incidentManager.verifyLocally(
                "${event.sessionId}:${event.id}", event.level,
                event.reason, event.manual, event.elapsedRealtimeMs
            )
        }
    }
    override fun onCoverage(status: String) { ui.post { monitoringUi.show(status) } }
}
val engine = org.safora.ai.DecisionEngine(sink)
val pipeline = org.safora.ai.AudioPipeline(context, engine, sink, serviceScope, "en")
// Call from a visible, opted-in microphone foreground-service flow after permission.
pipeline.start()
// Resource policy may enable optional ASR after observed capacity permits it.
pipeline.setConversationEnabled(true)
```

The `incidentManager` and `monitoringUi` above name your host integration objects;
they are not APIs supplied by this library. Callback threads may be worker threads.
Host code must dispatch UI work appropriately. Manual SOS should call
`engine.manual(SystemClock.elapsedRealtime())` directly from both the physical
ESP32 transport and app controls, without model/permission/confirmation dependencies.
The ESP32 transport itself is not implemented in this repository.

`RiskEvent` contains a session UUID, monotonically increasing per-engine ID,
level, reason labels, manual flag and elapsed-realtime timestamp. Use session+ID
as the in-process deduplication key. Timestamp is the **decision time**, not a
scream-onset annotation or wall-clock timestamp. The host owns durable incident
IDs, process-death recovery, user preferences, cancellation timers, verification,
configured contacts and persistence. Call `engine.resolve()` after resolution.
The demo never sends external messages or contacts anyone, including on manual SOS.

Level0 means no event. Level1 requests a discreet check-in; Level2 requests
incident verification. Scream confirmation uses experimental .30 / two-of-three /
10s defaults. Model scores are not calibrated probabilities of danger. Do not
convert uncertain baseline labels into accusations or automatic escalation.

## Lifecycle, audio gaps and privacy

Only one component owns AudioRecord. Do not create a second microphone recorder
for ASR. Capture uses a four-buffer pool and a two-block nonblocking channel;
pressure drops blocks explicitly instead of indefinitely blocking capture. Gaps
reset the audio ring/temporal history and discard partial speech/context. YAMNet
and bounded eight-second ASR chunks run serially. The unvalidated energy VAD is
independent of scream scores, but can miss quiet speech; replace it with a
validated VAD before claiming quiet-conversation coverage.

Use a microphone foreground service with RECORD_AUDIO and the appropriate FGS
permissions/type. Start from a visible activity after opt-in. Android14+ background
start restrictions remain in force. Show coverage continuously; never label Lite
mode as full conversation monitoring. Thermal/low-memory pressure closes optional
ASR. Audio focus interruptions stop the demo service; recovery is a visible restart.
Stopping immediately requests microphone stop and performs inference/model cleanup
off the main thread. Test revocation, calls, screen lock, OEM behavior and repeated
start/stop on actual phones.

No ordinary transcript/audio files are retained. Text context is transient and
limited to three prior chunks / 30 seconds; audio stays in bounded memory. Alerts
contain generic local check-in language, not transcripts. Notifications can be
unavailable when the user denies permission; the status screen reports that case.
The demo disables cloud backup/device transfer. Preserve equivalent controls when
merging into the main app, while reviewing the host's existing backup policy.
Incident evidence retention/encryption is intentionally absent. If the host adds
opt-in evidence, implement Keystore-protected AES-GCM, access checks and retention
rules separately before storing anything sensitive.

## Performance instrumentation

`pipeline.metrics.snapshot()` exposes YAMNet/ASR wall mean/max, worker-thread CPU
mean, run counts, app-dropped samples/blocks, estimated skipped analysis windows,
queue high-water, current/peak sampled process PSS and kernel peak process RSS.
The UI samples memory every five seconds and offers explicit JSON export through
the Storage Access Framework. Metrics contain no transcript, audio, risk labels,
identifiers or location. `sessionId` is part of incident callbacks, not metrics.

CPU counters measure the calling worker thread and exclude native worker threads;
wall time covers the native call. PSS is sampled, not a guaranteed instantaneous
peak; RSS high-water is process-wide and may include previous sessions. Zero RSS
means unavailable. Skipped windows are estimated from sample-clock gaps; app-drop
counts are exact, but device/driver losses are not measured. Do not equate zero
app drops with lossless microphone hardware. Targets <100/200/300MB incremental AI
RAM remain unmeasured device goals, not guarantees.

## Runtime verification

On a dedicated test device/emulator, install both APKs and run:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r -t app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w -r org.safora.demo.test/androidx.test.runner.AndroidJUnitRunner
```

The four instrumented checks exercise real bundled model loading, synthetic
speech→baseline→decision→local notification, manual SOS, and foreground microphone
capture/stop. They grant test permissions and programmatically opt in on the test
activity; use only an authorized test device. They are not accuracy or real-world
safety evaluation. `scripts/run_device_smoke.py` is an alternative for an authorized
TCP ADB endpoint (development dependency `adb-shell==0.4.4`). Record exact APK hashes
and device info. Physical-phone tests, representative labeled data, Hindi-English
code-switching, native 16KB pages and long-duration battery/thermal tests remain
separate requirements. Consult the current test report for what was actually run.
